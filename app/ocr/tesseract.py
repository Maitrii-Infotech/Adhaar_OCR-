"""
Tesseract OCR engine implementation with Hindi + English support.
"""

import time
import logging
from typing import List, Dict, Any
import numpy as np

from .base import OCREngine, OCRResult

logger = logging.getLogger(__name__)

# Check for Tesseract availability
try:
    import pytesseract
    import cv2
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract dependencies not available")


class TesseractEngine(OCREngine):
    """Tesseract OCR implementation with Hindi + English support"""
    
    def __init__(self, languages: List[str] = None):
        super().__init__(languages)
        self.engine_name = "tesseract"
        
        # Default config
        self.config = r'--oem 3 --psm 6'
        
        # Multiple configs for better results
        self.configs = {
            'default': r'--oem 3 --psm 6',
            'mixed_text': r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'numbers_focus': r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789 /',
            'clean_text': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/-:., \'()'
        }
        
        # Language mapping for Tesseract
        self.lang_map = {
            "eng": "eng",
            "hin": "hin", 
            "hindi": "hin",
            "english": "eng"
        }
        
        # Verify availability on initialization
        if TESSERACT_AVAILABLE:
            self._verify_installation()
    
    def _verify_installation(self):
        """Verify Tesseract installation and available languages"""
        try:
            # Check Tesseract version
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            
            # Check available languages
            available_langs = pytesseract.get_languages(config='')
            logger.info(f"Available Tesseract languages: {available_langs}")
            
            # Check if required languages are available
            for lang in self.languages:
                mapped_lang = self.lang_map.get(lang, lang)
                if mapped_lang not in available_langs:
                    logger.warning(f"Language '{mapped_lang}' not available in Tesseract")
            
        except Exception as e:
            logger.error(f"Tesseract verification failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Tesseract is properly installed"""
        if not TESSERACT_AVAILABLE:
            return False
        
        try:
            # Try to get Tesseract version
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False
    
    def _prepare_languages(self) -> str:
        """Prepare language string for Tesseract"""
        tesseract_langs = []
        for lang in self.languages:
            mapped_lang = self.lang_map.get(lang, lang)
            tesseract_langs.append(mapped_lang)
        
        return "+".join(tesseract_langs)
    
    def _detect_languages(self, image: np.ndarray) -> List[str]:
        """Detect languages in the image"""
        try:
            # Get script confidence scores using OSD (Orientation and Script Detection)
            lang_string = self._prepare_languages()
            
            # Extract text to analyze character patterns
            text = pytesseract.image_to_string(image, lang=lang_string, config=self.config)
            
            detected_langs = []
            
            # Simple heuristic: if we find Devanagari-like characters, it's Hindi
            if any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in text):
                detected_langs.append("hin")
            
            # If we have ASCII characters, it's English
            if any(char.isascii() and char.isalpha() for char in text):
                detected_langs.append("eng")
            
            return detected_langs if detected_langs else ["eng"]
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return ["eng"]  # Default to English
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract OCR with multiple configuration attempts"""
        if not self.is_available():
            raise RuntimeError("Tesseract OCR is not available")
        
        start_time = time.time()
        
        try:
            # Prepare language string
            lang_string = self._prepare_languages()
            
            # Try multiple OCR configurations for better results
            configs_to_try = [
                ('mixed_text', self.configs['mixed_text']),
                ('default', self.configs['default']),
                ('numbers_focus', self.configs['numbers_focus']),
                ('clean_text', self.configs['clean_text'])
            ]
            
            best_result = None
            best_confidence = 0.0
            
            for config_name, config in configs_to_try:
                try:
                    # Extract text with current config
                    text = pytesseract.image_to_string(
                        image, 
                        lang=lang_string, 
                        config=config
                    ).strip()
                    
                    # Get detailed data for confidence calculation
                    data = pytesseract.image_to_data(
                        image, 
                        lang=lang_string, 
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Calculate confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    avg_confidence = avg_confidence / 100.0
                    
                    logger.debug(f"OCR config '{config_name}': {len(text)} chars, confidence {avg_confidence:.3f}")
                    
                    # Keep the best result
                    if avg_confidence > best_confidence and len(text) > 10:
                        best_confidence = avg_confidence
                        best_result = {
                            'text': text,
                            'data': data,
                            'config_name': config_name,
                            'confidence': avg_confidence
                        }
                
                except Exception as e:
                    logger.debug(f"OCR config '{config_name}' failed: {e}")
                    continue
            
            # Use best result or fallback
            if best_result:
                text = best_result['text']
                data = best_result['data']
                avg_confidence = best_result['confidence']
                config_used = best_result['config_name']
                logger.info(f"Best OCR result from config '{config_used}': {avg_confidence:.3f}")
            else:
                # Fallback to default if all failed
                logger.warning("All OCR configs failed, using default fallback")
                text = pytesseract.image_to_string(image, lang=lang_string, config=self.configs['default']).strip()
                data = pytesseract.image_to_data(image, lang=lang_string, config=self.configs['default'], output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                avg_confidence = avg_confidence / 100.0
                config_used = "fallback"
            
            # Detect languages
            detected_languages = self._detect_languages(image)
            
            # Extract bounding box data for advanced parsing
            bbox_data = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Filter low confidence
                    bbox_data.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]) / 100.0,
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i], 
                            'width': data['width'][i],
                            'height': data['height'][i]
                        },
                        'level': data['level'][i],
                        'word_num': data['word_num'][i]
                    })
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.debug(f"Final Tesseract result: {len(text)} characters, confidence {avg_confidence:.2f}, config: {config_used}")
            
            return OCRResult(
                text=text,
                confidence=avg_confidence,
                language_detected=detected_languages,
                processing_time_ms=processing_time,
                engine=f"tesseract-{lang_string}-{config_used}",
                bbox_data=bbox_data
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return OCRResult(
                text="",
                confidence=0.0,
                language_detected=[],
                processing_time_ms=processing_time,
                engine=f"tesseract-{self._prepare_languages()}-error",
                bbox_data=[]
            )
    
    def extract_text_simple(self, image: np.ndarray) -> str:
        """Simple text extraction without detailed metadata"""
        if not self.is_available():
            return ""
        
        try:
            lang_string = self._prepare_languages()
            return pytesseract.image_to_string(
                image, 
                lang=lang_string, 
                config=self.config
            ).strip()
        except Exception as e:
            logger.error(f"Simple Tesseract extraction failed: {e}")
            return ""
    
    def extract_with_boxes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text with bounding box information"""
        if not self.is_available():
            return []
        
        try:
            lang_string = self._prepare_languages()
            data = pytesseract.image_to_data(
                image, 
                lang=lang_string, 
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            boxes = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Include all detected text
                    boxes.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]) / 100.0,
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        },
                        'level': data['level'][i]
                    })
            
            return boxes
            
        except Exception as e:
            logger.error(f"Tesseract box extraction failed: {e}")
            return []


def create_tesseract_engine(languages: List[str] = None) -> TesseractEngine:
    """Factory function to create and configure Tesseract engine"""
    if languages is None:
        languages = ["eng", "hin"]
    
    engine = TesseractEngine(languages)
    
    if not engine.is_available():
        logger.warning("Tesseract OCR is not available. Please install tesseract and pytesseract.")
        logger.info("Installation instructions:")
        logger.info("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        logger.info("  Linux: sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin")
        logger.info("  macOS: brew install tesseract tesseract-lang")
        
    return engine


# Test function
def test_tesseract_engine():
    """Test Tesseract engine with a simple image"""
    
    if not TESSERACT_AVAILABLE:
        return {
            "available": False,
            "message": "Tesseract dependencies not installed",
            "install_command": "pip install pytesseract opencv-python"
        }
    
    engine = create_tesseract_engine()
    
    if not engine.is_available():
        return {
            "available": False,
            "dependencies_ok": True,
            "tesseract_binary": False,
            "message": "Tesseract binary not installed or not in PATH"
        }
    
    try:
        # Create a simple test image with text
        import numpy as np
        test_image = np.ones((100, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Add some simple text using OpenCV (if available)
        try:
            import cv2
            cv2.putText(test_image, "TEST 123", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        except ImportError:
            pass
        
        # Test OCR
        result = engine.extract_text(test_image)
        
        return {
            "available": True,
            "test_successful": True,
            "extracted_text": result.text,
            "confidence": result.confidence,
            "languages_detected": result.language_detected,
            "processing_time_ms": result.processing_time_ms,
            "engine": result.engine
        }
        
    except Exception as e:
        return {
            "available": True,
            "test_successful": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Run test if file is executed directly
    result = test_tesseract_engine()
    print("Tesseract engine test:", result)