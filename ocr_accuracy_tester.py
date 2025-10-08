# ocr_accuracy_tester.py
"""
Practical OCR accuracy testing that integrates with your existing system
Save this file and run it to test different OCR configurations
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging
import time
from typing import Dict, List, Any, Tuple

sys.path.append('.')  # Current directory
sys.path.append('..')  # Parent directory

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class OCRAccuracyTester:
    """Test different OCR configurations for accuracy improvement"""
    
    def __init__(self):
        self.ocr_manager = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize your existing OCR system"""
        try:
            from app.ocr.manager import get_ocr_manager
            self.ocr_manager = get_ocr_manager()
            logger.info(f"OCR Manager initialized with engines: {self.ocr_manager.get_available_engines()}")
        except Exception as e:
            logger.error(f"Failed to initialize OCR manager: {e}")
            self.ocr_manager = None
    
    def enhance_image_for_aadhaar(self, image: np.ndarray) -> np.ndarray:
        """Enhance image specifically for Aadhaar card OCR"""
        
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 1. Increase contrast significantly
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.8)
        
        # 2. Increase sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.2)
        
        # 3. Slight brightness adjustment  
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.15)
        
        # Convert back to OpenCV format
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 4. Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 5. Convert to grayscale and apply CLAHE
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 6. Apply morphological operations to clean up text
        kernel = np.ones((1,1), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR for compatibility
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def test_tesseract_configs(self, image: np.ndarray) -> Dict[str, Any]:
        """Test different Tesseract configurations"""
        
        if not self.ocr_manager:
            return {"error": "OCR manager not available"}
        
        # Different Tesseract configurations to test
        test_configs = {
            'current_default': {
                'description': 'Current default configuration',
                'preprocessing': None
            },
            'enhanced_aadhaar': {
                'description': 'Enhanced preprocessing + mixed language',
                'preprocessing': 'aadhaar_enhancement'
            },
            'high_contrast': {
                'description': 'High contrast preprocessing',
                'preprocessing': 'high_contrast'
            },
            'grayscale_only': {
                'description': 'Simple grayscale conversion',
                'preprocessing': 'grayscale'
            }
        }
        
        results = {}
        
        for config_name, config_details in test_configs.items():
            try:
                logger.info(f"Testing configuration: {config_name}")
                
                # Apply preprocessing if specified
                test_image = image.copy()
                
                if config_details['preprocessing'] == 'aadhaar_enhancement':
                    test_image = self.enhance_image_for_aadhaar(test_image)
                elif config_details['preprocessing'] == 'high_contrast':
                    test_image = self._apply_high_contrast(test_image)
                elif config_details['preprocessing'] == 'grayscale':
                    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                    test_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # Run OCR with your existing system
                start_time = time.time()
                ocr_result = self.ocr_manager.extract_text(test_image)
                processing_time = (time.time() - start_time) * 1000
                
                results[config_name] = {
                    'description': config_details['description'],
                    'extracted_text': ocr_result.text,
                    'confidence': ocr_result.confidence,
                    'engine': ocr_result.engine,
                    'processing_time_ms': processing_time,
                    'character_count': len(ocr_result.text),
                    'success': True
                }
                
                logger.info(f"  Result: '{ocr_result.text[:50]}...' (confidence: {ocr_result.confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Configuration {config_name} failed: {e}")
                results[config_name] = {
                    'description': config_details['description'],
                    'extracted_text': '',
                    'confidence': 0.0,
                    'engine': 'error',
                    'processing_time_ms': 0,
                    'character_count': 0,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def _apply_high_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply high contrast enhancement"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def calculate_accuracy_metrics(self, extracted_text: str, expected_fields: Dict[str, str]) -> Dict[str, float]:
        """Calculate accuracy metrics against expected fields"""
        
        metrics = {}
        
        for field_name, expected_value in expected_fields.items():
            if expected_value.lower() in extracted_text.lower():
                # Exact match found
                metrics[f"{field_name}_found"] = 1.0
            else:
                # Calculate character similarity
                similarity = self._calculate_similarity(extracted_text, expected_value)
                metrics[f"{field_name}_similarity"] = similarity
        
        # Overall text length comparison
        if expected_fields:
            total_expected_chars = sum(len(v) for v in expected_fields.values())
            extracted_chars = len(extracted_text)
            metrics['text_length_ratio'] = min(extracted_chars / max(total_expected_chars, 1), 2.0)
        
        return metrics
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple character overlap"""
        
        text1_clean = ''.join(c.lower() for c in text1 if c.isalnum())
        text2_clean = ''.join(c.lower() for c in text2 if c.isalnum())
        
        if not text2_clean:
            return 0.0
        
        # Count character overlap
        overlap = 0
        for char in text2_clean:
            if char in text1_clean:
                overlap += 1
                text1_clean = text1_clean.replace(char, '', 1)  # Remove one occurrence
        
        similarity = overlap / len(text2_clean)
        return similarity
    
    def test_on_aadhaar_image(self, image_path: str, expected_data: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Test OCR accuracy on an Aadhaar card image
        
        Args:
            image_path: Path to Aadhaar card image
            expected_data: Dictionary with expected values (optional)
                          e.g., {'name': 'Sanju Devi', 'number': '4389 9349 1869'}
        """
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Could not load image: {image_path}'}
        
        logger.info(f"Testing OCR accuracy on: {image_path}")
        logger.info(f"Image size: {image.shape}")
        
        # Test different configurations
        config_results = self.test_tesseract_configs(image)
        
        # Calculate accuracy metrics if expected data provided
        if expected_data:
            for config_name, result in config_results.items():
                if result.get('success', False):
                    accuracy_metrics = self.calculate_accuracy_metrics(
                        result['extracted_text'], expected_data
                    )
                    result['accuracy_metrics'] = accuracy_metrics
        
        # Find best configuration
        best_config = None
        best_score = 0
        
        for config_name, result in config_results.items():
            if result.get('success', False):
                # Score based on confidence and character count
                score = result['confidence'] * 0.7 + (min(result['character_count'], 100) / 100) * 0.3
                
                if score > best_score:
                    best_score = score
                    best_config = config_name
        
        return {
            'image_path': image_path,
            'image_size': image.shape,
            'config_results': config_results,
            'best_config': best_config,
            'best_score': best_score,
            'expected_data': expected_data
        }

def main():
    """Main testing function"""
    
    tester = OCRAccuracyTester()
    
    if not tester.ocr_manager:
        print("ERROR: Could not initialize OCR manager. Check your OCR setup.")
        return
    
    print("OCR Accuracy Tester")
    print("=" * 50)
    print(f"Available OCR engines: {tester.ocr_manager.get_available_engines()}")
    print()
    
    # Test with a sample image (you'll need to provide the path)
    test_image_path = input("Enter path to Aadhaar image (or press Enter to skip): ").strip()
    
    if test_image_path and os.path.exists(test_image_path):
        
        # Get expected data (optional)
        print("\nOptional: Enter expected values for accuracy measurement")
        expected_name = input("Expected name (or press Enter to skip): ").strip()
        expected_number = input("Expected Aadhaar number (or press Enter to skip): ").strip()
        
        expected_data = {}
        if expected_name:
            expected_data['name'] = expected_name
        if expected_number:
            expected_data['number'] = expected_number
        
        # Run test
        results = tester.test_on_aadhaar_image(
            test_image_path, 
            expected_data if expected_data else None
        )
        
        # Display results
        print("\nTest Results:")
        print("=" * 50)
        
        for config_name, result in results['config_results'].items():
            print(f"\nConfiguration: {config_name}")
            print(f"Description: {result['description']}")
            print(f"Success: {result['success']}")
            
            if result['success']:
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Processing time: {result['processing_time_ms']:.1f}ms")
                print(f"Characters extracted: {result['character_count']}")
                print(f"Engine: {result['engine']}")
                print(f"Text preview: '{result['extracted_text'][:100]}...'")
                
                if 'accuracy_metrics' in result:
                    print("Accuracy metrics:")
                    for metric, value in result['accuracy_metrics'].items():
                        print(f"  {metric}: {value:.3f}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        if results['best_config']:
            print(f"\nBest configuration: {results['best_config']} (score: {results['best_score']:.3f})")
        
    else:
        print("No test image provided or file not found.")
        print("You can test with your own Aadhaar image by running this script again.")
    
    print("\nNext steps:")
    print("1. Try different images with various qualities")
    print("2. Note which configuration works best for your documents")
    print("3. Integrate the best preprocessing into your main OCR pipeline")

if __name__ == "__main__":
    main()
