# enhanced_ocr_configs.py
"""
Enhanced OCR configurations for better accuracy on Indian government documents
Add this to your app/ocr/ directory or integrate into existing modules
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from app.ocr.enhanced_ocr_configs import ImagePreprocessor
import cv2

logger = logging.getLogger(__name__)
# Load image
image = cv2.imread("your_aadhaar.png")

# Apply Aadhaar-specific enhancement
preprocessor = ImagePreprocessor()
enhanced = preprocessor.enhance_for_ocr(image, enhancement_type='aadhaar')

# Save to see the difference
cv2.imwrite("aadhaar_enhanced.png", enhanced)

class EnhancedOCRConfigs:
    """Enhanced OCR configurations for different document types and text regions"""
    
    # Tesseract configurations optimized for different text types
    TESSERACT_CONFIGS = {
        # For Aadhaar names (Hindi + English)
        'aadhaar_name': {
            'config': '--oem 3 --psm 8 -c preserve_interword_spaces=1',
            'lang': 'eng+hin',
            'whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
            'description': 'Optimized for names in Aadhaar cards'
        },
        
        # For Aadhaar numbers (digits only)
        'aadhaar_number': {
            'config': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',
            'lang': 'eng',
            'whitelist': '0123456789 ',
            'description': 'Optimized for 12-digit Aadhaar numbers'
        },
        
        # For dates
        'date_field': {
            'config': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789/',
            'lang': 'eng',
            'whitelist': '0123456789/',
            'description': 'Optimized for date fields (DD/MM/YYYY)'
        },
        
        # For gender field (mixed script)
        'gender_field': {
            'config': '--oem 3 --psm 8',
            'lang': 'eng+hin',
            'whitelist': 'MaleFemaleपुरुषमहिला',
            'description': 'Optimized for gender field (Male/Female/पुरुष/महिला)'
        },
        
        # For addresses (most flexible)
        'address_field': {
            'config': '--oem 3 --psm 6 -c preserve_interword_spaces=1',
            'lang': 'eng+hin',
            'whitelist': None,  # No whitelist for addresses
            'description': 'Optimized for address text blocks'
        },
        
        # High accuracy mode (slower but better)
        'high_accuracy': {
            'config': '--oem 3 --psm 6 -c tessedit_ocr_engine_mode=2',
            'lang': 'eng+hin',
            'whitelist': None,
            'description': 'High accuracy mode for difficult text'
        }
    }
    
    # PaddleOCR configurations
    PADDLEOCR_CONFIGS = {
        'default': {
            'use_textline_orientation': True,
            'lang': 'en',
            'description': 'Standard PaddleOCR configuration'
        },
        
        'multilingual': {
            'use_textline_orientation': True,
            'lang': 'ch',  # Chinese model often works better for mixed scripts
            'description': 'Multilingual configuration for Hindi+English'
        }
    }

class ImagePreprocessor:
    """Enhanced image preprocessing for better OCR accuracy"""
    
    @staticmethod
    def enhance_for_ocr(image: np.ndarray, enhancement_type: str = 'default') -> np.ndarray:
        """
        Apply enhancement specific to OCR requirements
        
        Args:
            image: Input image
            enhancement_type: Type of enhancement ('default', 'aadhaar', 'high_contrast')
        """
        
        if enhancement_type == 'aadhaar':
            return ImagePreprocessor._enhance_aadhaar(image)
        elif enhancement_type == 'high_contrast':
            return ImagePreprocessor._enhance_high_contrast(image)
        else:
            return ImagePreprocessor._enhance_default(image)
    
    @staticmethod
    def _enhance_aadhaar(image: np.ndarray) -> np.ndarray:
        """Specific enhancements for Aadhaar cards"""
        
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 1. Increase contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        # 2. Increase sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        # 3. Slight brightness adjustment
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Convert back to OpenCV format
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 4. Apply Gaussian blur reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 5. Morphological operations to clean up text
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1,1), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    @staticmethod
    def _enhance_high_contrast(image: np.ndarray) -> np.ndarray:
        """High contrast enhancement for difficult text"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (1,1), 0)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    @staticmethod
    def _enhance_default(image: np.ndarray) -> np.ndarray:
        """Default enhancement"""
        
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Moderate contrast enhancement
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Moderate sharpness enhancement
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.3)
        
        # Convert back
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    @staticmethod
    def extract_text_regions(image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract potential text regions from image
        
        Returns:
            List of dictionaries with region coordinates and cropped images
        """
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size (likely text regions)
            if 10 < w < image.shape[1] * 0.8 and 8 < h < image.shape[0] * 0.3:
                # Extract region
                region = image[y:y+h, x:x+w]
                
                text_regions.append({
                    'coordinates': (x, y, w, h),
                    'image': region,
                    'area': w * h
                })
        
        # Sort by area (largest regions first)
        text_regions.sort(key=lambda x: x['area'], reverse=True)
        
        return text_regions

class AccuracyTester:
    """Tools for measuring OCR accuracy"""
    
    @staticmethod
    def character_accuracy(predicted: str, actual: str) -> float:
        """Calculate character-level accuracy using Levenshtein distance"""
        
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(predicted.lower(), actual.lower())
        max_len = max(len(predicted), len(actual))
        
        if max_len == 0:
            return 1.0
        
        accuracy = 1 - (distance / max_len)
        return max(0.0, accuracy)
    
    @staticmethod
    def test_config_on_text(image: np.ndarray, ground_truth: str, 
                           config_name: str) -> Dict[str, Any]:
        """
        Test a specific OCR configuration on an image
        
        Returns:
            Dictionary with accuracy metrics and extracted text
        """
        
        try:
            # This would integrate with your existing OCR engines
            # For now, returning a template
            return {
                'config_name': config_name,
                'extracted_text': '',  # Would contain actual OCR result
                'character_accuracy': 0.0,
                'confidence': 0.0,
                'processing_time_ms': 0,
                'success': False,
                'error': 'Not implemented - integrate with your OCR engines'
            }
            
        except Exception as e:
            return {
                'config_name': config_name,
                'extracted_text': '',
                'character_accuracy': 0.0,
                'confidence': 0.0,
                'processing_time_ms': 0,
                'success': False,
                'error': str(e)
            }

# Test different configurations
def test_enhanced_configs(image_path: str, ground_truth: str) -> Dict[str, Any]:
    """
    Test multiple OCR configurations on an image and return best results
    
    Args:
        image_path: Path to image file
        ground_truth: Expected text output
        
    Returns:
        Dictionary with results from all tested configurations
    """
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {'error': f'Could not load image: {image_path}'}
    
    results = {}
    preprocessor = ImagePreprocessor()
    tester = AccuracyTester()
    
    # Test different preprocessing approaches
    preprocessing_methods = ['default', 'aadhaar', 'high_contrast']
    
    for method in preprocessing_methods:
        try:
            # Preprocess image
            enhanced_image = preprocessor.enhance_for_ocr(image, method)
            
            # Test with different OCR configs
            configs = EnhancedOCRConfigs.TESSERACT_CONFIGS
            
            for config_name, config_details in configs.items():
                test_key = f"{method}_{config_name}"
                
                # This would call your actual OCR engines with the config
                result = tester.test_config_on_text(enhanced_image, ground_truth, config_name)
                results[test_key] = result
                
        except Exception as e:
            results[f"{method}_error"] = str(e)
    
    return results

if __name__ == "__main__":
    # Example usage
    configs = EnhancedOCRConfigs()
    
    print("Available Tesseract configurations:")
    for name, config in configs.TESSERACT_CONFIGS.items():
        print(f"  {name}: {config['description']}")
    
    print("\nAvailable PaddleOCR configurations:")
    for name, config in configs.PADDLEOCR_CONFIGS.items():
        print(f"  {name}: {config['description']}")
