# Fixed diagnostic test - run from TEST directory (not app/ directory)
import sys
import os
import numpy as np
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

def test_basic_imports():
    """Test basic imports first"""
    print("=== Basic Import Test ===")
    
    try:
        import pytesseract
        print("✅ pytesseract imported")
        
        # Test tesseract binary
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
    except ImportError:
        print("❌ pytesseract not installed")
        return False
    except Exception as e:
        print(f"❌ Tesseract binary not available: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported")
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow imported")
    except ImportError:
        print("❌ Pillow not installed")
        return False
    
    return True

def test_ocr_with_simple_image():
    """Test OCR with a simple created image"""
    print("\n=== Simple OCR Test ===")
    
    try:
        import pytesseract
        import cv2
        
        # Create simple test image
        img = np.ones((150, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Add clear black text
        cv2.putText(img, "AADHAAR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "1234 5678 9012", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test basic OCR
        text = pytesseract.image_to_string(img, lang='eng').strip()
        print(f"OCR extracted text: {repr(text)}")
        
        if "1234" in text and "AADHAAR" in text:
            print("✅ Basic OCR working correctly")
            return True
        else:
            print("❌ OCR not extracting text properly")
            return False
            
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

def test_app_imports():
    """Test if app modules can be imported"""
    print("\n=== App Module Test ===")
    
    try:
        # Test OCR imports
        from app.ocr.base import OCRResult, OCREngine
        print("✅ OCR base classes imported")
        
        from app.ocr.tesseract import TesseractEngine
        print("✅ Tesseract engine imported")
        
        from app.ocr.manager import OCRManager
        print("✅ OCR manager imported")
        
    except ImportError as e:
        print(f"❌ OCR imports failed: {e}")
        return False
    
    try:
        # Test parser imports
        from app.parsers.base import BaseParser, ParseResult
        print("✅ Parser base classes imported")
        
        from app.parsers.aadhaar import AadhaarParser
        print("✅ Aadhaar parser imported")
        
        from app.parsers.registry import get_parser_registry
        print("✅ Parser registry imported")
        
    except ImportError as e:
        print(f"❌ Parser imports failed: {e}")
        return False
    
    return True

def test_ocr_manager():
    """Test OCR manager functionality"""
    print("\n=== OCR Manager Test ===")
    
    try:
        from app.ocr.manager import get_ocr_manager
        import cv2
        
        manager = get_ocr_manager()
        stats = manager.get_manager_stats()
        print(f"OCR Manager stats: {stats}")
        
        if stats['total_engines'] == 0:
            print("❌ No OCR engines available")
            return False
        
        # Test with simple image
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, "TEST 123", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        result = manager.extract_text(img)
        print(f"Manager OCR result: {repr(result.text)}")
        print(f"Confidence: {result.confidence:.3f}")
        
        if "TEST" in result.text or "123" in result.text:
            print("✅ OCR Manager working")
            return True
        else:
            print("❌ OCR Manager not extracting text")
            return False
            
    except Exception as e:
        print(f"❌ OCR Manager test failed: {e}")
        return False

def test_parser_system():
    """Test parser functionality"""
    print("\n=== Parser System Test ===")
    
    try:
        from app.parsers import parse_document, DocumentType
        
        # Test with clear Aadhaar-like text
        test_text = """
        Aadhaar
        Name: John Doe
        DOB: 25/02/2000
        Gender: MALE
        1234 5678 9012
        Address: 123 Main Street, Mumbai, Maharashtra, 400001
        """
        
        result = parse_document(DocumentType.AADHAAR, test_text)
        print(f"Parser confidence: {result.confidence_score:.3f}")
        print(f"Warnings: {result.warnings}")
        
        if result.fields and hasattr(result.fields, 'id_number') and result.fields.id_number:
            print(f"✅ Parser extracted ID: {result.fields.id_number.value}")
            return True
        else:
            print("❌ Parser didn't extract fields")
            return False
            
    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("OCR + Parser System Diagnostic")
    print("=" * 40)
    
    # Run tests
    basic_ok = test_basic_imports()
    simple_ocr_ok = test_ocr_with_simple_image() if basic_ok else False
    app_imports_ok = test_app_imports()
    ocr_manager_ok = test_ocr_manager() if app_imports_ok else False
    parser_ok = test_parser_system() if app_imports_ok else False
    
    print(f"\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Basic Dependencies: {'✅' if basic_ok else '❌'}")
    print(f"Simple OCR: {'✅' if simple_ocr_ok else '❌'}")
    print(f"App Imports: {'✅' if app_imports_ok else '❌'}")
    print(f"OCR Manager: {'✅' if ocr_manager_ok else '❌'}")
    print(f"Parser System: {'✅' if parser_ok else '❌'}")
    
    if all([basic_ok, simple_ocr_ok, app_imports_ok, ocr_manager_ok, parser_ok]):
        print("\n🎉 ALL SYSTEMS WORKING!")
        print("The issue is likely document quality or format.")
        print("\nSuggestions:")
        print("1. Try a clearer/higher resolution document")
        print("2. Try a different document type (PAN card)")
        print("3. Try manual image preprocessing")
    else:
        print("\n⚠️ SYSTEM ISSUES DETECTED")
        if not basic_ok:
            print("Install missing dependencies: pip install pytesseract opencv-python Pillow")
        if not app_imports_ok:
            print("Check that you're running from the TEST directory, not app/")

if __name__ == "__main__":
    main()