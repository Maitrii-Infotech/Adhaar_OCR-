# service_integration_test.py
"""
Test to find the disconnect between OCR modules and main service
"""

import sys
import os
sys.path.append('app')

def test_service_ocr_integration():
    """Test how your main service checks for OCR availability"""
    
    print("=== Testing Service OCR Integration ===")
    
    # Test 1: Check OCR Manager directly
    print("\n1. Testing OCR Manager directly:")
    try:
        from app.ocr.manager import get_ocr_manager, extract_text_from_image
        
        manager = get_ocr_manager()
        stats = manager.get_manager_stats()
        
        print(f"   Manager health: {stats['health']}")
        print(f"   Available engines: {stats['available_engines']}")
        print(f"   Total engines: {stats['total_engines']}")
        
        if stats['available_engines']:
            print("   ✅ OCR Manager says OCR is available")
        else:
            print("   ❌ OCR Manager says no engines available")
            
    except Exception as e:
        print(f"   ❌ OCR Manager test failed: {e}")
    
    # Test 2: Test how your service might be checking OCR
    print("\n2. Testing common OCR availability patterns:")
    
    # Pattern A: Direct import test
    try:
        from app.ocr.manager import OCRManager
        mgr = OCRManager()
        available = mgr.get_available_engines()
        print(f"   Pattern A - Direct OCRManager: {len(available)} engines")
        if available:
            print("   ✅ Pattern A says OCR available")
        else:
            print("   ❌ Pattern A says no OCR")
    except Exception as e:
        print(f"   ❌ Pattern A failed: {e}")
    
    # Pattern B: Check if your service uses a different import path
    print("\n3. Checking possible service import paths:")
    
    # Check if your service imports from a different location
    possible_paths = [
        'app.ocr',
        'ocr', 
        'app.ocr.manager',
        'ocr.manager'
    ]
    
    for path in possible_paths:
        try:
            module = __import__(path, fromlist=[''])
            print(f"   ✅ Can import: {path}")
            
            # Try to get OCR manager from this path
            if hasattr(module, 'get_ocr_manager'):
                mgr = module.get_ocr_manager()
                engines = mgr.get_available_engines()
                print(f"      -> {len(engines)} engines available via {path}")
            elif hasattr(module, 'OCRManager'):
                mgr = module.OCRManager()
                engines = mgr.get_available_engines()
                print(f"      -> {len(engines)} engines available via {path}")
                
        except Exception as e:
            print(f"   ❌ Cannot import {path}: {e}")
    
    # Test 3: Check your actual service code
    print("\n4. Looking for your service's OCR check:")
    
    # Try to find where your service checks for OCR
    service_files = [
        'app/api/routes.py',
        'app/main.py', 
        'app/api/__init__.py',
        'main.py',
        'routes.py'
    ]
    
    for file_path in service_files:
        if os.path.exists(file_path):
            print(f"   Found service file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for OCR-related imports and checks
                if 'OCR' in content or 'ocr' in content:
                    print(f"      -> Contains OCR references")
                    
                    # Look for specific patterns
                    if 'missing' in content and 'OCR' in content:
                        print("      -> Found 'missing OCR' logic!")
                        # Extract the relevant lines
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'missing' in line and ('OCR' in line or 'ocr' in line):
                                print(f"         Line {i+1}: {line.strip()}")
                                # Show context
                                for j in range(max(0, i-2), min(len(lines), i+3)):
                                    if j != i:
                                        print(f"         Line {j+1}: {lines[j].strip()}")
                                break
                        
            except Exception as e:
                print(f"      -> Error reading {file_path}: {e}")
        else:
            print(f"   File not found: {file_path}")
    
    # Test 4: Check if there's a different OCR availability function
    print("\n5. Testing PaddleOCR availability:")
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("   ✅ PaddleOCR can be initialized")
    except Exception as e:
        print(f"   ❌ PaddleOCR initialization failed: {e}")
    
    # Test 5: Create a simple OCR availability check
    print("\n6. Creating unified OCR availability check:")
    try:
        def check_ocr_availability():
            available_engines = []
            errors = []
            
            # Check Tesseract
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                available_engines.append("tesseract")
            except Exception as e:
                errors.append(f"tesseract: {e}")
            
            # Check PaddleOCR
            try:
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
                available_engines.append("paddleocr")
            except Exception as e:
                errors.append(f"paddleocr: {e}")
            
            return {
                "available": len(available_engines) > 0,
                "engines": available_engines,
                "errors": errors
            }
        
        availability = check_ocr_availability()
        print(f"   Unified check result: {availability}")
        
        if availability["available"]:
            print("   ✅ Unified check says OCR is available")
        else:
            print("   ❌ Unified check says no OCR available")
            
    except Exception as e:
        print(f"   ❌ Unified check failed: {e}")

if __name__ == "__main__":
    test_service_ocr_integration()