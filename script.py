# Save as: test_tesseract_psm.py
import cv2
import pytesseract
from app.ocr.enhanced_ocr_configs import ImagePreprocessor

# Load and enhance image
image = cv2.imread("Screenshot 2025-09-15 141754.png")
preprocessor = ImagePreprocessor()
enhanced = preprocessor._enhance_high_contrast(image)

print("Testing different Tesseract PSM modes:\n")
print("=" * 60)

# Test different PSM modes
psm_modes = {
    3: "Fully automatic page segmentation (default)",
    4: "Assume single column of text",
    6: "Assume uniform block of text",
    11: "Sparse text - find as much as possible",
    12: "Sparse text with OSD (Orientation and Script Detection)"
}

for psm, description in psm_modes.items():
    print(f"\nPSM {psm}: {description}")
    print("-" * 60)
    
    config = f'--oem 3 --psm {psm}'
    
    try:
        text = pytesseract.image_to_string(enhanced, lang='eng+hin', config=config)
        
        # Check what was extracted
        has_number = '4389' in text or '9349' in text or '1869' in text
        has_dob = '01/01/1986' in text or '01011986' in text
        has_gender = 'Female' in text or 'महिला' in text
        
        print(f"Length: {len(text)} chars")
        print(f"✅ Aadhaar Number: {'YES' if has_number else 'NO'}")
        print(f"✅ DOB: {'YES' if has_dob else 'NO'}")
        print(f"✅ Gender: {'YES' if has_gender else 'NO'}")
        print(f"\nFirst 200 chars:\n{text[:200]}")
        
        if has_number and has_dob and has_gender:
            print("\n🎉 THIS PSM MODE WORKS BEST!")
            print(f"\nFull extracted text:\n{text}")
            break
            
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 60)