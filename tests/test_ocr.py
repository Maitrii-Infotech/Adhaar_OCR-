# Save as: test_with_high_contrast.py
import sys
import cv2

sys.path.append('.')  # Current directory
sys.path.append('..')  # Parent directory

from app.ocr.manager import get_ocr_manager
from app.ocr.enhanced_ocr_configs import ImagePreprocessor
from app.parsers import parse_document, DocumentType

# Load your Aadhaar image
image_path = r"Screenshot 2025-09-15 141754.png"
image = cv2.imread(image_path)

print("=== Testing with High Contrast Preprocessing ===\n")

# Apply high contrast preprocessing
preprocessor = ImagePreprocessor()
enhanced = preprocessor._enhance_high_contrast(image)

# Get OCR manager (disable preprocessing since we already enhanced the image)
manager = get_ocr_manager(preprocessing_mode='high_contrast', enable_preprocessing=False)

# Run OCR on enhanced image
ocr_result = manager.extract_text(enhanced)

print(f"OCR Confidence: {ocr_result.confidence:.3f}")
print(f"\nExtracted Text:\n{ocr_result.text}\n")
print("=" * 50)

# Parse with Aadhaar parser
parse_result = parse_document(DocumentType.AADHAAR, ocr_result.text)

print("\n=== Parsed Fields ===")
if parse_result.fields.name:
    print(f"✅ Name: {parse_result.fields.name.value} (confidence: {parse_result.fields.name.confidence:.2f})")
else:
    print("❌ Name: Not extracted")

if parse_result.fields.id_number:
    print(f"✅ Aadhaar: {parse_result.fields.id_number.value} (confidence: {parse_result.fields.id_number.confidence:.2f})")
else:
    print("❌ Aadhaar: Not extracted")

if parse_result.fields.dob:
    print(f"✅ DOB: {parse_result.fields.dob.value}")
else:
    print("❌ DOB: Not extracted")

if parse_result.fields.gender:
    print(f"✅ Gender: {parse_result.fields.gender.value}")
else:
    print("❌ Gender: Not extracted")

print(f"\nParser Confidence: {parse_result.confidence_score:.3f}")
print(f"Warnings: {parse_result.warnings}")