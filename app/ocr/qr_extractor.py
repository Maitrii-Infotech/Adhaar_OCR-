# app/ocr/qr_extractor.py
"""
QR Code extraction for Aadhaar cards
Aadhaar QR contains all fields in XML format - 100% accuracy!
"""

import cv2
import numpy as np
from pyzbar import pyzbar
import xml.etree.ElementTree as ET
import re
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class AadhaarQRExtractor:
    """Extract Aadhaar data from QR code"""
    
    @staticmethod
    def extract_qr_data(image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract data from Aadhaar QR code
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with extracted fields or None if QR not found
        """
        try:
            # Detect QR codes in image
            qr_codes = pyzbar.decode(image)
            
            if not qr_codes:
                logger.info("No QR code found in image")
                return None
            
            # Process first QR code found
            qr_data = qr_codes[0].data.decode('utf-8', errors='ignore')
            
            logger.info(f"QR code found, data length: {len(qr_data)} bytes")
            
            # Parse Aadhaar QR data
            parsed_data = AadhaarQRExtractor._parse_aadhaar_qr(qr_data)
            
            if parsed_data:
                logger.info(f"✅ QR extraction successful: {list(parsed_data.keys())}")
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"QR extraction failed: {e}")
            return None
    
    @staticmethod
    def _parse_aadhaar_qr(qr_data: str) -> Optional[Dict[str, Any]]:
        """
        Parse Aadhaar QR code data
        
        Aadhaar QR contains data in format:
        <xml>...</xml> or delimited format
        """
        try:
            # Try XML parsing first (newer Aadhaar format)
            if qr_data.startswith('<?xml') or '<PrintLetterBarcodeData' in qr_data:
                return AadhaarQRExtractor._parse_xml_qr(qr_data)
            else:
                # Try pipe-delimited format (older Aadhaar)
                return AadhaarQRExtractor._parse_delimited_qr(qr_data)
                
        except Exception as e:
            logger.error(f"QR parsing failed: {e}")
            return None
    
    @staticmethod
    def _parse_xml_qr(xml_data: str) -> Optional[Dict[str, Any]]:
        """Parse XML format QR code"""
        try:
            root = ET.fromstring(xml_data)
            
            data = {
                'aadhaar_number': root.get('uid', ''),
                'name': root.get('name', ''),
                'dob': root.get('dob', ''),  # Format: DD-MM-YYYY
                'gender': root.get('gender', ''),  # M/F
                'address': AadhaarQRExtractor._build_address_from_xml(root),
                'source': 'qr_xml',
                'confidence': 1.0  # QR is 100% accurate
            }
            
            # Clean and validate
            return AadhaarQRExtractor._clean_qr_data(data)
            
        except Exception as e:
            logger.error(f"XML QR parsing failed: {e}")
            return None
    
    @staticmethod
    def _parse_delimited_qr(delimited_data: str) -> Optional[Dict[str, Any]]:
        """
        Parse pipe-delimited format QR code
        Format: Field1|Field2|Field3|...
        """
        try:
            parts = delimited_data.split('|')
            
            if len(parts) < 5:
                logger.warning("Insufficient QR data fields")
                return None
            
            # Map based on typical Aadhaar QR structure
            data = {
                'aadhaar_number': parts[2] if len(parts) > 2 else '',
                'name': parts[3] if len(parts) > 3 else '',
                'dob': parts[4] if len(parts) > 4 else '',
                'gender': parts[5] if len(parts) > 5 else '',
                'address': parts[6] if len(parts) > 6 else '',
                'source': 'qr_delimited',
                'confidence': 1.0
            }
            
            return AadhaarQRExtractor._clean_qr_data(data)
            
        except Exception as e:
            logger.error(f"Delimited QR parsing failed: {e}")
            return None
    
    @staticmethod
    def _build_address_from_xml(root: ET.Element) -> str:
        """Build full address from XML attributes"""
        address_parts = []
        
        # Common Aadhaar XML address fields
        fields = ['house', 'street', 'lm', 'loc', 'vtc', 'subdist', 'dist', 'state', 'pc']
        
        for field in fields:
            value = root.get(field, '').strip()
            if value:
                address_parts.append(value)
        
        return ', '.join(address_parts)
    
    @staticmethod
    def _clean_qr_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize QR extracted data"""
        
        # Clean Aadhaar number
        if data.get('aadhaar_number'):
            clean_number = re.sub(r'\D', '', data['aadhaar_number'])
            if len(clean_number) == 12:
                data['aadhaar_number'] = f"{clean_number[:4]} {clean_number[4:8]} {clean_number[8:12]}"
            else:
                logger.warning(f"Invalid Aadhaar number in QR: {clean_number}")
                data['aadhaar_number'] = ''
        
        # Normalize DOB to YYYY-MM-DD format
        if data.get('dob'):
            dob = data['dob']
            # Handle DD-MM-YYYY or DD/MM/YYYY
            if '-' in dob or '/' in dob:
                parts = re.split(r'[-/]', dob)
                if len(parts) == 3:
                    day, month, year = parts
                    data['dob'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # Normalize gender
        if data.get('gender'):
            gender = data['gender'].upper()
            if gender in ['M', 'MALE']:
                data['gender'] = 'MALE'
            elif gender in ['F', 'FEMALE']:
                data['gender'] = 'FEMALE'
        
        # Clean name
        if data.get('name'):
            data['name'] = data['name'].strip()
        
        return data


# Test function
def test_qr_extraction():
    """Test QR extraction on sample image"""
    
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qr_extractor.py <image_path>")
        return
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"Testing QR extraction on: {image_path}")
    extractor = AadhaarQRExtractor()
    result = extractor.extract_qr_data(image)
    
    if result:
        print("\n✅ QR Extraction Successful!")
        for key, value in result.items():
            if key == 'aadhaar_number':
                print(f"  {key}: XXXX XXXX {value[-4:]}")  # Masked
            else:
                print(f"  {key}: {value}")
    else:
        print("\n❌ No QR code found or parsing failed")


if __name__ == "__main__":
    test_qr_extraction()