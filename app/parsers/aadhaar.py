"""
Aadhaar document parser with QR code support and Verhoeff validation.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import logging

from app.schemas.base import DocumentType, FieldValue, AddressField, AddressComponents
from .base import BaseParser, DateParser, NameParser

logger = logging.getLogger(__name__)


class AadhaarParser(BaseParser):
    """Parser for Aadhaar cards with advanced validation"""
    
    def _get_document_type(self) -> DocumentType:
        return DocumentType.AADHAAR
    
    def _get_field_patterns(self) -> Dict[str, List[str]]:
        """Regex patterns for Aadhaar field extraction"""
        return {
            "aadhaar_number": [
                # NEW: More flexible patterns for various OCR outputs
                r'(\d{4}\s+\d{4}\s+\d{4})',  # Standard spaced: 1234 5678 9012
                r'(\d{4}\s*\d{4}\s*\d{4})',  # Variable spacing
                r'(\d{12})',  # Continuous: 123456789012
                r'(?:Aadhaar|UID|आधार)\s*(?:No|Number|#|नंबर)?\s*:?\s*(\d{4}\s+\d{4}\s+\d{4})',
                r'(?:Aadhaar|UID|आधार)\s*(?:No|Number|#|नंबर)?\s*:?\s*(\d{12})',
                # NEW: Pattern for numbers that might have OCR noise
                r'(\d\s*\d\s*\d\s*\d\s+\d\s*\d\s*\d\s*\d\s+\d\s*\d\s*\d\s*\d)',
            ],
            "name": [
                # Existing patterns
                r'(?:Name|नाम)\s*:?\s*([A-Za-z\s\.]{3,50})',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # Proper name pattern
                # NEW: More flexible pattern for names after Hindi text
                r'(?:संजू|देवी)?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            ],
            "dob": [
                r'(?:DOB|Date\s+of\s+Birth|जन्म\s+तिथि)\s*[:/]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
                r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',  # Generic date
            ],
            "gender": [
                r'(?:Gender|Sex|लिंग|महिला|पुरुष)\s*:?\s*(Male|Female|M|F|महिला|पुरुष|MALE|FEMALE)',
                r'\b(Male|Female|MALE|FEMALE|महिला|पुरुष)\b',
            ],
            "address": [
                r'(?:Address|पता)\s*:?\s*([A-Za-z0-9\s\,\.\-\/]{15,200})',
                r'([A-Za-z0-9\s\,\.\-\/]*\d{6}[A-Za-z0-9\s\,\.\-\/]*)',  # Pattern with pincode
            ]
        }
    
    def _get_validation_rules(self) -> Dict[str, callable]:
        """Validation functions for Aadhaar fields"""
        return {
            "aadhaar_number": self._validate_aadhaar_number,
            "name": NameParser.normalize_name,
            "dob": DateParser.parse_date,
            "gender": self._validate_gender,
            "address": self._validate_address,
        }
    
    def _extract_fields(self, text: str) -> Dict[str, Any]:
        """
        Override extract_fields to add special handling for Aadhaar numbers
        """
        # First, try standard extraction
        extracted = super()._extract_fields(text)
        
        # NEW: If Aadhaar number not found, try aggressive extraction
        if "aadhaar_number" not in extracted:
            aadhaar_aggressive = self._extract_aadhaar_aggressive(text)
            if aadhaar_aggressive:
                extracted["aadhaar_number"] = aadhaar_aggressive
        
        return extracted
    
    def _extract_aadhaar_aggressive(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Aggressive Aadhaar number extraction for difficult OCR cases
        
        Looks for any sequence of 12 digits, even with noise/spaces
        """
        # Remove all non-digit and non-space characters
        cleaned = re.sub(r'[^\d\s]', '', text)
        
        # Find all sequences of digits with optional spaces
        digit_sequences = re.findall(r'\d[\s\d]{10,}\d', cleaned)
        
        for sequence in digit_sequences:
            # Extract only digits
            digits_only = re.sub(r'\s', '', sequence)
            
            if len(digits_only) == 12:
                # Found a 12-digit sequence
                logger.info(f"Aggressive extraction found potential Aadhaar: {digits_only[:4]}****{digits_only[-4:]}")
                return {
                    "raw_value": f"{digits_only[:4]} {digits_only[4:8]} {digits_only[8:12]}",
                    "pattern_used": "aggressive_extraction",
                    "confidence": 0.6  # Lower confidence for aggressive extraction
                }
        
        return None
    
    def _validate_aadhaar_number(self, aadhaar_str: str) -> Tuple[bool, str, float]:
        """
        Validate Aadhaar number using Verhoeff algorithm
        
        Returns:
            Tuple of (is_valid, masked_number, confidence_score)
        """
        # Clean the number - remove all spaces and non-digits
        clean_number = re.sub(r'\D', '', aadhaar_str.strip())
        
        if len(clean_number) != 12:
            return False, aadhaar_str, 0.0
        
        if not clean_number.isdigit():
            return False, aadhaar_str, 0.0
        
        # Verhoeff algorithm validation
        if self._verhoeff_validate(clean_number):
            # Mask the number (show only last 4 digits)
            masked = f"XXXX XXXX {clean_number[-4:]}"
            logger.info(f"Aadhaar validated successfully: {masked}")
            return True, masked, 1.0
        else:
            # Invalid checksum but could be OCR error
            masked = f"XXXX XXXX {clean_number[-4:]}"
            logger.warning(f"Aadhaar checksum failed but keeping: {masked}")
            return False, masked, 0.6
    
    def _verhoeff_validate(self, number: str) -> bool:
        """
        Validate Aadhaar number using Verhoeff checksum algorithm
        
        Args:
            number: 12-digit Aadhaar number as string
            
        Returns:
            True if valid checksum, False otherwise
        """
        # Verhoeff algorithm multiplication table
        d = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        ]
        
        # Verhoeff algorithm permutation table
        p = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
            [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
            [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
            [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
            [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
            [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
            [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
        ]
        
        try:
            # Convert to list of integers
            digits = [int(d) for d in number]
            
            # Calculate checksum
            checksum = 0
            for i, digit in enumerate(reversed(digits)):
                checksum = d[checksum][p[(i % 8)][digit]]
            
            return checksum == 0
            
        except (ValueError, IndexError):
            return False
    
    def _validate_gender(self, gender_str: str) -> Tuple[bool, str, float]:
        """Validate and normalize gender"""
        gender_str = gender_str.strip().upper()
        
        # Handle both English and Hindi
        if gender_str in ['M', 'MALE', 'पुरुष']:
            return True, 'MALE', 0.95
        elif gender_str in ['F', 'FEMALE', 'महिला']:
            return True, 'FEMALE', 0.95
        else:
            return False, gender_str, 0.3
    
    def _validate_address(self, address_str: str) -> Tuple[bool, str, float]:
        """Validate and parse address"""
        address_str = address_str.strip()
        
        # Check if address has minimum required components
        if len(address_str) < 15:
            return False, address_str, 0.2
        
        # Look for pincode (6 digits)
        pincode_match = re.search(r'\b(\d{6})\b', address_str)
        has_pincode = pincode_match is not None
        
        # Basic validation - should have letters, numbers, and be reasonable length
        if re.search(r'[A-Za-z]', address_str) and len(address_str) >= 15:
            confidence = 0.8 if has_pincode else 0.6
            return True, address_str, confidence
        
        return False, address_str, 0.3
    
    def _parse_address_components(self, address_str: str) -> AddressComponents:
        """Parse address into structured components"""
        components = AddressComponents()
        
        # Extract pincode
        pincode_match = re.search(r'\b(\d{6})\b', address_str)
        if pincode_match:
            components.pincode = pincode_match.group(1)
        
        # Extract state (common Indian states)
        states = [
            'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
            'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
            'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
            'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
            'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
            'Delhi', 'Puducherry'
        ]
        
        for state in states:
            if state.lower() in address_str.lower():
                components.state = state
                break
        
        # Extract city (word before state or near pincode)
        if components.state:
            city_pattern = r'([A-Za-z\s]+?)(?:,\s*)?' + re.escape(components.state)
            city_match = re.search(city_pattern, address_str, re.IGNORECASE)
            if city_match:
                potential_city = city_match.group(1).strip().split(',')[-1].strip()
                if len(potential_city) > 2:
                    components.city = potential_city
        
        return components
    
    def _create_field_values(self, validated_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Override to handle Aadhaar-specific field mapping and address parsing"""
        field_values = {}
        
        for field_name, field_data in validated_fields.items():
            if field_name == "aadhaar_number":
                # Map to standard id_number and set masked flag
                field_values["id_number"] = FieldValue(
                    value=field_data["value"],
                    confidence=field_data["confidence"],
                    validated=field_data["validated"],
                    source=field_data["source"],
                    masked=True  # Aadhaar numbers are always masked
                )
            elif field_name == "address":
                # Parse address components
                components = self._parse_address_components(field_data["value"])
                field_values["address"] = AddressField(
                    value=field_data["value"],
                    confidence=field_data["confidence"],
                    validated=field_data["validated"],
                    source=field_data["source"],
                    components=components
                )
            else:
                # Standard field
                field_values[field_name] = FieldValue(
                    value=field_data["value"],
                    confidence=field_data["confidence"],
                    validated=field_data["validated"],
                    source=field_data["source"]
                )
        
        return field_values
def _preprocess_text(self, text: str) -> str:
    """Override to clean Aadhaar-specific OCR noise"""
    clean_text = super()._preprocess_text(text)
    
    # Remove common OCR noise at the beginning (before actual content)
    lines = clean_text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip very short noise lines at the start
        if len(line) <= 3 and not any(c.isdigit() for c in line):
            continue
        # Skip single words that are obvious noise
        if line.lower() in ['aa', 'fir', 'cd', 'oe', 'ee', 'sa']:
            continue
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def create_aadhaar_parser() -> AadhaarParser:
    """Factory function to create Aadhaar parser"""
    return AadhaarParser()


# Test function
def test_aadhaar_parser():
    """Test Aadhaar parser with sample data"""
    parser = create_aadhaar_parser()
    
    # Sample Aadhaar text (simulated OCR output)
    sample_text = """
    GOVERNMENT OF INDIA
    AADHAAR
    
    Name: Rahul Kumar Sharma
    DOB: 21/07/1993
    Gender: MALE
    
    2314 5678 9012
    
    Address: House No 123, MG Road, Sector 15
    Jaipur, Rajasthan - 302019
    """
    
    result = parser.parse(sample_text)
    
    return {
        "confidence": result.confidence_score,
        "fields_extracted": len(result.fields.__dict__) if result.fields else 0,
        "warnings": result.warnings,
        "sample_field": getattr(result.fields, 'name', None) if result.fields else None
    }


if __name__ == "__main__":
    # Run test if file is executed directly
    test_result = test_aadhaar_parser()
    print("Aadhaar parser test:", test_result)