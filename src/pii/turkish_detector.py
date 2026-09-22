"""
src/pii/turkish_detector.py

Türkçe PII (Personally Identifiable Information) Detector

Bu modül Türkiye'ye özgü kişisel verileri tespit eder:
- TC Kimlik Numarası (11 haneli, algoritma doğrulaması)
- Telefon Numaraları (0xxx xxx xx xx formatları)
- E-posta Adresleri
- IBAN (TR ile başlayan)
- Kredi Kartı Numaraları (Luhn algoritması)

KVKK (Kişisel Verilerin Korunması Kanunu) uyumluluğu için kritik.

Kullanım:
    >>> detector = TurkishPIIDetector()
    >>> matches = detector.scan_text("TC: 12345678901, Tel: 0555 123 45 67")
    >>> for match in matches:
    ...     print(f"{match.pii_type}: {match.value} at position {match.start}")

Version: 1.0.0
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """PII türleri."""
    TC_KIMLIK = "tc_kimlik"
    TELEFON = "telefon"
    EMAIL = "email"
    IBAN = "iban"
    KREDI_KARTI = "kredi_karti"
    ADRES = "adres"
    ISIM = "isim"


@dataclass
class PIIMatch:
    """
    Tespit edilen PII bilgisi.
    
    Attributes:
        pii_type: PII türü (TC_KIMLIK, TELEFON, vb.)
        value: Tespit edilen değer
        start: Metindeki başlangıç pozisyonu
        end: Metindeki bitiş pozisyonu
        confidence: Güven skoru (0.0-1.0)
        masked_value: Maskelenmiş değer (örn: 123****8901)
    """
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float
    masked_value: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict formatına dönüştür."""
        return {
            'pii_type': self.pii_type.value,
            'value': self.value,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
            'masked_value': self.masked_value
        }


class TurkishPIIDetector:
    """
    Türkçe metinlerde PII tespiti için ana sınıf.
    
    Features:
    - TC Kimlik Numarası validasyonu (11 haneli algoritma)
    - Telefon numarası formatları (05xx, 02xx, 90, +90)
    - E-posta regex pattern matching
    - IBAN TR kontrolü ve validasyon
    - Kredi kartı Luhn algoritması
    """
    
    # Regex patterns
    TC_KIMLIK_PATTERN = r'\b[1-9]\d{10}\b'
    
    # Telefon patterns: 0555 123 45 67, 05551234567, +90 555 123 45 67, 90 555 123 45 67
    TELEFON_PATTERNS = [
        r'\+90\s?\d{3}\s?\d{3}\s?\d{2}\s?\d{2}',  # +90 555 123 45 67
        r'90\s?\d{3}\s?\d{3}\s?\d{2}\s?\d{2}',    # 90 555 123 45 67
        r'0\d{3}\s?\d{3}\s?\d{2}\s?\d{2}',        # 0555 123 45 67
        r'\(0\d{3}\)\s?\d{3}\s?\d{2}\s?\d{2}',    # (0555) 123 45 67
    ]
    
    # E-posta pattern
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # IBAN pattern (TR ile başlayan)
    IBAN_PATTERN = r'\bTR\d{24}\b'
    
    # Kredi kartı pattern (16 haneli, boşluk/tire ile ayrılmış olabilir)
    KREDI_KARTI_PATTERN = r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'
    
    def __init__(self, mask_length: int = 3) -> None:
        """
        Initialize detector.
        
        Args:
            mask_length: Maskeleme için kaç karakter gösterilecek (baştan ve sondan)
        """
        self.mask_length = mask_length
        logger.info("TurkishPIIDetector initialized")
    
    def scan_text(self, text: str, min_confidence: float = 0.7) -> List[PIIMatch]:
        """
        Metni tara ve PII'ları tespit et.
        
        Args:
            text: Taranacak metin
            min_confidence: Minimum güven skoru (bu değerin altındakiler döndürülmez)
            
        Returns:
            Tespit edilen PII listesi
        """
        matches: List[PIIMatch] = []
        
        # TC Kimlik tara
        matches.extend(self._scan_tc_kimlik(text))
        
        # Telefon tara
        matches.extend(self._scan_telefon(text))
        
        # E-posta tara
        matches.extend(self._scan_email(text))
        
        # IBAN tara
        matches.extend(self._scan_iban(text))
        
        # Kredi kartı tara
        matches.extend(self._scan_kredi_karti(text))
        
        # Confidence filtreleme
        matches = [m for m in matches if m.confidence >= min_confidence]
        
        # Pozisyona göre sırala
        matches.sort(key=lambda x: x.start)
        
        logger.info(f"PII scan completed: {len(matches)} matches found")
        return matches
    
    def _scan_tc_kimlik(self, text: str) -> List[PIIMatch]:
        """TC Kimlik Numarası tara ve valide et."""
        matches: List[PIIMatch] = []
        
        for match in re.finditer(self.TC_KIMLIK_PATTERN, text):
            tc_no = match.group(0)
            
            # Algoritma ile valide et
            if self._validate_tc_kimlik(tc_no):
                masked = self._mask_value(tc_no, self.mask_length)
                matches.append(PIIMatch(
                    pii_type=PIIType.TC_KIMLIK,
                    value=tc_no,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,  # Algoritma doğrulaması geçti
                    masked_value=masked
                ))
        
        return matches
    
    def _validate_tc_kimlik(self, tc_no: str) -> bool:
        """
        TC Kimlik Numarası algoritma doğrulaması.
        
        Kural:
        - 11 haneli
        - İlk hane 0 olamaz
        - 10. hane = (1. + 3. + 5. + 7. + 9.) * 7 - (2. + 4. + 6. + 8.) mod 10
        - 11. hane = (1. + 2. + ... + 10.) mod 10
        """
        if len(tc_no) != 11:
            return False
        
        if not tc_no.isdigit():
            return False
        
        if tc_no[0] == '0':
            return False
        
        digits = [int(d) for d in tc_no]
        
        # 10. hane kontrolü
        sum_odd = digits[0] + digits[2] + digits[4] + digits[6] + digits[8]
        sum_even = digits[1] + digits[3] + digits[5] + digits[7]
        check_10 = (sum_odd * 7 - sum_even) % 10
        
        if check_10 != digits[9]:
            return False
        
        # 11. hane kontrolü
        sum_first_10 = sum(digits[:10])
        check_11 = sum_first_10 % 10
        
        if check_11 != digits[10]:
            return False
        
        return True
    
    def _scan_telefon(self, text: str) -> List[PIIMatch]:
        """Telefon numarası tara."""
        matches: List[PIIMatch] = []
        
        for pattern in self.TELEFON_PATTERNS:
            for match in re.finditer(pattern, text):
                telefon = match.group(0)
                
                # Normalize et (sadece rakamlar)
                digits = re.sub(r'\D', '', telefon)
                
                # Türkiye telefon numarası kontrolü
                # 10 haneli (05xx xxx xx xx) veya 12 haneli (90 5xx xxx xx xx)
                if len(digits) == 10 or len(digits) == 12:
                    masked = self._mask_telefon(telefon)
                    matches.append(PIIMatch(
                        pii_type=PIIType.TELEFON,
                        value=telefon,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                        masked_value=masked
                    ))
        
        return matches
    
    def _scan_email(self, text: str) -> List[PIIMatch]:
        """E-posta adresi tara."""
        matches: List[PIIMatch] = []
        
        for match in re.finditer(self.EMAIL_PATTERN, text):
            email = match.group(0)
            masked = self._mask_email(email)
            matches.append(PIIMatch(
                pii_type=PIIType.EMAIL,
                value=email,
                start=match.start(),
                end=match.end(),
                confidence=0.9,
                masked_value=masked
            ))
        
        return matches
    
    def _scan_iban(self, text: str) -> List[PIIMatch]:
        """IBAN tara (TR ile başlayan)."""
        matches: List[PIIMatch] = []
        
        for match in re.finditer(self.IBAN_PATTERN, text):
            iban = match.group(0)
            
            # Basit validasyon: TR + 24 hane
            if len(iban) == 26 and iban.startswith('TR'):
                masked = self._mask_value(iban, self.mask_length)
                matches.append(PIIMatch(
                    pii_type=PIIType.IBAN,
                    value=iban,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    masked_value=masked
                ))
        
        return matches
    
    def _scan_kredi_karti(self, text: str) -> List[PIIMatch]:
        """Kredi kartı numarası tara ve Luhn algoritması ile valide et."""
        matches: List[PIIMatch] = []
        
        for match in re.finditer(self.KREDI_KARTI_PATTERN, text):
            card_no = match.group(0)
            
            # Normalize et (sadece rakamlar)
            digits_only = re.sub(r'\D', '', card_no)
            
            # 16 haneli olmalı ve Luhn algoritması geçmeli
            if len(digits_only) == 16 and self._validate_luhn(digits_only):
                masked = self._mask_kredi_karti(card_no)
                matches.append(PIIMatch(
                    pii_type=PIIType.KREDI_KARTI,
                    value=card_no,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    masked_value=masked
                ))
        
        return matches
    
    def _validate_luhn(self, card_number: str) -> bool:
        """
        Luhn algoritması ile kredi kartı validasyonu.
        
        https://en.wikipedia.org/wiki/Luhn_algorithm
        """
        def digits_of(n: str) -> List[int]:
            return [int(d) for d in n]
        
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(str(d * 2)))
        
        return checksum % 10 == 0
    
    def _mask_value(self, value: str, visible_length: int = 3) -> str:
        """Değeri maskele (baştan ve sondan belirli karakter göster)."""
        if len(value) <= visible_length * 2:
            return '*' * len(value)
        
        return value[:visible_length] + '*' * (len(value) - visible_length * 2) + value[-visible_length:]
    
    def _mask_telefon(self, telefon: str) -> str:
        """Telefon numarasını maskele."""
        # Sadece son 2 hanesi görünsün
        digits = re.sub(r'\D', '', telefon)
        masked_digits = '*' * (len(digits) - 2) + digits[-2:]
        
        # Orijinal formatı koru
        result = telefon
        for i, char in enumerate(telefon):
            if char.isdigit():
                if masked_digits:
                    result = result[:i] + masked_digits[0] + result[i+1:]
                    masked_digits = masked_digits[1:]
        
        return result
    
    def _mask_email(self, email: str) -> str:
        """E-posta adresini maskele."""
        parts = email.split('@')
        if len(parts) != 2:
            return email
        
        username, domain = parts
        
        # Username'in ilk 2 karakteri görünsün
        if len(username) <= 2:
            masked_username = '*' * len(username)
        else:
            masked_username = username[:2] + '*' * (len(username) - 2)
        
        return f"{masked_username}@{domain}"
    
    def _mask_kredi_karti(self, card_no: str) -> str:
        """Kredi kartı numarasını maskele (sadece son 4 hane görünsün)."""
        digits = re.sub(r'\D', '', card_no)
        masked_digits = '*' * (len(digits) - 4) + digits[-4:]
        
        # Orijinal formatı koru
        result = card_no
        for i, char in enumerate(card_no):
            if char.isdigit():
                if masked_digits:
                    result = result[:i] + masked_digits[0] + result[i+1:]
                    masked_digits = masked_digits[1:]
        
        return result
    
    def mask_text(self, text: str, matches: List[PIIMatch]) -> str:
        """
        Metindeki PII'ları maskele.
        
        Args:
            text: Orijinal metin
            matches: Tespit edilen PII'lar
            
        Returns:
            Maskelenmiş metin
        """
        # Tersine sırala ki indexler bozulmasın
        sorted_matches = sorted(matches, key=lambda x: x.start, reverse=True)
        
        masked_text = text
        for match in sorted_matches:
            masked_text = (
                masked_text[:match.start] + 
                match.masked_value + 
                masked_text[match.end:]
            )
        
        return masked_text
    
    def get_pii_summary(self, matches: List[PIIMatch]) -> Dict[str, int]:
        """
        PII özet istatistikleri.
        
        Returns:
            PII türlerine göre sayılar
        """
        summary: Dict[str, int] = {}
        
        for match in matches:
            pii_type = match.pii_type.value
            summary[pii_type] = summary.get(pii_type, 0) + 1
        
        return summary


def scan_and_mask_text(text: str, min_confidence: float = 0.7) -> Tuple[str, List[PIIMatch], Dict[str, int]]:
    """
    Convenience function: Metni tara, maskele ve özet döndür.
    
    Args:
        text: Taranacak metin
        min_confidence: Minimum güven skoru
        
    Returns:
        Tuple of (masked_text, matches, summary)
    """
    detector = TurkishPIIDetector()
    matches = detector.scan_text(text, min_confidence)
    masked_text = detector.mask_text(text, matches)
    summary = detector.get_pii_summary(matches)
    
    return masked_text, matches, summary
