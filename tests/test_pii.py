"""
tests/test_pii.py

Unit tests for Turkish PII Detection
"""

import pytest
from src.pii.turkish_detector import (
    TurkishPIIDetector,
    PIIType,
    scan_and_mask_text,
)


@pytest.fixture
def detector():
    return TurkishPIIDetector()


def test_tc_kimlik_valid(detector):
    # Algoritmik olarak geçerli TC kimlik
    text = "Kullanıcı TC: 10000000146 sisteme kaydedildi."
    matches = detector.scan_text(text)
    tc_matches = [m for m in matches if m.pii_type == PIIType.TC_KIMLIK]
    assert len(tc_matches) == 1
    assert tc_matches[0].value == "10000000146"


def test_tc_kimlik_invalid(detector):
    # 11 haneli ama checksum algoritmasını geçemeyen geçersiz numara
    text = "Geçersiz TC: 12345678900"
    matches = detector.scan_text(text)
    tc_matches = [m for m in matches if m.pii_type == PIIType.TC_KIMLIK]
    assert len(tc_matches) == 0


def test_phone_numbers(detector):
    text = "İletişim için 0555 123 45 67 veya +90 532 987 65 43 arayabilirsiniz."
    matches = detector.scan_text(text)
    tel_matches = [m for m in matches if m.pii_type == PIIType.TELEFON]
    assert len(tel_matches) >= 1


def test_email_detection(detector):
    text = "Bize iletisim@testlab.com adresinden ulaşabilirsiniz."
    matches = detector.scan_text(text)
    email_matches = [m for m in matches if m.pii_type == PIIType.EMAIL]
    assert len(email_matches) == 1
    assert email_matches[0].value == "iletisim@testlab.com"


def test_iban_detection(detector):
    text = "Banka hesabımız: TR330006100511123456789012"
    matches = detector.scan_text(text)
    iban_matches = [m for m in matches if m.pii_type == PIIType.IBAN]
    assert len(iban_matches) == 1


def test_scan_and_mask_convenience():
    text = "Ali e-posta: ali@example.com ve TC: 10000000146"
    masked, matches, summary = scan_and_mask_text(text)
    assert len(matches) == 2
    assert "ali@example.com" not in masked
    assert "10000000146" not in masked
    assert summary["email"] == 1
    assert summary["tc_kimlik"] == 1
