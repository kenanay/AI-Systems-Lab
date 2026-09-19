"""
tests/conftest.py

Pytest Configuration ve Test Fixtures

Bu modül tüm testler için ortak fixtures sağlar:
- Geçici dizinler (isolated test environment)
- Test database (in-memory SQLite)
- Test ayarları
- Sample dosyalar (TXT, MD)

Her test kendi izole environment'ta çalışır.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from backend.database import Base
from backend.config import Settings


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """
    Geçici test dizini oluştur.
    
    Scope: function - Her test için yeni dizin
    
    Yields:
        Path: Geçici dizin yolu
        
    Cleanup:
        Test sonunda dizin ve içeriği silinir
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def test_db() -> Generator[Session, None, None]:
    """
    Test database session oluştur.
    
    In-memory SQLite database kullanır.
    Her test kendi isolated database'i alır.
    
    Yields:
        Session: SQLAlchemy database session
        
    Cleanup:
        Session kapatılır
    """
    # In-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    TestSessionLocal = sessionmaker(bind=engine)
    session = TestSessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture(scope="function")
def test_settings(temp_dir: Path) -> Settings:
    """
    Test ortamı için Settings instance.
    
    Geçici dizinleri kullanır, production ayarlarını etkilemez.
    
    Args:
        temp_dir: Geçici test dizini fixture
        
    Returns:
        Settings: Test konfigürasyonu
    """
    return Settings(
        database_url="sqlite:///:memory:",
        data_root=temp_dir / "datasets",
        raw_data_path=temp_dir / "datasets" / "raw",
        processed_data_path=temp_dir / "datasets" / "normalized",
        temp_upload_dir=temp_dir / "temp",
    )


@pytest.fixture(scope="function")
def sample_text_file(temp_dir: Path) -> Path:
    """
    Test için örnek text dosyası.
    
    Content:
        - 3 satır Türkçe text
        - UTF-8 encoding
        
    Args:
        temp_dir: Geçici test dizini
        
    Returns:
        Path: Sample text dosyası yolu
    """
    file_path = temp_dir / "test_sample.txt"
    file_path.write_text("Bu bir test dosyasıdır.\nİkinci satır.\nÜçüncü satır.", encoding="utf-8")
    return file_path


@pytest.fixture(scope="function")
def sample_md_file(temp_dir: Path) -> Path:
    """
    Test için örnek markdown dosyası.
    
    Content:
        - Markdown formatting (header, bold, list)
        - UTF-8 encoding
        
    Args:
        temp_dir: Geçici test dizini
        
    Returns:
        Path: Sample markdown dosyası yolu
    """
    file_path = temp_dir / "test_sample.md"
    file_path.write_text(
        "# Test\n\nBu bir **test** dosyasıdır.\n\n- Item 1\n- Item 2",
        encoding="utf-8"
    )
    return file_path
