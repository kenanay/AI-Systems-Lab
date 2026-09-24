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
    import backend.models  # noqa: F401
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


def pytest_sessionfinish(session, exitstatus):
    """Capture final exit status before unconfigure."""
    session.config._final_exitstatus = exitstatus


def pytest_unconfigure(config):
    """
    Safely exit process to prevent C++ thread pool destructors (PyTorch / ONNX / OpenMP)
    from calling std::terminate() during Python interpreter teardown (Py_Finalize).
    """
    import sys
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    exitstatus = getattr(config, "_final_exitstatus", 0)
    if hasattr(exitstatus, "value"):
        exitstatus = exitstatus.value
    os._exit(int(exitstatus))


@pytest.fixture(autouse=True)
def authenticated_legacy_feature_tests(request):
    """Feature suites run as an admin; security suites exercise real login and denial."""
    if request.node.path.name in {'test_auth_and_security.py', 'test_reliability.py'}:
        yield
        return
    from types import SimpleNamespace
    from backend.main import app
    from backend.security.dependencies import get_current_user
    old = app.dependency_overrides.get(get_current_user)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id='test-admin', role='admin')
    try:
        yield
    finally:
        if old is None:
            app.dependency_overrides.pop(get_current_user, None)
        else:
            app.dependency_overrides[get_current_user] = old


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Birim testler arasında rate limiter durumunu sıfırlar."""
    try:
        from backend.security.rate_limiter import global_rate_limiter
        global_rate_limiter.reset()
    except Exception:
        pass
    yield
    try:
        from backend.security.rate_limiter import global_rate_limiter
        global_rate_limiter.reset()
    except Exception:
        pass

