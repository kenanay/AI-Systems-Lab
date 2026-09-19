"""
Database Bağlantı Yönetimi

SQLAlchemy engine ve session yapılandırması.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import logging

from backend.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy engine oluştur
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False  # SQL query'leri loglamak için True yapılabilir
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency.
    
    FastAPI endpoint'lerinde kullanılır:
    
    ```python
    @app.get("/items")
    def get_items(db: Session = Depends(get_db)):
        return db.query(Item).all()
    ```
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Database'i initialize et.
    
    Tüm tabloları oluşturur.
    """
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


def drop_db() -> None:
    """
    Tüm tabloları sil.
    
    **UYARI:** Production'da kullanılmamalı!
    Sadece development ve testing için.
    """
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All tables dropped")
