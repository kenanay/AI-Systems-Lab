"""
Database Bağlantı Yönetimi

SQLAlchemy engine ve session yapılandırması.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import Generator
import logging

from backend.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy engine oluştur
engine = create_engine(
    settings.database_url,
    connect_args={
        "check_same_thread": False,
        # Enable WAL mode for better concurrency
        "timeout": 30.0  # 30 second timeout for locks
    } if "sqlite" in settings.database_url else {},
    echo=False,  # SQL query'leri loglamak için True yapılabilir
    pool_pre_ping=True  # Verify connections before using
)

# Enable WAL mode for SQLite
if "sqlite" in settings.database_url:
    from sqlalchemy import event
    
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        """Set SQLite pragmas on connect."""
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.close()
        logger.info("SQLite WAL mode enabled")


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


def seed_default_users() -> None:
    """
    Eğer hiç kullanıcı yoksa varsayılan admin ve researcher hesaplarını oluşturur.
    """
    if not settings.seed_demo_users:
        return
    from backend.models import UserRecord
    from backend.security.password import hash_password
    
    db = SessionLocal()
    try:
        user_count = db.query(UserRecord).count()
        if user_count == 0:
            logger.info("🌱 Varsayılan kullanıcılar veritabanına ekleniyor...")
            admin_user = UserRecord(
                username="admin",
                email="admin@ailab.local",
                hashed_password=hash_password(settings.default_admin_password),
                role="admin",
                full_name="System Administrator",
                is_active=True
            )
            researcher_user = UserRecord(
                username="researcher",
                email="researcher@ailab.local",
                hashed_password=hash_password(settings.default_researcher_password),
                role="researcher",
                full_name="AI Researcher",
                is_active=True
            )
            db.add(admin_user)
            db.add(researcher_user)
            db.commit()
            logger.info("✅ Varsayılan admin ve researcher kullanıcıları oluşturuldu.")
    except Exception as e:
        logger.error(f"Kullanıcı seed işlemi sırasında hata oluştu: {e}")
        db.rollback()
    finally:
        db.close()


def init_db() -> None:
    """
    Database'i initialize et.
    
    Tüm tabloları oluşturur ve başlangıç kullanıcılarını yükler.
    """
    logger.info("Initializing database...")
    import backend.models  # Ensure all models are registered with Base.metadata
    Base.metadata.create_all(bind=engine)
    from backend.migrations import migrate
    migrate(engine)
    logger.info("Database initialized successfully")
    seed_default_users()


def drop_db() -> None:
    """
    Tüm tabloları sil.
    
    **UYARI:** Production'da kullanılmamalı!
    Sadece development ve testing için.
    """
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All tables dropped")
