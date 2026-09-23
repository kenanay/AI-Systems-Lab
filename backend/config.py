"""
Backend Konfigürasyon Yönetimi

Environment variable'lardan konfigürasyon okur.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Uygulama ayarları"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Backend
    backend_host: str = Field(default="localhost", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    backend_reload: bool = Field(default=True, alias="BACKEND_RELOAD")
    
    # Database
    database_url: str = Field(
        default="sqlite:///./local_ai_lab.db",
        alias="DATABASE_URL"
    )
    
    # Security & Auth
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        alias="SECRET_KEY"
    )
    jwt_secret_key: Optional[str] = Field(default=None, alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=14, alias="REFRESH_TOKEN_EXPIRE_DAYS")
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    default_admin_password: str = Field(default="admin", alias="DEFAULT_ADMIN_PASSWORD")
    default_researcher_password: str = Field(default="researcher123", alias="DEFAULT_RESEARCHER_PASSWORD")
    
    # Data Paths
    data_root: Path = Field(default=Path("./datasets"), alias="DATA_ROOT")
    raw_data_path: Path = Field(
        default=Path("./datasets/raw"),
        alias="RAW_DATA_PATH"
    )
    processed_data_path: Path = Field(
        default=Path("./datasets/normalized"),
        alias="PROCESSED_DATA_PATH"
    )
    
    # File Upload
    max_upload_size: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        alias="MAX_UPLOAD_SIZE"
    )
    allowed_extensions: list[str] = Field(
        default=[".txt", ".md", ".pdf"]  # Sadece parser'ı olan formatlar
    )
    temp_upload_dir: Path = Field(
        default=Path("./temp/uploads"),
        alias="TEMP_UPLOAD_DIR"
    )
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Optional[Path] = Field(default=Path("./logs/app.log"), alias="LOG_FILE")
    
    # Feature Flags
    pii_detection_enabled: bool = Field(default=True, alias="PII_DETECTION_ENABLED")
    auto_training_allowed: bool = Field(default=False, alias="AUTO_TRAINING_ALLOWED")
    
    # CORS
    enable_cors: bool = Field(default=True, alias="ENABLE_CORS")
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        alias="ALLOWED_ORIGINS"
    )
    
    def create_directories(self) -> None:
        """Gerekli dizinleri oluştur"""
        dirs = [
            self.data_root,
            self.raw_data_path,
            self.processed_data_path,
            self.temp_upload_dir,
        ]
        
        if self.log_file:
            dirs.append(self.log_file.parent)
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Dizinleri oluştur
settings.create_directories()
