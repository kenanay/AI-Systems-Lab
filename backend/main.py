"""
Backend API Ana Modülü

FastAPI uygulamasının giriş noktası.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import Dict

from backend.config import settings
from backend.database import init_db
from backend.routers import files, datasets, tokenizer

# Logger yapılandırması
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI uygulaması
app = FastAPI(
    title="Local AI Research Lab API",
    description="Local-First AI Systems Research & Learning Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS yapılandırması
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Routers
app.include_router(files.router)
app.include_router(datasets.router)
app.include_router(tokenizer.router)


@app.get("/")
async def root() -> Dict[str, str]:
    """
    API ana endpoint'i.
    
    Returns:
        Hoş geldiniz mesajı
    """
    return {
        "message": "Local AI Research Lab API",
        "version": "0.1.0",
        "status": "active",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Sağlık kontrolü endpoint'i.
    
    Returns:
        Sistem durumu
    """
    return {
        "status": "healthy",
        "service": "Local AI Research Lab API"
    }


@app.get("/api/v1/info", response_model=None)
async def system_info() -> dict:
    """
    Sistem bilgisi endpoint'i.
    
    Returns:
        Sistem ve yapılandırma bilgileri
    """
    import sys
    
    info = {
        "python_version": sys.version,
    }
    
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["pytorch_version"] = "Not installed"
        info["cuda_available"] = False
    
    return info


# Uygulama başlatıldığında
@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışır."""
    logger.info("🚀 Local AI Research Lab API başlatılıyor...")
    logger.info("📁 Dizinler oluşturuluyor...")
    settings.create_directories()
    logger.info("🗄️  Database initialize ediliyor...")
    init_db()
    logger.info("📚 Steering dosyaları ve hook'lar yükleniyor...")
    logger.info("✅ API hazır!")


# Uygulama kapatılırken
@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapanırken çalışır."""
    logger.info("👋 Local AI Research Lab API kapatılıyor...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
