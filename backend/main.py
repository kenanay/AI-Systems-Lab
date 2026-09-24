"""
Backend API Ana Modülü

FastAPI uygulamasının giriş noktası.

Author: Kenan AY
Location: Kütahya, TÜRKİYE
Version: 1.2.0
"""

import sys
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager
import logging

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI, Depends
from backend.security.scope import require_access
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.database import init_db
from backend.database import SessionLocal
from backend.services.content_store import cleanup_deleting_content
from backend.routers import files, datasets, tokenizer, datasets_compiler, training, models, inference, evaluation, embeddings, rag, tensor_lab, math_lab, transformer_lab, journey, systems_lab, synthetic_lab, nn_lab, auth, architecture_atlas

# Logger yapılandırması
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama yaşam döngüsü yöneticisi."""
    logger.info("🚀 Local AI Research Lab API başlatılıyor...")
    logger.info("📁 Dizinler oluşturuluyor...")
    settings.create_directories()
    logger.info("🗄️  Database initialize ediliyor...")
    init_db()
    cleanup_db = SessionLocal()
    try:
        cleaned, pending = cleanup_deleting_content(cleanup_db)
        logger.info("🧹 Content cleanup tamamlandı: %s temizlendi, %s beklemede", cleaned, pending)
    finally:
        cleanup_db.close()
    logger.info("📚 Steering dosyaları ve hook'lar yükleniyor...")
    logger.info("✅ API hazır!")
    yield
    logger.info("👋 Local AI Research Lab API kapatılıyor...")


# FastAPI uygulaması
app = FastAPI(
    title="Local AI Research Lab API",
    description="Local-First AI Systems Research & Learning Platform - Developed by Kenan AY",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Kenan AY",
        "url": "https://github.com/kenanay/AI-Systems-Lab",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
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


@app.exception_handler(PermissionError)
async def permission_error_handler(request, exc: PermissionError):
    return JSONResponse(
        status_code=403,
        content={"detail": str(exc)},
    )


# Routers
app.include_router(files.router, dependencies=[Depends(require_access)])
app.include_router(datasets.router, dependencies=[Depends(require_access)])
app.include_router(tokenizer.router, dependencies=[Depends(require_access)])
app.include_router(datasets_compiler.router, dependencies=[Depends(require_access)])
app.include_router(training.router, dependencies=[Depends(require_access)])
app.include_router(models.router, dependencies=[Depends(require_access)])
app.include_router(inference.router, dependencies=[Depends(require_access)])
app.include_router(evaluation.router, dependencies=[Depends(require_access)])
app.include_router(embeddings.router, dependencies=[Depends(require_access)])
app.include_router(rag.router, dependencies=[Depends(require_access)])
app.include_router(tensor_lab.router, dependencies=[Depends(require_access)])
app.include_router(math_lab.router, dependencies=[Depends(require_access)])
app.include_router(transformer_lab.router, dependencies=[Depends(require_access)])
app.include_router(journey.router, dependencies=[Depends(require_access)])
app.include_router(systems_lab.router, dependencies=[Depends(require_access)])
app.include_router(synthetic_lab.router, dependencies=[Depends(require_access)])
app.include_router(nn_lab.router, dependencies=[Depends(require_access)])
app.include_router(auth.router)
app.include_router(architecture_atlas.router, dependencies=[Depends(require_access)])


@app.get("/")
async def root() -> Dict[str, str]:
    """
    API ana endpoint'i.
    
    Returns:
        Hoş geldiniz mesajı
    """
    return {
        "message": "Local AI Research Lab API",
        "version": "1.2.0",
        "status": "active",
        "docs": "/docs",
        "developer": "Kenan AY"
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
async def system_info() -> Dict[str, Any]:
    """
    Sistem bilgisi endpoint'i.
    
    Returns:
        Sistem ve yapılandırma bilgileri
    """
    info: Dict[str, Any] = {
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



if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
