"""
src/server/inference_server.py

Production FastAPI Inference Server

GPT model serving with REST API - generation, streaming, health checks.

Bu modül production-ready model serving sağlar:
- REST API: FastAPI ile HTTP endpoints
- Text Generation: Prompt-based generation
- Streaming: Real-time token streaming
- Health Checks: Liveness ve readiness probes
- Model Loading: Registry'den model loading
- Error Handling: Production error responses
- Rate Limiting: Basic rate limiting (optional)

Features:
- /generate endpoint: Text generation
- /generate/stream endpoint: Streaming generation
- /health endpoint: Liveness probe
- /ready endpoint: Readiness probe
- /models endpoint: Available models
- Request validation
- Response formatting
- Error handling

Kaynaklar:
    - FastAPI Best Practices
    - Production ML Serving Patterns
    - https://fastapi.tiangolo.com

Usage:
    # Start server
    python -m uvicorn src.server.inference_server:app --host 0.0.0.0 --port 8000
    
    # Test endpoint
    curl -X POST http://localhost:8000/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Türkiye", "max_length": 20}'
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, AsyncIterator, AsyncGenerator, List
import torch
import logging
import time
from pathlib import Path
import sys
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.gpt import GPTModel, GPTConfig
from src.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from src.inference.generation import generate_text
from src.registry.model_registry import ModelRegistry

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Request/Response Models
# =========================

class GenerateRequest(BaseModel):
    """
    Text generation request.
    
    Attributes:
        prompt: Input text prompt
        max_length: Maximum generation length
        temperature: Sampling temperature (0.1-2.0)
        top_k: Top-k sampling (optional)
        top_p: Nucleus sampling (optional)
        model_name: Model to use (optional, default=loaded model)
    """
    prompt: str = Field(..., description="Input text prompt", min_length=1)
    max_length: int = Field(50, description="Maximum tokens to generate", ge=1, le=512)
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.1, le=2.0)
    top_k: Optional[int] = Field(None, description="Top-k sampling", ge=1)
    top_p: Optional[float] = Field(None, description="Nucleus sampling", ge=0.0, le=1.0)
    model_name: Optional[str] = Field(None, description="Model name (optional)")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "prompt": "Türkiye'nin başkenti",
                "max_length": 20,
                "temperature": 0.8
            }
        }


class GenerateResponse(BaseModel):
    """
    Text generation response.
    
    Attributes:
        generated_text: Generated text
        prompt: Original prompt
        tokens_generated: Number of tokens generated
        generation_time_ms: Generation time in milliseconds
        model_name: Model used
    """
    generated_text: str
    prompt: str
    tokens_generated: int
    generation_time_ms: float
    model_name: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    model_loaded: bool
    model_name: Optional[str]
    version: str


class ModelsResponse(BaseModel):
    """Available models response."""
    models: List[Dict[str, Any]]
    count: int


# =========================
# Model Manager
# =========================

class ModelManager:
    """
    Model lifecycle manager.
    
    Handles model loading, caching, and inference.
    """
    
    def __init__(self) -> None:
        """Initialize manager."""
        self.model: Optional[GPTModel] = None
        self.tokenizer: Optional[Any] = None
        self.model_name: Optional[str] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.registry = None
        logger.info(f"ModelManager initialized (device: {self.device})")
    
    def load_model(
        self,
        registry_dir: str,
        model_name: str,
        version: Optional[str] = None
    ) -> None:
        """
        Load model from registry.
        
        Args:
            registry_dir: Registry directory
            model_name: Model name
            version: Version (None = latest)
        """
        logger.info(f"Loading model: {model_name} v{version or 'latest'}")
        
        # Initialize registry
        self.registry = ModelRegistry(registry_dir)
        
        # Load model info
        model_info = self.registry.load_model(
            model_name,
            version,
            load_weights=True
        )
        
        metadata = model_info['metadata']
        
        # Create model config
        training_config = metadata['training_config']
        model_config = GPTConfig(
            vocab_size=training_config.get('vocab_size', 500),
            max_seq_len=training_config.get('max_seq_len', 64),
            d_model=training_config.get('d_model', 128),
            n_layers=training_config.get('n_layers', 4),
            n_heads=training_config.get('n_heads', 4),
            d_ff=training_config.get('d_ff', 512),
            dropout=training_config.get('dropout', 0.1)
        )
        
        # Create and load model
        self.model = GPTModel(model_config)
        self.model.load_state_dict(model_info['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        tok_path = model_info.get('tokenizer_path')
        if tok_path and Path(tok_path).exists():
            try:
                self.tokenizer = SentencePieceTokenizer()
                self.tokenizer.load(tok_path)
            except Exception:
                try:
                    from src.tokenizer.bpe import BPETokenizer
                    self.tokenizer = BPETokenizer.load(tok_path)
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer from {tok_path}: {e}")

        # Fallback to model directory if tokenizer still None
        if self.tokenizer is None:
            model_dir = Path(model_info.get('model_dir', ''))
            candidate_tok = model_dir / 'tokenizer.model'
            if candidate_tok.exists():
                try:
                    self.tokenizer = SentencePieceTokenizer()
                    self.tokenizer.load(str(candidate_tok))
                except Exception as e:
                    logger.warning(f"Failed to load fallback tokenizer: {e}")
        
        self.model_name = f"{model_name} v{metadata['version']}"
        
        logger.info(f"✓ Model loaded: {self.model_name}")
        logger.info(f"  Parameters: {metadata['parameters']:,}")
        logger.info(f"  Device: {self.device}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> tuple[str, int, float]:
        """
        Generate text.
        
        Args:
            prompt: Input prompt
            max_length: Max generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Tuple[str, int, float]: (generated_text, num_tokens, time_ms)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Tokenize prompt
        # prompt_tokens shape: [seq_len] variable length
        prompt_tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        
        # prompt_ids shape: [1, seq_len]
        prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(self.device)
        
        # Generate
        # generated shape: [1, seq_len + max_length]
        with torch.no_grad():
            generated = generate_text(
                self.model,
                prompt_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode
        generated_tokens = generated[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate time
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Count generated tokens (excluding prompt)
        num_generated = len(generated_tokens) - len(prompt_tokens)
        
        return generated_text, num_generated, elapsed_ms
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None


# =========================
# FastAPI Application
# =========================

# Global model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting inference server...")
    
    # Note: Model loading done on first request or via /load endpoint
    # This allows server to start quickly
    
    yield
    
    # Shutdown
    logger.info("Shutting down inference server...")


# Create FastAPI app
app = FastAPI(
    title="GPT Inference Server",
    description="Production inference server for GPT models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Endpoints
# =========================

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "GPT Inference Server",
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /generate",
            "health": "GET /health",
            "ready": "GET /ready",
            "models": "GET /models"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint (liveness probe).
    
    Returns server health status.
    """
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version="1.0.0"
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """
    Readiness check endpoint (readiness probe).
    
    Returns whether server is ready to handle requests.
    """
    return ReadyResponse(
        ready=model_manager.is_loaded(),
        model_loaded=model_manager.is_loaded(),
        model_name=model_manager.model_name,
        version="1.0.0"
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Text generation endpoint.
    
    Generate text based on prompt.
    
    Args:
        request: Generation request
        
    Returns:
        GenerateResponse: Generated text and metadata
        
    Raises:
        HTTPException: If model not loaded or generation fails
    """
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please load a model first."
        )
    
    try:
        # Generate
        generated_text, num_tokens, time_ms = model_manager.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            tokens_generated=num_tokens,
            generation_time_ms=round(time_ms, 2),
            model_name=model_manager.model_name or "unknown"
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@app.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """
    List available models.
    
    Returns list of models from registry.
    """
    if model_manager.registry is None:
        return ModelsResponse(models=[], count=0)
    
    try:
        models = model_manager.registry.list_models()
        
        model_list = [
            {
                "name": m.model_name,
                "version": m.version,
                "environment": m.environment,
                "metrics": m.metrics,
                "parameters": m.parameters
            }
            for m in models
        ]
        
        return ModelsResponse(
            models=model_list,
            count=len(model_list)
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@app.post("/load")
async def load_model(
    registry_dir: str,
    model_name: str,
    version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load model endpoint.
    
    Load a model from registry.
    
    Args:
        registry_dir: Registry directory
        model_name: Model name
        version: Version (optional)
        
    Returns:
        Dict: Loading status
    """
    try:
        model_manager.load_model(registry_dir, model_name, version)
        
        return {
            "status": "success",
            "message": f"Model loaded: {model_manager.model_name}",
            "model_name": model_manager.model_name
        }
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


# =========================
# Main (for testing)
# =========================

def main() -> None:
    """
    Run server for testing.
    
    Usage:
        python src/server/inference_server.py
    """
    import uvicorn
    
    print("\n" + "=" * 80)
    print("Starting FastAPI Inference Server")
    print("=" * 80)
    print("\nEndpoints:")
    print("  - http://localhost:8000")
    print("  - http://localhost:8000/docs (Swagger UI)")
    print("  - http://localhost:8000/redoc (ReDoc)")
    print("\nPress Ctrl+C to stop")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        "src.server.inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
