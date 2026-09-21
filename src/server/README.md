# GPT Inference Server

Production-ready FastAPI inference server for GPT models.

## Özellikler

- **REST API**: FastAPI ile HTTP endpoints
- **Text Generation**: Prompt-based metin üretimi
- **Streaming**: Real-time token streaming (gelecek versiyon)
- **Health Checks**: Kubernetes-ready liveness/readiness probes
- **Model Loading**: Registry'den model yükleme
- **Validation**: Pydantic ile request validation
- **Error Handling**: Production-grade error responses
- **CORS**: Cross-origin resource sharing desteği

## Kurulum

```bash
# Dependencies (FastAPI, Uvicorn)
pip install fastapi uvicorn[standard] pydantic
```

## Kullanım

### 1. Server'ı Başlatma

```bash
# Development mode
python -m uvicorn src.server.inference_server:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m uvicorn src.server.inference_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. API Dokümantasyonu

Server başladıktan sonra:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Endpoints

#### GET / - Root
```bash
curl http://localhost:8000/
```

#### GET /health - Health Check (Liveness Probe)
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-09-20 21:42:00",
  "version": "1.0.0"
}
```

#### GET /ready - Readiness Check
```bash
curl http://localhost:8000/ready
```

Response:
```json
{
  "ready": true,
  "model_loaded": true,
  "model_name": "gpt-turkish-tiny v1.0.4",
  "version": "1.0.0"
}
```

#### POST /load - Load Model
```bash
curl -X POST "http://localhost:8000/load" \
  -H "Content-Type: application/json" \
  -d '{
    "registry_dir": "./models",
    "model_name": "gpt-turkish-tiny",
    "version": "1.0.4"
  }'
```

#### POST /generate - Text Generation
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Türkiye'\''nin başkenti",
    "max_length": 20,
    "temperature": 0.8,
    "top_k": 50
  }'
```

Response:
```json
{
  "generated_text": "Türkiye'nin başkenti Ankara'dır...",
  "prompt": "Türkiye'nin başkenti",
  "tokens_generated": 15,
  "generation_time_ms": 42.35,
  "model_name": "gpt-turkish-tiny v1.0.4"
}
```

#### GET /models - List Available Models
```bash
curl http://localhost:8000/models
```

## Request Parameters

### GenerateRequest

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| prompt | string | ✓ | - | Input text prompt (min: 1 char) |
| max_length | int | ✗ | 50 | Maximum tokens to generate (1-512) |
| temperature | float | ✗ | 1.0 | Sampling temperature (0.1-2.0) |
| top_k | int | ✗ | None | Top-k sampling |
| top_p | float | ✗ | None | Nucleus sampling (0.0-1.0) |
| model_name | string | ✗ | None | Model name (if multiple loaded) |

## Error Handling

### HTTP Status Codes

- `200 OK`: Success
- `422 Unprocessable Entity`: Validation error
- `503 Service Unavailable`: Model not loaded
- `500 Internal Server Error`: Generation error

### Error Response Format

```json
{
  "detail": "Model not loaded. Please load a model first."
}
```

## Testing

### Unit Tests
```bash
# All endpoints (without model)
python tests/test_inference_server.py
```

### E2E Tests
```bash
# With real model
python examples/test_inference_e2e.py
```

### Manual Testing
```bash
# Start server in one terminal
uvicorn src.server.inference_server:app --reload

# In another terminal
# Health check
curl http://localhost:8000/health

# Generate (will fail without model)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "max_length": 10}'
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.server.inference_server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```bash
# Build
docker build -t gpt-inference-server .

# Run
docker run -p 8000:8000 gpt-inference-server
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpt-inference-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gpt-inference-server
  template:
    metadata:
      labels:
        app: gpt-inference-server
    spec:
      containers:
      - name: server
        image: gpt-inference-server:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: gpt-inference-server
spec:
  selector:
    app: gpt-inference-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Gunicorn (Alternative)

```bash
# Install
pip install gunicorn

# Run with workers
gunicorn src.server.inference_server:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## Performance

### Benchmarking

```bash
# Install Apache Bench
apt-get install apache2-utils

# Benchmark generate endpoint
ab -n 1000 -c 10 -p request.json -T application/json \
   http://localhost:8000/generate
```

request.json:
```json
{"prompt": "test", "max_length": 10}
```

### Optimization Tips

1. **Multi-Worker**: Use multiple Uvicorn workers
   ```bash
   uvicorn src.server.inference_server:app --workers 4
   ```

2. **GPU**: Ensure CUDA available for GPU inference
   ```python
   # Automatically detected in ModelManager
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

3. **Batch Processing**: Future feature for batch generation

4. **Model Caching**: Models stay in memory after loading

## Monitoring

### Prometheus Metrics (Future)

```python
from prometheus_client import Counter, Histogram

generation_requests = Counter('generation_requests_total', 'Total generation requests')
generation_duration = Histogram('generation_duration_seconds', 'Generation duration')
```

### Logging

```python
# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Troubleshooting

### Model Not Loading

```bash
# Check registry directory
ls -la models/

# Check model metadata
cat models/metadata/gpt-turkish-tiny-v1.0.4.json

# Verify tokenizer
ls -la models/gpt-turkish-tiny/tokenizer.model
```

### CUDA Out of Memory

```python
# Reduce batch size or use CPU
device = torch.device('cpu')
```

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn src.server.inference_server:app --port 8001
```

## Gelecek Özellikler

- [ ] Streaming generation endpoint (/generate/stream)
- [ ] Batch generation
- [ ] Rate limiting
- [ ] Authentication/Authorization
- [ ] Prometheus metrics
- [ ] Model hot-swapping
- [ ] Multi-model serving
- [ ] Request queueing
- [ ] Response caching

## Kaynaklar

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Uvicorn Documentation](https://www.uvicorn.org)
- [Production Best Practices](https://fastapi.tiangolo.com/deployment/)
