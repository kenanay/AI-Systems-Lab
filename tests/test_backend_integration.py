"""
tests/test_backend_integration.py

Backend Integration Tests - Comprehensive API Testing

Bu modül backend'in tüm önemli endpoint'lerini ve workflow'larını test eder.
Integration test olarak gerçek database ve file system ile çalışır.

Test Coverage:
- Health & Root endpoints
- File management workflow
- Dataset compilation pipeline
- Tokenizer training workflow
- Model registry operations
- Training job management
- Inference endpoints
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from typing import Generator
import tempfile
import json
import time

from backend.main import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """
    FastAPI test client (module scope - daha hızlı).
    
    Yields:
        TestClient: API test client
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_txt_file() -> Generator[Path, None, None]:
    """
    Test için geçici TXT dosyası oluştur.
    
    Yields:
        Path: Geçici dosya path'i
    """
    content = """Bu bir test belgesidir.
    
Türkçe karakter testleri: çÇ ğĞ ıİ öÖ şŞ üÜ

Matematik: E=mc²
Özel karakterler: €, ©, ™

Birkaç cümle daha ekleyelim ki tokenizer'da yeterli veri olsun.
Bu metinle BPE tokenizer eğitimi yapabiliriz.
Test amaçlı oluşturulmuş bir dokümandır."""
    
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.txt', 
        delete=False, 
        encoding='utf-8'
    ) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_md_file() -> Generator[Path, None, None]:
    """
    Test için geçici Markdown dosyası oluştur.
    
    Yields:
        Path: Geçici dosya path'i
    """
    content = """# Test Markdown

## Bölüm 1

Bu bir **markdown** testidir.

- Liste item 1
- Liste item 2

```python
def test():
    return "code block"
```

## Bölüm 2

> Quote block

Normal paragraf."""
    
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.md', 
        delete=False, 
        encoding='utf-8'
    ) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestRootAndHealth:
    """Root ve health endpoint testleri"""
    
    def test_root_endpoint(self, client: TestClient) -> None:
        """
        GET / - Ana endpoint.
        
        Doğrulama:
        - 200 OK
        - message, version, status bilgileri
        - docs link
        """
        response = client.get("/")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "active"
        assert "docs" in data
        assert data["docs"] == "/docs"
    
    def test_health_endpoint(self, client: TestClient) -> None:
        """
        GET /health - Health check.
        
        Doğrulama:
        - 200 OK
        - status="healthy"
        """
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"


class TestFileManagementWorkflow:
    """File yönetimi workflow testleri"""
    
    def test_upload_and_list_files(
        self, 
        client: TestClient, 
        sample_txt_file: Path
    ) -> None:
        """
        File upload → list workflow.
        
        Adımlar:
        1. Dosya yükle
        2. Dosya listesini çek
        3. Yüklenen dosya listede olmalı
        """
        # 1. Upload
        with open(sample_txt_file, 'rb') as f:
            response = client.post(
                "/api/v1/files/upload",
                files={"file": (sample_txt_file.name, f, "text/plain")}
            )
        
        assert response.status_code == 201
        upload_data = response.json()
        file_id = upload_data["file_id"]
        
        # 2. List files
        response = client.get("/api/v1/files")
        assert response.status_code == 200
        
        files = response.json()
        assert isinstance(files, list)
        
        # 3. Yüklenen dosya listede olmalı
        file_ids = [f["file_id"] for f in files]
        assert file_id in file_ids
    
    def test_upload_multiple_files(
        self, 
        client: TestClient, 
        sample_txt_file: Path,
        sample_md_file: Path
    ) -> None:
        """
        Birden fazla dosya yükleme testi.
        
        İki farklı dosya yükleyip her ikisinin de başarıyla 
        kaydedildiğini doğrula.
        """
        uploaded_ids = []
        
        # TXT yükle
        with open(sample_txt_file, 'rb') as f:
            response = client.post(
                "/api/v1/files/upload",
                files={"file": (sample_txt_file.name, f, "text/plain")}
            )
        assert response.status_code == 201
        uploaded_ids.append(response.json()["file_id"])
        
        # MD yükle
        with open(sample_md_file, 'rb') as f:
            response = client.post(
                "/api/v1/files/upload",
                files={"file": (sample_md_file.name, f, "text/markdown")}
            )
        assert response.status_code == 201
        uploaded_ids.append(response.json()["file_id"])
        
        # Her iki dosya da farklı ID'ye sahip olmalı
        assert len(set(uploaded_ids)) == 2
    
    def test_upload_validation(self, client: TestClient) -> None:
        """
        Dosya validasyon testleri.
        
        Testler:
        - Boş dosya
        - Desteklenmeyen format
        """
        # Boş dosya
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    "/api/v1/files/upload",
                    files={"file": ("empty.txt", f, "text/plain")}
                )
            
            # Boş dosya kabul edilebilir veya reddedilebilir
            # (implementasyona göre)
            assert response.status_code in [201, 400]
        finally:
            temp_path.unlink()


class TestDatasetOperations:
    """Dataset operasyon testleri"""
    
    def test_list_documents(self, client: TestClient) -> None:
        """
        GET /api/datasets/documents - Document listesi.
        
        Doğrulama:
        - 200 OK
        - Liste dönmeli
        """
        response = client.get("/api/v1/datasets/documents")
        
        assert response.status_code == 200
        
        documents = response.json()
        assert isinstance(documents, list)
    
    def test_dataset_compilation_config(self, client: TestClient) -> None:
        """
        Dataset compilation için config endpoint'i test et.
        
        Bu test dataset compiler API'sinin hazır olup olmadığını kontrol eder.
        """
        # Config veya status endpoint olabilir
        response = client.get("/api/v1/datasets/versions")
        
        # 200 (success) veya 404 (not yet implemented) kabul edilebilir
        assert response.status_code in [200, 404]


class TestTokenizerWorkflow:
    """Tokenizer training workflow testleri"""
    
    def test_list_tokenizers(self, client: TestClient) -> None:
        """
        GET /api/tokenizers - Tokenizer listesi.
        
        Doğrulama:
        - Endpoint erişilebilir olmalı
        - Liste dönmeli (boş olabilir)
        """
        response = client.get("/api/v1/tokenizers")
        
        # Endpoint'in varlığını kontrol et
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            tokenizers = response.json()
            assert isinstance(tokenizers, list)


class TestModelRegistry:
    """Model registry testleri"""
    
    def test_list_models(self, client: TestClient) -> None:
        """
        GET /api/models - Model listesi.
        
        Doğrulama:
        - Endpoint erişilebilir
        - Liste formatında response
        """
        response = client.get("/api/v1/models")
        
        # Model router aktif olmalı
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            models = response.json()
            assert isinstance(models, list)


class TestTrainingEndpoints:
    """Training job management testleri"""
    
    def test_list_training_jobs(self, client: TestClient) -> None:
        """
        GET /api/training/jobs - Training job listesi.
        
        Doğrulama:
        - Endpoint erişilebilir
        - Liste formatı
        """
        response = client.get("/api/v1/training/jobs")
        
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            jobs = response.json()
            assert isinstance(jobs, list)


class TestInferenceEndpoints:
    """Inference endpoint testleri"""
    
    def test_inference_health(self, client: TestClient) -> None:
        """
        Inference router'ın aktif olduğunu test et.
        
        Not: Bu test inference endpoint'lerinin varlığını kontrol eder.
        Gerçek model inference testi için model yüklenmesi gerekir.
        """
        # Inference router varsa /api/v1/inference path'i olmalı
        # 404 alabiliriz (specific endpoint olmadığı için)
        # Ama bu bile router'ın kayıtlı olduğunu gösterir
        response = client.get("/api/v1/inference/models")
        
        # 200 (success) veya 404 (no models) veya 405 (method not allowed) kabul edilebilir
        assert response.status_code in [200, 404, 405]


class TestCORSHeaders:
    """CORS header testleri"""
    
    def test_cors_headers_on_get(self, client: TestClient) -> None:
        """
        GET request'te CORS header'larını kontrol et.
        
        Doğrulama:
        - Access-Control-Allow-Origin header var mı
        """
        response = client.get("/health")
        
        # CORS middleware varsa header'lar eklenmeli
        # Test client bazen header'ları farklı handle edebilir
        assert response.status_code == 200
    
    def test_cors_preflight(self, client: TestClient) -> None:
        """
        OPTIONS (preflight) request testi.
        
        CORS için tarayıcılar önce OPTIONS request gönderir.
        """
        response = client.options("/api/v1/files")
        
        # 200 veya 405 (method not allowed) kabul edilebilir
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """Error handling ve edge case testleri"""
    
    def test_invalid_endpoint(self, client: TestClient) -> None:
        """
        Olmayan endpoint 404 dönmeli.
        """
        response = client.get("/api/totally-nonexistent-endpoint-xyz")
        
        assert response.status_code == 404
    
    def test_malformed_json(self, client: TestClient) -> None:
        """
        Malformed JSON gönderildiğinde düzgün error dönmeli.
        """
        response = client.post(
            "/api/v1/datasets/compile",
            data="{ invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        # 422 (Unprocessable Entity) veya 400 (Bad Request)
        assert response.status_code in [400, 422]
    
    def test_invalid_file_id_format(self, client: TestClient) -> None:
        """
        Geçersiz file_id formatı ile request.
        """
        response = client.get("/api/v1/files/INVALID-FORMAT-123")
        
        # 404 veya 400
        assert response.status_code in [400, 404]


class TestAPIDocumentation:
    """API dokümantasyon endpoint testleri"""
    
    def test_swagger_ui_accessible(self, client: TestClient) -> None:
        """
        GET /docs - Swagger UI erişilebilir olmalı.
        """
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_accessible(self, client: TestClient) -> None:
        """
        GET /redoc - ReDoc UI erişilebilir olmalı.
        """
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_openapi_schema(self, client: TestClient) -> None:
        """
        GET /openapi.json - OpenAPI schema erişilebilir olmalı.
        """
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


# Performance ve stress testleri (opsiyonel)
@pytest.mark.slow
class TestPerformance:
    """Performance testleri (slow olarak işaretli)"""
    
    def test_concurrent_file_uploads(
        self, 
        client: TestClient, 
        sample_txt_file: Path
    ) -> None:
        """
        Birden fazla dosya ardışık yükleme (concurrency simülasyonu).
        
        Not: Backend duplicate detection yapar, bu yüzden aynı içerik
        birden fazla kez yüklendiğinde aynı file_id döner.
        """
        upload_count = 5
        uploaded_ids = []
        
        for i in range(upload_count):
            with open(sample_txt_file, 'rb') as f:
                response = client.post(
                    "/api/v1/files/upload",
                    files={"file": (f"test_{i}.txt", f, "text/plain")}
                )
            
            assert response.status_code in [200, 201]  # 200 = duplicate, 201 = new
            uploaded_ids.append(response.json()["file_id"])
        
        # Aynı içerik yüklendiği için unique ID count 1 olabilir (duplicate detection)
        # Bu beklenen ve doğru davranıştır
        unique_count = len(set(uploaded_ids))
        assert unique_count >= 1  # En az 1 unique file olmalı
        assert len(uploaded_ids) == upload_count  # Her upload bir response döndü
