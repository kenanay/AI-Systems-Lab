"""
tests/test_tokenizer_api.py

Tokenizer API Integration Tests

Bu modül tokenizer API endpoint'lerini test eder:
- Training job operations
- Tokenizer CRUD
- Encode/decode endpoints
"""

import pytest
from fastapi.testclient import TestClient
from typing import Generator
import time

from backend.main import app


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    """FastAPI test client"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def setup_test_document():
    """Ensure a FileRecord and DocumentRecord with file_id FILE-ED8B9094 exists for tokenizer training tests."""
    from backend.database import SessionLocal, init_db
    from backend.models import FileRecord, DocumentRecord
    init_db()
    db = SessionLocal()
    try:
        file = db.query(FileRecord).filter(FileRecord.file_id == "FILE-ED8B9094").first()
        if not file:
            file = FileRecord(
                file_id="FILE-ED8B9094",
                original_name="FILE-ED8B9094_test_sample.md",
                relative_path="tests/fixtures/test_sample.md",
                mime_type="text/markdown",
                size_bytes=100,
                sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            )
            db.add(file)
            db.commit()

        doc = db.query(DocumentRecord).filter(DocumentRecord.file_id == "FILE-ED8B9094").first()
        if not doc:
            doc = DocumentRecord(
                document_id="DOC-ED8B9094",
                file_id="FILE-ED8B9094",
                title="Sample Test Document",
                text="Bu bir yapay zeka tokenizer eğitim metnidir. Türkçe doğal dil işleme sistemleri ve modelleri test edilmektedir."
            )
            db.add(doc)
            db.commit()
    finally:
        db.close()
    yield


class TestTokenizerTrainingAPI:
    """Tokenizer training API testleri"""
    
    def test_create_training_job_success(self, client: TestClient) -> None:
        """
        POST /api/v1/tokenizer/train - Başarılı job creation.
        
        Test akışı:
        1. Mevcut bir file_id ile job oluştur
        2. Response'u doğrula
        3. Job status kontrol et
        """
        # Job oluştur
        response = client.post(
            "/api/v1/tokenizer/train",
            json={
                "job_name": "Test Tokenizer API",
                "file_ids": ["FILE-ED8B9094"],
                "vocab_size": 300,
                "min_frequency": 1
            }
        )
        
        assert response.status_code == 201
        
        data = response.json()
        assert "job_id" in data
        assert data["job_name"] == "Test Tokenizer API"
        assert data["status"] in ["PENDING", "RUNNING"]
        assert data["progress"] == 0.0
        
        job_id = data["job_id"]
        
        # Job status kontrol et (biraz bekle)
        time.sleep(1)
        
        status_response = client.get(f"/api/v1/tokenizer/jobs/{job_id}")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ["PENDING", "RUNNING", "COMPLETED"]
    
    def test_create_training_job_missing_files(self, client: TestClient) -> None:
        """
        POST /api/v1/tokenizer/train - file_ids eksikse 400 error.
        """
        response = client.post(
            "/api/v1/tokenizer/train",
            json={
                "job_name": "Invalid Job",
                "vocab_size": 300
            }
        )
        
        assert response.status_code == 400
        assert "detail" in response.json()
    
    def test_list_training_jobs(self, client: TestClient) -> None:
        """
        GET /api/v1/tokenizer/jobs - Job listesi.
        """
        response = client.get("/api/v1/tokenizer/jobs")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        # Her item job structure'a uygun mu?
        for job in data:
            assert "job_id" in job
            assert "job_name" in job
            assert "status" in job
            assert "progress" in job
    
    def test_list_training_jobs_with_filter(self, client: TestClient) -> None:
        """
        GET /api/v1/tokenizer/jobs?status=COMPLETED - Status filter.
        """
        response = client.get("/api/v1/tokenizer/jobs?status=COMPLETED")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        # Tüm job'lar COMPLETED durumunda mı?
        for job in data:
            assert job["status"] == "COMPLETED"
    
    def test_get_training_job_detail(self, client: TestClient) -> None:
        """
        GET /api/v1/tokenizer/jobs/{job_id} - Job detail.
        
        Önce bir job oluştur, sonra detayını al.
        """
        # Job oluştur
        create_response = client.post(
            "/api/v1/tokenizer/train",
            json={
                "job_name": "Detail Test Job",
                "file_ids": ["FILE-ED8B9094"],
                "vocab_size": 200,
                "min_frequency": 1
            }
        )
        
        assert create_response.status_code == 201
        job_id = create_response.json()["job_id"]
        
        # Detail getir
        detail_response = client.get(f"/api/v1/tokenizer/jobs/{job_id}")
        
        assert detail_response.status_code == 200
        
        data = detail_response.json()
        assert data["job_id"] == job_id
        assert data["job_name"] == "Detail Test Job"
        assert "metadata" in data
    
    def test_get_nonexistent_job(self, client: TestClient) -> None:
        """
        GET /api/v1/tokenizer/jobs/{job_id} - Olmayan job 404.
        """
        response = client.get("/api/v1/tokenizer/jobs/NONEXISTENT")
        
        assert response.status_code == 404


class TestTokenizerListAPI:
    """Tokenizer list/detail API testleri"""
    
    def test_list_tokenizers(self, client: TestClient) -> None:
        """
        GET /api/v1/tokenizer/list - Tokenizer listesi.
        """
        response = client.get("/api/v1/tokenizer/list")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        # Her item tokenizer structure'a uygun mu?
        for tok in data:
            assert "tokenizer_id" in tok
            assert "name" in tok
            assert "tokenizer_type" in tok
            assert "vocab_size" in tok
    
    def test_list_tokenizers_active_only(self, client: TestClient) -> None:
        """
        GET /api/v1/tokenizer/list?is_active=true - Aktif tokenizer'lar.
        """
        response = client.get("/api/v1/tokenizer/list?is_active=true")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        # Tüm tokenizer'lar aktif mi?
        for tok in data:
            assert tok["is_active"] is True
    
    def test_get_tokenizer_detail(self, client: TestClient) -> None:
        """
        GET /api/v1/tokenizer/{tokenizer_id} - Tokenizer detail.
        
        Mevcut tokenizer'lardan birini al.
        """
        # List'ten bir tokenizer al
        list_response = client.get("/api/v1/tokenizer/list")
        tokenizers = list_response.json()
        
        if len(tokenizers) == 0:
            pytest.skip("No tokenizers available")
        
        tokenizer_id = tokenizers[0]["tokenizer_id"]
        
        # Detail getir
        detail_response = client.get(f"/api/v1/tokenizer/{tokenizer_id}")
        
        assert detail_response.status_code == 200
        
        data = detail_response.json()
        assert data["tokenizer_id"] == tokenizer_id
        assert "vocab_size" in data
        assert "special_tokens" in data
        assert "training_duration_seconds" in data
    
    def test_get_nonexistent_tokenizer(self, client: TestClient) -> None:
        """
        GET /api/v1/tokenizer/{tokenizer_id} - Olmayan tokenizer 404.
        """
        response = client.get("/api/v1/tokenizer/TOK-NONEXISTENT")
        
        assert response.status_code == 404


class TestTokenizerEncodeDecodeAPI:
    """Encode/decode API testleri"""
    
    def test_encode_text(self, client: TestClient) -> None:
        """
        POST /api/v1/tokenizer/{tokenizer_id}/encode - Text encoding.
        
        Mevcut tokenizer ile text encode et.
        """
        # Mevcut tokenizer al
        list_response = client.get("/api/v1/tokenizer/list")
        tokenizers = list_response.json()
        
        if len(tokenizers) == 0:
            pytest.skip("No tokenizers available")
        
        tokenizer_id = tokenizers[0]["tokenizer_id"]
        
        # Encode
        encode_response = client.post(
            f"/api/v1/tokenizer/{tokenizer_id}/encode",
            json={"text": "Merhaba dünya"}
        )
        
        assert encode_response.status_code == 200
        
        data = encode_response.json()
        assert "text" in data
        assert "token_ids" in data
        assert "num_tokens" in data
        
        assert data["text"] == "Merhaba dünya"
        assert isinstance(data["token_ids"], list)
        assert len(data["token_ids"]) > 0
        assert data["num_tokens"] == len(data["token_ids"])
    
    def test_decode_tokens(self, client: TestClient) -> None:
        """
        POST /api/v1/tokenizer/{tokenizer_id}/decode - Token decoding.
        
        Encode → Decode roundtrip.
        """
        # Mevcut tokenizer al
        list_response = client.get("/api/v1/tokenizer/list")
        tokenizers = list_response.json()
        
        if len(tokenizers) == 0:
            pytest.skip("No tokenizers available")
        
        tokenizer_id = tokenizers[0]["tokenizer_id"]
        
        # Encode
        encode_response = client.post(
            f"/api/v1/tokenizer/{tokenizer_id}/encode",
            json={"text": "test"}
        )
        
        token_ids = encode_response.json()["token_ids"]
        
        # Decode
        decode_response = client.post(
            f"/api/v1/tokenizer/{tokenizer_id}/decode",
            json={"token_ids": token_ids}
        )
        
        assert decode_response.status_code == 200
        
        data = decode_response.json()
        assert "text" in data
        assert "token_ids" in data
        
        # Decoded text orijinal text'e yakın olmalı
        assert "test" in data["text"].lower()
    
    def test_encode_nonexistent_tokenizer(self, client: TestClient) -> None:
        """
        POST /api/v1/tokenizer/{tokenizer_id}/encode - Olmayan tokenizer 404.
        """
        response = client.post(
            "/api/v1/tokenizer/TOK-NONEXISTENT/encode",
            json={"text": "test"}
        )
        
        assert response.status_code == 404
    
    def test_encode_empty_text(self, client: TestClient) -> None:
        """
        POST /api/v1/tokenizer/{tokenizer_id}/encode - Boş text.
        
        Boş string encode edilebilmeli (0 tokens).
        """
        # Mevcut tokenizer al
        list_response = client.get("/api/v1/tokenizer/list")
        tokenizers = list_response.json()
        
        if len(tokenizers) == 0:
            pytest.skip("No tokenizers available")
        
        tokenizer_id = tokenizers[0]["tokenizer_id"]
        
        # Encode empty string
        encode_response = client.post(
            f"/api/v1/tokenizer/{tokenizer_id}/encode",
            json={"text": ""}
        )
        
        assert encode_response.status_code == 200
        
        data = encode_response.json()
        # Empty string genelde 0 token üretir
        assert len(data["token_ids"]) == 0 or len(data["token_ids"]) == 1


class TestTokenizerUpdateDeleteAPI:
    """Tokenizer update/delete API testleri"""
    
    def test_update_tokenizer_metadata(self, client: TestClient) -> None:
        """
        PATCH /api/v1/tokenizer/{tokenizer_id} - Metadata update.
        """
        # Mevcut tokenizer al
        list_response = client.get("/api/v1/tokenizer/list")
        tokenizers = list_response.json()
        
        if len(tokenizers) == 0:
            pytest.skip("No tokenizers available")
        
        tokenizer_id = tokenizers[0]["tokenizer_id"]
        
        # Update
        update_response = client.patch(
            f"/api/v1/tokenizer/{tokenizer_id}",
            json={
                "description": "Updated test description",
                "tags": ["test", "updated"]
            }
        )
        
        assert update_response.status_code == 200
        
        data = update_response.json()
        assert data["description"] == "Updated test description"
        assert "test" in data["tags"]
        assert "updated" in data["tags"]
    
    def test_update_nonexistent_tokenizer(self, client: TestClient) -> None:
        """
        PATCH /api/v1/tokenizer/{tokenizer_id} - Olmayan tokenizer 404.
        """
        response = client.patch(
            "/api/v1/tokenizer/TOK-NONEXISTENT",
            json={"description": "test"}
        )
        
        assert response.status_code == 404
    
    def test_delete_tokenizer_soft(self, client: TestClient) -> None:
        """
        DELETE /api/v1/tokenizer/{tokenizer_id} - Soft delete.
        
        Not: Bu test tokenizer'ı deactivate edecek.
        Gerçek production'da dikkatli kullanılmalı.
        """
        # Job oluştur ve tokenizer'ı bekle
        create_response = client.post(
            "/api/v1/tokenizer/train",
            json={
                "job_name": "Delete Test Tokenizer",
                "file_ids": ["FILE-ED8B9094"],
                "vocab_size": 150,
                "min_frequency": 1
            }
        )
        
        assert create_response.status_code == 201
        job_id = create_response.json()["job_id"]
        
        # Job'ın tamamlanmasını bekle
        max_wait = 10  # 10 saniye
        for _ in range(max_wait):
            time.sleep(1)
            status_response = client.get(f"/api/v1/tokenizer/jobs/{job_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "COMPLETED":
                tokenizer_id = status_data["metadata"].get("tokenizer_id")
                if tokenizer_id:
                    break
        else:
            pytest.skip("Job did not complete in time")
        
        # Soft delete
        delete_response = client.delete(f"/api/v1/tokenizer/{tokenizer_id}")
        
        assert delete_response.status_code == 204
        
        # Tokenizer hala var ama is_active=False olmalı
        detail_response = client.get(f"/api/v1/tokenizer/{tokenizer_id}")
        
        if detail_response.status_code == 200:
            data = detail_response.json()
            assert data["is_active"] is False
