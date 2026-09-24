"""
Unit and Integration Tests for Auth & Security Module

Parola hashleme, JWT token yaşam döngüsü, API Key yönetimi,
Rate Limiter ve /api/v1/auth endpoint testleri.

Author: Kenan AY
"""

import time
from datetime import timedelta
import pytest
from fastapi.testclient import TestClient

from typing import Generator
from backend.database import init_db
from backend.main import app
from backend.models import UserRecord, APIKeyRecord
from backend.security.password import hash_password, verify_password
from backend.security.jwt import create_access_token, create_refresh_token, decode_token
from backend.security.api_keys import generate_api_key, hash_api_key
from backend.security.rate_limiter import SlidingWindowRateLimiter, global_rate_limiter


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    from backend.config import settings
    from backend.database import seed_default_users
    orig_seed = settings.seed_demo_users
    settings.seed_demo_users = True
    init_db()
    seed_default_users()
    settings.seed_demo_users = orig_seed
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# 1. Parola Güvenliği Testleri
# ============================================================================

def test_password_hashing_and_verification():
    raw_pass = "GuvenliParola_2026!"
    hashed = hash_password(raw_pass)

    assert hashed.startswith("pbkdf2_sha256$100000$")
    assert verify_password(raw_pass, hashed) is True
    assert verify_password("YanlisParola", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(raw_pass, "") is False


def test_password_tamper_resistance():
    raw_pass = "Test1234!"
    hashed = hash_password(raw_pass)
    # Manipüle edilmiş hash
    parts = hashed.split("$")
    corrupted = f"{parts[0]}${parts[1]}${parts[2]}${'0' * len(parts[3])}"
    assert verify_password(raw_pass, corrupted) is False


# ============================================================================
# 2. JWT Token Testleri
# ============================================================================

def test_jwt_access_and_refresh_tokens():
    payload = {"sub": "usr_test123", "username": "testuser", "role": "researcher"}
    
    access_tok = create_access_token(payload, expires_delta=timedelta(minutes=15))
    decoded_access = decode_token(access_tok)
    
    assert decoded_access["sub"] == "usr_test123"
    assert decoded_access["username"] == "testuser"
    assert decoded_access["role"] == "researcher"
    assert decoded_access["type"] == "access"
    assert "exp" in decoded_access
    assert "iat" in decoded_access

    refresh_tok = create_refresh_token(payload, expires_delta=timedelta(days=7))
    decoded_refresh = decode_token(refresh_tok)
    assert decoded_refresh["sub"] == "usr_test123"
    assert decoded_refresh["type"] == "refresh"


def test_jwt_tamper_and_expiration():
    payload = {"sub": "usr_test123"}
    # Süresi geçmiş token
    expired_tok = create_access_token(payload, expires_delta=timedelta(seconds=-10))
    with pytest.raises(ValueError, match="expired"):
        decode_token(expired_tok, verify_exp=True)

    # İmzası bozulmuş token
    valid_tok = create_access_token(payload, expires_delta=timedelta(minutes=10))
    parts = valid_tok.split(".")
    tampered_sig = parts[0] + "." + parts[1] + ".invalidSignature123"
    with pytest.raises(ValueError, match="imzası doğrulanamadı"):
        decode_token(tampered_sig)


# ============================================================================
# 3. API Key Testleri
# ============================================================================

def test_api_key_generation_and_hashing():
    raw_key, prefix, key_hash = generate_api_key(prefix="sk_live")
    
    assert raw_key.startswith("sk_live_")
    assert prefix.startswith("sk_live_")
    assert prefix.endswith("...")
    assert len(key_hash) == 64
    assert hash_api_key(raw_key) == key_hash
    assert hash_api_key("sk_live_fakedifferent") != key_hash


def test_api_key_role_scope_is_enforced_for_admin_endpoints(client: TestClient):
    """A scoped API key must not regain the parent user's admin privileges."""
    global_rate_limiter.reset()

    login = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "admin", "password": "admin"},
    )
    assert login.status_code == 200
    admin_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    create_key = client.post(
        "/api/v1/auth/api-keys",
        json={"name": "read-only-admin-key", "role": "researcher"},
        headers=admin_headers,
    )
    assert create_key.status_code == 201, create_key.text
    scoped_key = create_key.json()["raw_key"]

    # This endpoint is admin-only. The parent account is admin, but the key is
    # explicitly scoped to researcher and must therefore be rejected.
    response = client.get("/api/v1/auth/users", headers={"X-API-Key": scoped_key})
    assert response.status_code == 403


def test_api_key_role_cannot_be_invalid_or_escalated(client: TestClient):
    """API key creation fails closed for unknown and higher-privilege roles."""
    global_rate_limiter.reset()

    login = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": "researcher", "password": "researcher123"},
    )
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    invalid = client.post(
        "/api/v1/auth/api-keys",
        json={"name": "invalid-role-key", "role": "superadmin"},
        headers=headers,
    )
    assert invalid.status_code == 422

    escalated = client.post(
        "/api/v1/auth/api-keys",
        json={"name": "escalated-key", "role": "admin"},
        headers=headers,
    )
    assert escalated.status_code == 403


# ============================================================================
# 4. Rate Limiter Testleri
# ============================================================================

def test_sliding_window_rate_limiter():
    limiter = SlidingWindowRateLimiter()
    key = "client_ip_test_1"

    # 3 istek/saniye kotası
    allowed1, rem1, _ = limiter.is_allowed(key, max_requests=3, window_seconds=2)
    assert allowed1 is True
    assert rem1 == 2

    allowed2, rem2, _ = limiter.is_allowed(key, max_requests=3, window_seconds=2)
    assert allowed2 is True
    assert rem2 == 1

    allowed3, rem3, _ = limiter.is_allowed(key, max_requests=3, window_seconds=2)
    assert allowed3 is True
    assert rem3 == 0

    # 4. istek kotayı aşmalı
    allowed4, rem4, retry_after = limiter.is_allowed(key, max_requests=3, window_seconds=2)
    assert allowed4 is False
    assert rem4 == 0
    assert retry_after >= 1


# ============================================================================
# 5. Auth API Entegrasyon Testleri
# ============================================================================

def test_auth_register_and_login_flow(client: TestClient):
    import uuid
    global_rate_limiter.reset()

    uid = uuid.uuid4().hex[:8]
    test_uname = f"deniz_{uid}"
    test_email = f"deniz_{uid}@ailab.local"

    # 1. Register
    reg_data = {
        "username": test_uname,
        "email": test_email,
        "password": "GucluSifre2026*",
        "full_name": "Dr. Deniz Kaya",
        "role": "researcher"
    }
    res_reg = client.post("/api/v1/auth/register", json=reg_data)
    assert res_reg.status_code == 201
    user_info = res_reg.json()["user"]
    assert user_info["username"] == test_uname
    assert user_info["role"] == "researcher"

    # Duplicate username check
    res_dup = client.post("/api/v1/auth/register", json=reg_data)
    assert res_dup.status_code == 400

    # 2. Login
    login_data = {
        "username_or_email": test_uname,
        "password": "GucluSifre2026*"
    }
    res_login = client.post("/api/v1/auth/login", json=login_data)
    assert res_login.status_code == 200
    token_resp = res_login.json()
    assert "access_token" in token_resp
    assert "refresh_token" in token_resp
    assert token_resp["token_type"] == "bearer"
    access_token = token_resp["access_token"]
    refresh_token = token_resp["refresh_token"]

    # 3. GET /me with Bearer token
    headers = {"Authorization": f"Bearer {access_token}"}
    res_me = client.get("/api/v1/auth/me", headers=headers)
    assert res_me.status_code == 200
    me_data = res_me.json()
    assert me_data["user"]["username"] == test_uname
    assert me_data["permissions"]["can_train"] is True
    assert me_data["permissions"]["can_delete_models"] is False  # Researcher cannot delete models

    # 4. Refresh Token Flow
    res_ref = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert res_ref.status_code == 200
    new_access_token = res_ref.json()["access_token"]
    assert new_access_token != ""

    # 5. Create API Key
    res_key = client.post(
        "/api/v1/auth/api-keys",
        json={"name": "Test Key for PyTest", "expires_in_days": 30},
        headers=headers
    )
    assert res_key.status_code == 201
    key_data = res_key.json()
    assert "raw_key" in key_data
    raw_api_key = key_data["raw_key"]
    key_id = key_data["key_id"]
    assert raw_api_key.startswith("sk_live_")

    # 6. Authenticate with X-API-Key
    res_me_key = client.get("/api/v1/auth/me", headers={"X-API-Key": raw_api_key})
    assert res_me_key.status_code == 200
    assert res_me_key.json()["user"]["username"] == test_uname

    # 7. List and Delete API Key
    res_list_keys = client.get("/api/v1/auth/api-keys", headers=headers)
    assert res_list_keys.status_code == 200
    keys_list = res_list_keys.json()
    assert len(keys_list) >= 1
    assert "raw_key" not in keys_list[0] or keys_list[0]["raw_key"] is None

    res_del_key = client.delete(f"/api/v1/auth/api-keys/{key_id}", headers=headers)
    assert res_del_key.status_code == 200


def test_rbac_admin_vs_researcher(client: TestClient):
    global_rate_limiter.reset()

    # Varsayılan seed edilmiş admin ile giriş
    login_admin = client.post("/api/v1/auth/login", json={
        "username_or_email": "admin",
        "password": "admin"
    })
    assert login_admin.status_code == 200
    admin_token = login_admin.json()["access_token"]

    # Researcher kullanıcısı ile giriş
    login_res = client.post("/api/v1/auth/login", json={
        "username_or_email": "researcher",
        "password": "researcher123"
    })
    assert login_res.status_code == 200
    researcher_token = login_res.json()["access_token"]

    # 1. Admin /users listesini çekebilir
    res_admin_users = client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_admin_users.status_code == 200
    assert res_admin_users.json()["total"] >= 2

    # 2. Researcher /users endpoint'ine erişmeye çalıştığında 403 Forbidden almalıdır
    res_unauth_users = client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {researcher_token}"})
    assert res_unauth_users.status_code == 403
    assert "yetkiniz yetersiz" in res_unauth_users.json()["detail"].lower()


def test_unauthenticated_requests(client: TestClient):
    # Token olmadan korumalı endpoint'e erişim
    res_no_auth = client.get("/api/v1/auth/me")
    assert res_no_auth.status_code == 401

    # Geçersiz token
    res_bad_tok = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer bad.token.here"})
    assert res_bad_tok.status_code == 401


def test_row_level_data_isolation_between_users(client: TestClient):
    """
    Doğrulama: Kullanıcılar arası veri izolasyonu ve kayıt düzeyinde yetkilendirme (P1).
    - Kullanıcı A'nın yüklediği dosya/dokümana Kullanıcı B erişemez (403 Forbidden).
    - Admin kullanıcı tüm kaynaklara erişebilir.
    """
    global_rate_limiter.reset()
    import time
    ts = int(time.time() * 1000)

    # 1. Kullanıcı A kaydı ve girişi
    uname_a = f"researcher_a_{ts}"
    client.post("/api/v1/auth/register", json={
        "username": uname_a,
        "email": f"{uname_a}@example.com",
        "password": "Password123!",
        "role": "researcher"
    })
    res_a = client.post("/api/v1/auth/login", json={"username_or_email": uname_a, "password": "Password123!"})
    token_a = res_a.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # 2. Kullanıcı B kaydı ve girişi
    uname_b = f"researcher_b_{ts}"
    client.post("/api/v1/auth/register", json={
        "username": uname_b,
        "email": f"{uname_b}@example.com",
        "password": "Password123!",
        "role": "researcher"
    })
    res_b = client.post("/api/v1/auth/login", json={"username_or_email": uname_b, "password": "Password123!"})
    token_b = res_b.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # Admin token
    res_admin = client.post("/api/v1/auth/login", json={"username_or_email": "admin", "password": "admin"})
    admin_token = res_admin.json()["access_token"]
    headers_admin = {"Authorization": f"Bearer {admin_token}"}

    # 3. Kullanıcı A bir dosya yükler
    upload_res = client.post(
        "/api/v1/files/upload",
        files={"file": (f"user_a_private_{ts}.txt", b"Gizli Arastirma Verisi A", "text/plain")},
        headers=headers_a
    )
    assert upload_res.status_code == 201
    file_id_a = upload_res.json()["file_id"]

    # 4. Kullanıcı A kendi dosyasını görebilir
    get_a = client.get(f"/api/v1/files/{file_id_a}", headers=headers_a)
    assert get_a.status_code == 200

    # 5. Kullanıcı B, Kullanıcı A'nın dosyasını listeleyemez ve okuyamaz (403 veya 404)
    list_b = client.get("/api/v1/files/", headers=headers_b)
    assert list_b.status_code == 200
    b_file_ids = [f["file_id"] for f in list_b.json()]
    assert file_id_a not in b_file_ids

    get_b = client.get(f"/api/v1/files/{file_id_a}", headers=headers_b)
    assert get_b.status_code in (403, 404)

    # 6. Kullanıcı B bağlı dokümanları sorgulamaya kalktığında da 403 veya 404 almalıdır
    doc_b = client.get(f"/api/v1/datasets/documents/by-file/{file_id_a}", headers=headers_b)
    assert doc_b.status_code in (403, 404)

    # 7. Admin Kullanıcı A'nın dosyasına erişebilir
    get_admin = client.get(f"/api/v1/files/{file_id_a}", headers=headers_admin)
    assert get_admin.status_code == 200


def test_rbac_crud_authorization_matrix(client: TestClient):
    """
    Kapsamlı RBAC CRUD Yetki Matrisi Testi:
    Admin, Researcher ve Viewer rolleri için Create, Read, Update, Delete işlemlerini
    kendi kaynağı, başka kullanıcının kaynağı ve genel demo (owner_id=None) kaynağı üzerinde doğrular.
    """
    global_rate_limiter.reset()
    import time
    from backend.database import SessionLocal
    from backend.models import FileRecord
    from backend.security.password import hash_password

    ts = int(time.time() * 1000)

    # 1. Kullanıcıları oluştur veya hazırla
    db = SessionLocal()
    try:
        # Viewer kullanıcısı
        viewer_u = db.query(UserRecord).filter(UserRecord.username == f"viewer_{ts}").first()
        if not viewer_u:
            viewer_u = UserRecord(
                username=f"viewer_{ts}",
                email=f"viewer_{ts}@example.com",
                hashed_password=hash_password("ViewerPass123!"),
                role="viewer",
                is_active=True
            )
            db.add(viewer_u)

        # Demo dosyası (owner_id = None)
        demo_file = FileRecord(
            file_id=f"demo_file_{ts}",
            original_name="demo_dataset.txt",
            relative_path=f"demo_dataset_{ts}.txt",
            mime_type="text/plain",
            size_bytes=100,
            sha256="0" * 64,
            owner_id=None,  # Genel demo verisi
        )
        db.add(demo_file)
        db.commit()
    finally:
        db.close()

    # Token alımları
    login_admin = client.post("/api/v1/auth/login", json={"username_or_email": "admin", "password": "admin"})
    admin_tok = login_admin.json()["access_token"]
    headers_admin = {"Authorization": f"Bearer {admin_tok}"}

    login_res = client.post("/api/v1/auth/login", json={"username_or_email": "researcher", "password": "researcher123"})
    res_tok = login_res.json()["access_token"]
    headers_res = {"Authorization": f"Bearer {res_tok}"}

    login_view = client.post("/api/v1/auth/login", json={"username_or_email": f"viewer_{ts}", "password": "ViewerPass123!"})
    view_tok = login_view.json()["access_token"]
    headers_view = {"Authorization": f"Bearer {view_tok}"}

    demo_id = f"demo_file_{ts}"

    # --- 1. VIEWER ROLÜ ---
    # Read: Demo dosyasını okuyabilir
    r_view_demo = client.get(f"/api/v1/files/{demo_id}", headers=headers_view)
    assert r_view_demo.status_code == 200

    # Create: Dosya yükleyemez (403 Forbidden)
    c_view = client.post(
        "/api/v1/files/upload",
        files={"file": ("viewer_test.txt", b"viewer content", "text/plain")},
        headers=headers_view
    )
    assert c_view.status_code == 403

    # Update: Demo dosyasını güncelleyemez (403 Forbidden)
    u_view = client.patch(
        f"/api/v1/files/{demo_id}",
        json={"security_level": "restricted"},
        headers=headers_view
    )
    assert u_view.status_code == 403

    # Delete: Demo dosyasını silemez (403 Forbidden)
    d_view = client.delete(f"/api/v1/files/{demo_id}", headers=headers_view)
    assert d_view.status_code == 403

    # --- 2. RESEARCHER ROLÜ ---
    # Create: Kendi dosyasını oluşturabilir
    c_res = client.post(
        "/api/v1/files/upload",
        files={"file": (f"res_file_{ts}.txt", b"researcher content", "text/plain")},
        headers=headers_res
    )
    assert c_res.status_code == 201
    res_file_id = c_res.json()["file_id"]

    # Read: Demo dosyasını ve kendi dosyasını okuyabilir
    assert client.get(f"/api/v1/files/{demo_id}", headers=headers_res).status_code == 200
    assert client.get(f"/api/v1/files/{res_file_id}", headers=headers_res).status_code == 200

    # Update: Kendi dosyasını güncelleyebilir
    u_res_own = client.patch(
        f"/api/v1/files/{res_file_id}",
        json={"security_level": "INTERNAL"},
        headers=headers_res
    )
    assert u_res_own.status_code == 200

    # Update: Demo dosyasını DEĞİŞTİREMEZ (403 Forbidden)
    u_res_demo = client.patch(
        f"/api/v1/files/{demo_id}",
        json={"security_level": "RESTRICTED"},
        headers=headers_res
    )
    assert u_res_demo.status_code == 403

    # Delete: Demo dosyasını SİLEMEZ (403 Forbidden)
    d_res_demo = client.delete(f"/api/v1/files/{demo_id}", headers=headers_res)
    assert d_res_demo.status_code == 403

    # Delete: Kendi dosyasını silebilir
    d_res_own = client.delete(f"/api/v1/files/{res_file_id}", headers=headers_res)
    assert d_res_own.status_code == 204

    # --- 3. ADMIN ROLÜ ---
    # Read: Demo dosyasını okuyabilir
    assert client.get(f"/api/v1/files/{demo_id}", headers=headers_admin).status_code == 200

    # Update: Demo dosyasını güncelleyebilir (Admin yetkisi tamdır)
    u_admin_demo = client.patch(
        f"/api/v1/files/{demo_id}",
        json={"security_level": "PUBLIC"},
        headers=headers_admin
    )
    assert u_admin_demo.status_code == 200

    # Delete: Demo dosyasını silebilir (Admin yetkisi tamdır)
    d_admin_demo = client.delete(f"/api/v1/files/{demo_id}", headers=headers_admin)
    assert d_admin_demo.status_code == 204


def test_path_traversal_prevention_in_storage_manager():
    """Dosya sistemi düzeyinde directory traversal saldırılarının engellendiğini doğrula."""
    from backend.storage import storage_manager
    
    with pytest.raises(PermissionError, match="Path traversal detected"):
        storage_manager.get_file_path("../../etc/passwd")

    with pytest.raises(PermissionError, match="Path traversal detected"):
        storage_manager.get_file_path("../../../../../system/secrets.json")


def test_file_upload_user_isolation_and_deduplication(client):
    """Farklı kullanıcılar aynı içeriği yüklediğinde kullanıcı kayıtlarının izole edildiğini doğrula."""
    # Kullanıcı A (researcher) ve Kullanıcı B (admin) için token al
    login_a = client.post("/api/v1/auth/login", json={"username_or_email": "researcher", "password": "researcher123"})
    token_a = login_a.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    login_b = client.post("/api/v1/auth/login", json={"username_or_email": "admin", "password": "admin"})
    token_b = login_b.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}
    import uuid
    unique_marker = str(uuid.uuid4())
    common_content = f"Ortak icerik metni - deduplication ve izolasyon testi {unique_marker}".encode()

    # 1. Kullanıcı A dosyayı yükler
    res_a = client.post(
        "/api/v1/files/upload",
        files={"file": ("shared_doc.txt", common_content, "text/plain")},
        headers=headers_a
    )
    assert res_a.status_code == 201
    data_a = res_a.json()
    file_id_a = data_a["file_id"]
    assert data_a["is_duplicate"] is False

    # 2. Kullanıcı B aynı fiziksel içeriği yükler
    res_b = client.post(
        "/api/v1/files/upload",
        files={"file": ("shared_doc.txt", common_content, "text/plain")},
        headers=headers_b
    )
    assert res_b.status_code == 201
    data_b = res_b.json()
    file_id_b = data_b["file_id"]

    # Kullanıcı B, Kullanıcı A'nın file_id'sini ALMAMALI; kendine ait ayrı bir kayıt almalıdır!
    assert file_id_b != file_id_a, "Farklı kullanıcılar aynı içeriği yüklediğinde dosya ID'leri karışmamalıdır!"
    assert data_b["is_duplicate"] is False

    # 3. Kullanıcı A aynı dosyayı ikinci kez yüklerse, kendisine ait duplicate uyarısı almalıdır
    res_a_dup = client.post(
        "/api/v1/files/upload",
        files={"file": ("shared_doc.txt", common_content, "text/plain")},
        headers=headers_a
    )
    assert res_a_dup.status_code == 201
    data_a_dup = res_a_dup.json()
    assert data_a_dup["file_id"] == file_id_a
    assert data_a_dup["is_duplicate"] is True


def test_shared_physical_file_lifecycle_and_deletion(client):
    """Kullanıcı A dosyasını sildiğinde, aynı içeriği kullanan Kullanıcı B'nin fiziksel dosyasının silinmediğini doğrula."""
    from backend.storage import storage_manager

    login_a = client.post("/api/v1/auth/login", json={"username_or_email": "researcher", "password": "researcher123"})
    headers_a = {"Authorization": f"Bearer {login_a.json()['access_token']}"}
    login_b = client.post("/api/v1/auth/login", json={"username_or_email": "admin", "password": "admin"})
    headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

    import uuid
    content = f"Shared content for deletion test {uuid.uuid4()}".encode()

    # 1. Kullanıcı A ve Kullanıcı B aynı içeriği yükler
    res_a = client.post("/api/v1/files/upload", files={"file": ("doc_a.txt", content, "text/plain")}, headers=headers_a)
    assert res_a.status_code == 201
    file_id_a = res_a.json()["file_id"]

    res_b = client.post("/api/v1/files/upload", files={"file": ("doc_b.txt", content, "text/plain")}, headers=headers_b)
    assert res_b.status_code == 201
    file_id_b = res_b.json()["file_id"]

    # File details üzerinden relative_path al
    info_a = client.get(f"/api/v1/files/{file_id_a}", headers=headers_a).json()
    rel_path = info_a["relative_path"]

    # Fiziksel dosya diskte var olmalıdır
    assert storage_manager.file_exists(rel_path) is True

    # 2. Kullanıcı A kendi dosyasını siler
    del_a = client.delete(f"/api/v1/files/{file_id_a}", headers=headers_a)
    assert del_a.status_code == 204

    # Kullanıcı A'nın kaydı silinmiş olmalıdır (404)
    get_a = client.get(f"/api/v1/files/{file_id_a}", headers=headers_a)
    assert get_a.status_code == 404

    # KRİTİK: Kullanıcı B'nin dosyası ve diskteki fiziksel dosya HALA VAR OLMALIDIR!
    get_b = client.get(f"/api/v1/files/{file_id_b}", headers=headers_b)
    assert get_b.status_code == 200
    assert storage_manager.file_exists(rel_path) is True

    # 3. Kullanıcı B de kendi dosyasını sildiğinde fiziksel dosya artık diskten silinmelidir
    del_b = client.delete(f"/api/v1/files/{file_id_b}", headers=headers_b)
    assert del_b.status_code == 204

    # Artık hiçbir referans kalmadığından fiziksel dosya da silinmiş olmalıdır
    assert storage_manager.file_exists(rel_path) is False


def test_registry_dir_restriction_in_inference_load(client):
    """İstemcinin 'models' dışında yetkisiz bir registry_dir belirtmesinin engellendiğini doğrula."""
    login = client.post("/api/v1/auth/login", json={"username_or_email": "admin", "password": "admin"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    # İstemci keyfi bir sistem dizini belirttiğinde HTTP 400 ile reddedilmeli
    res = client.post("/api/v1/inference/load", json={
        "model_name": "test_model",
        "registry_dir": "arbitrary_custom_dir"
    }, headers=headers)
    assert res.status_code == 400
    assert "Yetkisiz veya geçersiz registry dizini" in res.json().get("detail", "")


def test_malicious_checkpoint_weights_only_blocking(tmp_path, client):
    """Zararlı pickle kodu içeren checkpoint dosyasının kayıt, yükleme ve inference aşamalarında açıkça reddedildiğini doğrula."""
    import torch
    import pytest
    import uuid
    from src.registry.model_registry import ModelRegistry

    flag_file = tmp_path / "pwned.txt"

    class Exploit:
        def __reduce__(self):
            import os
            return (os.system, (f"touch {flag_file}",))

    malicious_data = {"payload": Exploit()}
    chk_path = tmp_path / "malicious.pt"
    torch.save(malicious_data, chk_path)

    registry = ModelRegistry("models")
    unique_model_name = f"malicious-model-{uuid.uuid4().hex[:8]}"

    # 1. Kayıt aşamasında checkpoint güvenliği: Açıkça ValueError fırlatılmalıdır
    with pytest.raises(ValueError, match="weights_only=True"):
        registry.register_model(
            model_name=unique_model_name,
            version="1.0.0",
            checkpoint_path=chk_path
        )

    # 2. Doğrudan yükleme denendiğinde de weights_only=True ile reddedildiğini doğrula
    with pytest.raises(Exception):
        torch.load(chk_path, weights_only=True)

    # InferencePipeline'ın gerçek checkpoint giriş noktasını da doğrudan
    # doğrula; yalnızca bilinmeyen model için 400 dönmesi yeterli değildir.
    from src.inference.pipeline import InferencePipeline
    with pytest.raises(ValueError, match="weights_only=True"):
        InferencePipeline.from_pretrained(chk_path, tmp_path / "unused-tokenizer.json")

    # 3. Inference /load API endpoint'i üzerinden çağrıldığında HTTP 400 döndürüldüğünü doğrula
    login = client.post("/api/v1/auth/login", json={"username_or_email": "admin", "password": "admin"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    res_load = client.post("/api/v1/inference/load", json={
        "model_name": unique_model_name,
        "version": "1.0.0"
    }, headers=headers)
    assert res_load.status_code == 400

    # İstismar komutunun ASLA çalıştırılmadığını doğrula
    assert not flag_file.exists(), "GÜVENLİK AÇIĞI: Zararlı checkpoint içindeki komut çalıştırıldı!"


def test_concurrent_file_uploads_and_deletions(client):
    """Farklı kullanıcıların aynı içeriğe yönelik eşzamanlı yükleme ve silme işlemlerinde veri kaybı ve yarış durumu olmadığını doğrula."""
    from concurrent.futures import ThreadPoolExecutor
    from backend.storage import storage_manager
    import uuid

    # 5 farklı kayıtlı kullanıcı oluştur ve token'larını al
    user_headers = []
    for i in range(5):
        u_name = f"usr_conc_{uuid.uuid4().hex[:8]}"
        client.post("/api/v1/auth/register", json={
            "username": u_name,
            "email": f"{u_name}@test.com",
            "password": "Password123!",
            "role": "researcher"
        })
        res_login = client.post("/api/v1/auth/login", json={"username_or_email": u_name, "password": "Password123!"})
        user_headers.append({"Authorization": f"Bearer {res_login.json()['access_token']}"})

    common_content = f"Concurrent test payload content {uuid.uuid4()}".encode()

    # 1. Beş farklı kullanıcı aynı anda aynı içeriği yükler
    def do_upload(idx):
        return client.post(
            "/api/v1/files/upload",
            files={"file": (f"concurrent_doc_{idx}.txt", common_content, "text/plain")},
            headers=user_headers[idx]
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        upload_results = list(executor.map(do_upload, range(5)))

    file_ids = []
    for res in upload_results:
        assert res.status_code == 201
        data = res.json()
        file_ids.append(data["file_id"])

    # Her kullanıcının kendine ait benzersiz file_id aldığını doğrula
    assert len(set(file_ids)) == 5

    # Fiziksel dosya tekil olarak diskte bulunmalıdır
    first_info = client.get(f"/api/v1/files/{file_ids[0]}", headers=user_headers[0]).json()
    rel_path = first_info["relative_path"]
    assert storage_manager.file_exists(rel_path) is True

    # 2. İlk 4 kullanıcı dosyalarını eşzamanlı olarak siler
    def do_delete(idx):
        return client.delete(f"/api/v1/files/{file_ids[idx]}", headers=user_headers[idx])

    with ThreadPoolExecutor(max_workers=4) as executor:
        delete_results = list(executor.map(do_delete, range(4)))

    for res in delete_results:
        assert res.status_code in {204, 404}

    # KRİTİK: 5. kullanıcının dosyası hala veritabanında olduğundan fiziksel dosya silinmemiş OLMALIDIR!
    res_last = client.get(f"/api/v1/files/{file_ids[4]}", headers=user_headers[4])
    assert res_last.status_code == 200
    assert storage_manager.file_exists(rel_path) is True

    # 3. Son referansı sil
    del_last = client.delete(f"/api/v1/files/{file_ids[4]}", headers=user_headers[4])
    assert del_last.status_code == 204

    # Artık tüm referanslar bittiğinden fiziksel dosya da temizlenmiş olmalıdır
    assert storage_manager.file_exists(rel_path) is False

    # Upload ve son referansın silinmesi kesiştiğinde yeni kaydın fiziksel
    # dosyası silinmemelidir. Bu, yalnızca "önce tüm upload, sonra tüm delete"
    # senaryosundan daha güçlü bir interleaving kontrolüdür.
    from threading import Barrier
    for round_no in range(3):
        race_content = f"Interleaved upload/delete {uuid.uuid4()}".encode()
        baseline = client.post(
            "/api/v1/files/upload",
            files={"file": (f"race-base-{round_no}.txt", race_content, "text/plain")},
            headers=user_headers[0]
        )
        assert baseline.status_code == 201
        baseline_id = baseline.json()["file_id"]
        barrier = Barrier(2)

        def upload_during_delete():
            barrier.wait()
            return client.post(
                "/api/v1/files/upload",
                files={"file": (f"race-new-{round_no}.txt", race_content, "text/plain")},
                headers=user_headers[1]
            )

        def delete_during_upload():
            barrier.wait()
            return client.delete(f"/api/v1/files/{baseline_id}", headers=user_headers[0])

        with ThreadPoolExecutor(max_workers=2) as executor:
            upload_result, delete_result = list(executor.map(
                lambda fn: fn(), (upload_during_delete, delete_during_upload)
            ))
        assert upload_result.status_code == 201
        assert delete_result.status_code == 204
        new_id = upload_result.json()["file_id"]
        new_info = client.get(f"/api/v1/files/{new_id}", headers=user_headers[1])
        assert new_info.status_code == 200
        assert storage_manager.file_exists(new_info.json()["relative_path"]) is True
        assert client.delete(f"/api/v1/files/{new_id}", headers=user_headers[1]).status_code == 204


def test_sft_preflight_missing_response_column(tmp_path, monkeypatch):
    """SFT veri kümesinde 'instruction' bulunup zorunlu 'response' sütununun eksik olduğu durumda eğitimin engellendiğini doğrula."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest
    from backend.database import SessionLocal
    from backend.models import DatasetVersion, TokenizerRecord
    from backend.services.training_service import TrainingService

    # 1. instruction sütunu olan ama response sütunu OLMAYAN bozuk bir SFT parquet hazırla
    bad_table = pa.Table.from_pydict({
        "instruction": ["Soru 1", "Soru 2"],
        "split": ["train", "validation"]
    })
    bad_parquet = tmp_path / "bad_sft.parquet"
    pq.write_table(bad_table, bad_parquet)

    # 2. Hem instruction hem response içeren geçerli bir SFT parquet hazırla
    valid_table = pa.Table.from_pydict({
        "instruction": ["Soru 1", "Soru 2"],
        "response": ["Cevap 1", "Cevap 2"],
        "split": ["train", "validation"]
    })
    valid_parquet = tmp_path / "valid_sft.parquet"
    pq.write_table(valid_table, valid_parquet)

    db = SessionLocal()
    tok_record = TokenizerRecord(tokenizer_id="tok-1", vocab_size=200, storage_path=str(tmp_path / "dummy_tok"))

    try:
        # Mock load_artifacts to return the bad dataset
        monkeypatch.setattr(
            "backend.services.training_service.load_artifacts",
            lambda db, did, tid: (
                DatasetVersion(dataset_id="ds-bad", storage_path=str(bad_parquet)),
                tok_record,
                None,
                {}
            )
        )
        service = TrainingService(db)

        # 1. instruction var ama response yok -> ValueError fırlatılmalı
        with pytest.raises(ValueError, match="zorunlu 'response' sütunu eksik"):
            service.create_job(
                job_name="Bad SFT Schema Test",
                model_name="sft-schema-fail-model",
                job_type="SFT",
                dataset_id="ds-bad",
                tokenizer_id="tok-1",
                config={"epochs": 1, "vocab_size": 200}
            )

        # 2. Hem instruction hem response olmayan geçersiz şema -> ValueError fırlatılmalı
        invalid_table = pa.Table.from_pydict({
            "unrelated_column": [1, 2]
        })
        invalid_parquet = tmp_path / "invalid_sft.parquet"
        pq.write_table(invalid_table, invalid_parquet)

        monkeypatch.setattr(
            "backend.services.training_service.load_artifacts",
            lambda db, did, tid: (
                DatasetVersion(dataset_id="ds-invalid", storage_path=str(invalid_parquet)),
                tok_record,
                None,
                {}
            )
        )
        with pytest.raises(ValueError, match="zorunlu \\('instruction', 'response'\\) veya 'text' sütunları bulunamadı"):
            service.create_job(
                job_name="Invalid SFT Schema Test",
                model_name="sft-schema-invalid-model",
                job_type="SFT",
                dataset_id="ds-invalid",
                tokenizer_id="tok-1",
                config={"epochs": 1, "vocab_size": 200}
            )

        # token_ids SFT için geçerli bir alternatif değildir; runtime bu
        # formatı değil instruction/response veya JSON text formatını okur.
        token_only_table = pa.Table.from_pydict({
            "instruction": ["Soru 1", "Soru 2"],
            "token_ids": [[1, 2], [3, 4]],
            "split": ["train", "validation"]
        })
        token_only_parquet = tmp_path / "token_only_sft.parquet"
        pq.write_table(token_only_table, token_only_parquet)
        monkeypatch.setattr(
            "backend.services.training_service.load_artifacts",
            lambda db, did, tid: (
                DatasetVersion(dataset_id="ds-token-only", storage_path=str(token_only_parquet)),
                tok_record,
                None,
                {}
            )
        )
        with pytest.raises(ValueError, match="zorunlu 'response' sütunu eksik"):
            service.create_job(
                job_name="Token Only SFT Schema Test",
                model_name="sft-schema-token-only-model",
                job_type="SFT",
                dataset_id="ds-token-only",
                tokenizer_id="tok-1",
                config={"epochs": 1, "vocab_size": 200}
            )

        # 3. Hem instruction hem response içeren geçerli şema -> Şema doğrulaması başarıyla geçmeli
        # (Şema kontrolünü geçtiği için bir sonraki preflight adımı olan base_model gereksinimine ulaşır)
        monkeypatch.setattr(
            "backend.services.training_service.load_artifacts",
            lambda db, did, tid: (
                DatasetVersion(dataset_id="ds-valid", storage_path=str(valid_parquet)),
                tok_record,
                None,
                {}
            )
        )
        with pytest.raises(ValueError, match="Fine-tuning requires a base model"):
            service.create_job(
                job_name="Valid SFT Schema Test",
                model_name="sft-schema-pass-model",
                job_type="SFT",
                dataset_id="ds-valid",
                tokenizer_id="tok-1",
                config={"epochs": 1, "vocab_size": 200}
            )

    finally:
        db.close()
