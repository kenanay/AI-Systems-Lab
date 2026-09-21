"""
tests/test_inference_server.py

Inference Server Test Suite

FastAPI endpoints'leri test eder.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from src.server.inference_server import app

# Test client
client = TestClient(app)


def test_root():
    """Test root endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Root Endpoint (/)")
    print("=" * 60)
    
    response = client.get("/")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data
    
    print("✓ Root endpoint working")


def test_health():
    """Test health check endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Health Endpoint (/health)")
    print("=" * 60)
    
    response = client.get("/health")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    
    print("✓ Health endpoint working")


def test_ready_no_model():
    """Test readiness endpoint (without loaded model)."""
    print("\n" + "=" * 60)
    print("TEST: Ready Endpoint (/ready) - No Model")
    print("=" * 60)
    
    response = client.get("/ready")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] == False  # Model not loaded yet
    assert data["model_loaded"] == False
    assert "version" in data
    
    print("✓ Ready endpoint working (not ready state)")


def test_generate_no_model():
    """Test generate endpoint without loaded model."""
    print("\n" + "=" * 60)
    print("TEST: Generate Endpoint (/generate) - No Model")
    print("=" * 60)
    
    request_data = {
        "prompt": "Türkiye",
        "max_length": 20,
        "temperature": 0.8
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    response = client.post("/generate", json=request_data)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Should return 503 (service unavailable) when model not loaded
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]
    
    print("✓ Generate endpoint correctly rejects without model")


def test_models_endpoint():
    """Test models listing endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Models Endpoint (/models)")
    print("=" * 60)
    
    response = client.get("/models")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "count" in data
    assert data["count"] == 0  # No models loaded yet
    
    print("✓ Models endpoint working")


def test_validation_errors():
    """Test request validation."""
    print("\n" + "=" * 60)
    print("TEST: Request Validation")
    print("=" * 60)
    
    # Missing prompt
    response = client.post("/generate", json={"max_length": 20})
    print(f"Missing prompt - Status: {response.status_code}")
    assert response.status_code == 422  # Validation error
    
    # Invalid temperature
    response = client.post("/generate", json={
        "prompt": "test",
        "temperature": 5.0  # Too high (max=2.0)
    })
    print(f"Invalid temperature - Status: {response.status_code}")
    assert response.status_code == 422
    
    # Invalid max_length
    response = client.post("/generate", json={
        "prompt": "test",
        "max_length": 0  # Too low (min=1)
    })
    print(f"Invalid max_length - Status: {response.status_code}")
    assert response.status_code == 422
    
    print("✓ Validation working correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("INFERENCE SERVER TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_root,
        test_health,
        test_ready_no_model,
        test_generate_no_model,
        test_models_endpoint,
        test_validation_errors
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ Test failed: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ Test error: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
