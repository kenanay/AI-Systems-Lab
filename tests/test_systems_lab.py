"""
tests/test_systems_lab.py

Unit tests for Systems for AI Lab & Hardware Simulation Engine
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.simulators.systems_hardware import (
    RooflineModelEngine,
    QuantizationSimulator,
    GPUMemorySimulator,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_roofline_presets():
    """Verify hardware presets list."""
    presets = RooflineModelEngine.get_presets()
    assert len(presets) >= 4
    preset_ids = [p.id for p in presets]
    assert "h100_sxm" in preset_ids
    assert "rtx_4090" in preset_ids


def test_roofline_analysis_memory_and_compute_bound():
    """Verify arithmetic intensity, knee point, and memory vs compute bound diagnosis."""
    # 1. Memory-Bound Scenario (e.g. single-token autoregressive generation: 2 FLOPs per 2 bytes = 1 FLOP/Byte)
    res_mem = RooflineModelEngine.analyze(
        peak_tflops=330.0,
        peak_bandwidth_gbps=1008.0,
        total_flops=2.0e9,
        total_bytes=2.0e9,
        hardware_name="RTX 4090"
    )
    assert res_mem["operational_intensity"] == 1.0
    assert res_mem["is_memory_bound"] is True
    assert "Memory-Bound" in res_mem["bound_type"]
    assert res_mem["attainable_tflops"] < res_mem["peak_tflops"]

    # 2. Compute-Bound Scenario (e.g. dense matrix multiplication batch: 1000 FLOPs/Byte)
    res_comp = RooflineModelEngine.analyze(
        peak_tflops=330.0,
        peak_bandwidth_gbps=1008.0,
        total_flops=1000.0e9,
        total_bytes=1.0e9,
        hardware_name="RTX 4090"
    )
    assert res_comp["operational_intensity"] == 1000.0
    assert res_comp["is_memory_bound"] is False
    assert "Compute-Bound" in res_comp["bound_type"]
    assert res_comp["attainable_tflops"] == res_comp["peak_tflops"]
    assert res_comp["efficiency_pct"] == 100.0


def test_quantization_simulator_metrics():
    """Verify FP32, FP16, INT8, and INT4 quantization errors and histograms."""
    res = QuantizationSimulator.simulate(
        distribution="normal",
        num_elements=5000,
        std=0.4,
        outlier_ratio=0.01
    )
    assert res["num_elements"] == 5000
    metrics = res["metrics"]

    # Compression ratios
    assert metrics["fp32"]["compression_ratio"] == 1.0
    assert metrics["fp16"]["compression_ratio"] == 2.0
    assert metrics["int8"]["compression_ratio"] == 4.0
    assert metrics["int4"]["compression_ratio"] == 8.0

    # Error hierarchy: INT4 error > INT8 error > FP16 error
    assert metrics["int4"]["mse"] >= metrics["int8"]["mse"]
    assert metrics["int8"]["mse"] >= metrics["fp16"]["mse"]

    # SNR: higher is better
    assert metrics["int8"]["snr_db"] > metrics["int4"]["snr_db"]

    # Histogram checks
    assert len(res["histogram"]["bin_centers"]) == 20
    assert len(res["histogram"]["int8_counts"]) == 20
    assert sum(res["histogram"]["fp32_counts"]) == 5000


def test_gpu_memory_simulator_training_and_inference():
    """Verify GPU VRAM memory breakdown and OOM risk detection."""
    # 1. Training mode on 7B model
    train_res = GPUMemorySimulator.simulate(
        mode="training",
        param_count_billions=7.0,
        precision="fp16",
        batch_size=4,
        seq_len=2048,
        optimizer_type="adamw",
        activation_checkpointing=True,
        gpu_capacity_gb=24.0
    )
    assert train_res["mode"] == "training"
    assert train_res["breakdown_gb"]["weights"] > 0
    assert train_res["breakdown_gb"]["gradients"] > 0
    assert train_res["breakdown_gb"]["optimizer"] > 0
    assert train_res["breakdown_gb"]["total"] > 24.0
    assert train_res["oom_risk"] is True
    assert "OOM" in train_res["status"]

    # 2. Inference mode on 7B INT4 (Quantized like AWQ/GPTQ) fits easily in 24GB
    inf_res = GPUMemorySimulator.simulate(
        mode="inference",
        param_count_billions=7.0,
        precision="int4",
        batch_size=1,
        seq_len=2048,
        gpu_capacity_gb=24.0
    )
    assert inf_res["mode"] == "inference"
    assert inf_res["breakdown_gb"]["weights"] < 5.0
    assert inf_res["oom_risk"] is False
    assert "FIT" in inf_res["status"]
    assert inf_res["headroom_gb"] > 10.0


def test_systems_lab_api_endpoints(client):
    """Verify FastAPI router endpoints for Systems Lab."""
    # 1. GET presets
    p_resp = client.get("/api/v1/systems-lab/roofline/presets")
    assert p_resp.status_code == 200
    assert len(p_resp.json()) >= 4

    # 2. POST analyze roofline
    roof_payload = {
        "peak_tflops": 1000.0,
        "peak_bandwidth_gbps": 3350.0,
        "total_flops": 1.0e10,
        "total_bytes": 1.0e8,
        "hardware_name": "H100 SXM"
    }
    r_resp = client.post("/api/v1/systems-lab/roofline/analyze", json=roof_payload)
    assert r_resp.status_code == 200
    data = r_resp.json()
    assert data["operational_intensity"] == 100.0
    assert "curve_points" in data

    # 3. POST simulate quantization
    q_payload = {
        "distribution": "uniform",
        "num_elements": 2000,
        "std": 0.5,
        "outlier_ratio": 0.01
    }
    q_resp = client.post("/api/v1/systems-lab/quantization/simulate", json=q_payload)
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert "metrics" in q_data
    assert "int8" in q_data["metrics"]

    # 4. POST simulate memory
    m_payload = {
        "mode": "inference",
        "param_count_billions": 1.1,
        "precision": "fp16",
        "batch_size": 2,
        "seq_len": 1024,
        "gpu_capacity_gb": 8.0
    }
    m_resp = client.post("/api/v1/systems-lab/memory/simulate", json=m_payload)
    assert m_resp.status_code == 200
    m_data = m_resp.json()
    assert m_data["breakdown_gb"]["total"] < 8.0
    assert m_data["oom_risk"] is False
