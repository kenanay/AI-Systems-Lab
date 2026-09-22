"""
tests/test_model_export.py

Unit and Integration Tests for Model Export & Quantization Pipeline:
- FP16, INT8 Dynamic, and INT4 Weight-Only Quantization
- GGUF v3 Binary Writer & Parsing
- ONNX and TorchScript Exporters
- Deployment Snippet Generation
- FastAPI Model Export & Download Endpoints
"""

import os
import struct
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from src.model.gpt import GPTModel, GPTConfig
from src.registry.model_registry import ModelRegistry
from src.export.quantization import ModelQuantizer
from src.export.gguf_writer import (
    GGUFWriter,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
)
from src.export.model_exporter import ModelExporter
from backend.main import app


@pytest.fixture
def small_config():
    return GPTConfig(
        vocab_size=100,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        dropout=0.0
    )


@pytest.fixture
def small_model(small_config):
    torch.manual_seed(42)
    model = GPTModel(small_config)
    model.eval()
    return model


class TestModelQuantizer:
    """Test suite for ModelQuantizer precision reductions and error metrics."""

    def test_compute_metrics(self):
        t1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        t2 = torch.tensor([1.05, 1.95, 3.02, 3.98])
        metrics = ModelQuantizer.compute_metrics(t1, t2)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "snr_db" in metrics
        assert metrics["mse"] > 0
        assert metrics["snr_db"] > 20.0

    def test_quantize_fp16(self, small_model):
        fp16_model, metrics = ModelQuantizer.quantize_fp16(small_model)
        for param in fp16_model.parameters():
            assert param.dtype == torch.float16
        assert metrics["precision"] == "fp16"
        assert metrics["memory_reduction_pct"] == 50.0

    def test_quantize_int8_dynamic(self, small_model):
        int8_model, metrics = ModelQuantizer.quantize_int8_dynamic(small_model)
        assert metrics["precision"] == "int8"
        assert metrics["quantized_linear_layers"] > 0
        assert metrics["memory_reduction_pct"] > 60.0

    def test_quantize_int4_weight_only(self, small_model):
        int4_model, metrics = ModelQuantizer.quantize_int4_weight_only(small_model, group_size=32)
        assert metrics["precision"] == "int4"
        assert metrics["group_size"] == 32
        assert metrics["memory_reduction_pct"] >= 75.0
        assert "avg_snr_db" in metrics


class TestGGUFWriter:
    """Test suite for pure-Python GGUF v3 binary serialization."""

    def test_gguf_serialization_and_magic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_model.gguf"
            writer = GGUFWriter(str(out_file), architecture="gpt2")

            # Add metadata
            writer.add_string("general.name", "mini-gpt")
            writer.add_uint32("gpt2.context_length", 128)
            writer.add_float32("test.metric", 0.95)
            writer.add_array("general.tags", ["test", "quant"])

            # Add dummy tensors
            weight_f32 = torch.randn(8, 8).numpy()
            writer.add_tensor("linear.weight_f32", weight_f32, GGML_TYPE_F32)

            weight_f16 = torch.randn(8, 8).numpy()
            writer.add_tensor("linear.weight_f16", weight_f16, GGML_TYPE_F16)

            weight_q8 = torch.randn(32).numpy()
            writer.add_tensor("linear.weight_q8", weight_q8, GGML_TYPE_Q8_0)

            weight_q4 = torch.randn(32).numpy()
            writer.add_tensor("linear.weight_q4", weight_q4, GGML_TYPE_Q4_0)

            bytes_written = writer.write()
            assert out_file.exists()
            assert bytes_written == out_file.stat().st_size

            # Verify header
            with open(out_file, "rb") as f:
                magic = f.read(4)
                assert magic == b"GGUF"
                version = struct.unpack("<I", f.read(4))[0]
                assert version == 3
                tensor_count = struct.unpack("<Q", f.read(8))[0]
                assert tensor_count == 4
                metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
                assert metadata_kv_count == 6  # general.architecture, general.alignment + 4 added


class TestModelExporter:
    """Test suite for ONNX, TorchScript, and pipeline exports."""

    def test_export_onnx(self, small_model, small_config):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "model.onnx"
            exporter = ModelExporter(registry_dir=tmp_dir)
            result = exporter.export_onnx(small_model, small_config, out_path, dynamic_shapes=True)

            assert out_path.exists()
            assert result["format"] == "onnx"
            assert result["file_size_mb"] > 0
            assert len(result["sha256"]) == 64

    def test_export_torchscript(self, small_model, small_config):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "model.pt"
            exporter = ModelExporter(registry_dir=tmp_dir)
            result = exporter.export_torchscript(small_model, small_config, out_path, method="trace")

            assert out_path.exists()
            assert result["format"] == "torchscript"

            # Verify loaded TorchScript JIT model produces output
            loaded_jit = torch.jit.load(str(out_path))
            dummy_input = torch.randint(0, small_config.vocab_size, (1, 4), dtype=torch.long)
            out = loaded_jit(dummy_input)
            assert out.shape == (1, 4, small_config.vocab_size)

    def test_export_gguf(self, small_model, small_config):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "model.gguf"
            exporter = ModelExporter(registry_dir=tmp_dir)
            result = exporter.export_gguf(small_model, small_config, out_path, quantization="int8", model_name="test-gpt")

            assert out_path.exists()
            assert result["format"] == "gguf"
            assert result["quantization"] == "int8"
            assert result["tensor_count"] > 0

    def test_deployment_snippets(self):
        onnx_snip = ModelExporter.generate_deployment_snippet("onnx", "model.onnx", "gpt")
        assert "onnxruntime as ort" in onnx_snip

        ts_snip = ModelExporter.generate_deployment_snippet("torchscript", "model.pt", "gpt")
        assert "torch.jit.load" in ts_snip

        gguf_snip = ModelExporter.generate_deployment_snippet("gguf", "model.gguf", "gpt")
        assert "FROM ./model.gguf" in gguf_snip
        assert "ollama create" in gguf_snip


class TestExportApiEndpoints:
    """Integration test suite for FastAPI /api/v1/models export & download endpoints."""

    @pytest.fixture
    def setup_registered_model(self, small_model, small_config):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # We patch ModelRegistry to use tmp_dir
            registry = ModelRegistry(registry_dir=tmp_dir)
            chk_path = Path(tmp_dir) / "checkpoint.pt"
            tok_path = Path(tmp_dir) / "tokenizer.model"
            tok_path.write_text("dummy tok")

            torch.save({
                "model_state_dict": small_model.state_dict(),
                "config": small_config.__dict__
            }, chk_path)

            meta = registry.register_model(
                model_name="export-test-llm",
                version="1.0.0",
                checkpoint_path=chk_path,
                tokenizer_path=tok_path,
                description="Export testing model",
                parameters=1000,
                training_config={
                    "model_config": {
                        "vocab_size": small_config.vocab_size,
                        "max_seq_len": small_config.max_seq_len,
                        "d_model": small_config.d_model,
                        "n_layers": small_config.n_layers,
                        "n_heads": small_config.n_heads,
                        "d_ff": small_config.d_ff
                    }
                }
            )
            yield tmp_dir, "export-test-llm", "1.0.0"

    def test_export_pipeline_and_listing(self, setup_registered_model):
        tmp_dir, model_name, version = setup_registered_model
        exporter = ModelExporter(registry_dir=tmp_dir)

        # 1. Export ONNX FP32
        res1 = exporter.export_pipeline(
            model_name=model_name,
            version=version,
            export_format="onnx",
            quantization="none"
        )
        assert res1["success"] is True
        assert res1["export_format"] == "onnx"
        assert Path(res1["file_path"]).exists()

        # 2. Export GGUF INT8
        res2 = exporter.export_pipeline(
            model_name=model_name,
            version=version,
            export_format="gguf",
            quantization="int8"
        )
        assert res2["success"] is True
        assert res2["export_format"] == "gguf"
        assert res2["quantization"] == "int8"

        # 3. List exports
        exports = exporter.list_exports(model_name, version=version)
        assert len(exports) == 2
        file_names = [e["file_name"] for e in exports]
        assert res1["file_name"] in file_names
        assert res2["file_name"] in file_names

    def test_fastapi_export_routes(self, monkeypatch, setup_registered_model):
        tmp_dir, model_name, version = setup_registered_model

        # Monkeypatch ModelExporter and ModelRegistry default registry_dir
        orig_exporter_init = ModelExporter.__init__
        orig_registry_init = ModelRegistry.__init__

        monkeypatch.setattr(ModelExporter, "__init__", lambda self, registry_dir="models": orig_exporter_init(self, registry_dir=tmp_dir))
        monkeypatch.setattr(ModelRegistry, "__init__", lambda self, registry_dir="models": orig_registry_init(self, registry_dir=tmp_dir))

        client = TestClient(app)

        # POST /export
        resp = client.post(
            f"/api/v1/models/{model_name}/export",
            json={
                "version": version,
                "export_format": "torchscript",
                "quantization": "none"
            }
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["export_format"] == "torchscript"
        file_name = data["file_name"]

        # GET /exports
        list_resp = client.get(f"/api/v1/models/{model_name}/exports")
        assert list_resp.status_code == 200
        exports = list_resp.json()
        assert len(exports) >= 1
        assert any(e["file_name"] == file_name for e in exports)

        # GET /download/{file_name}
        dl_resp = client.get(f"/api/v1/models/{model_name}/download/{file_name}")
        assert dl_resp.status_code == 200
        assert dl_resp.headers["content-type"] == "application/octet-stream"
        assert len(dl_resp.content) > 0

        # GET /download with non-existent file
        dl_404 = client.get(f"/api/v1/models/{model_name}/download/non_existent.onnx")
        assert dl_404.status_code == 404
