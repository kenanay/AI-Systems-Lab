"""
src/export/model_exporter.py

Comprehensive Model Export Pipeline:
- ONNX Export with dynamic shapes
- TorchScript JIT Export (trace / script)
- GGUF v3 Export for Ollama & llama.cpp
- Integration with ModelQuantizer (FP16, INT8, INT4)
- Deployment snippet generators for Python, ONNXRuntime, Ollama Modelfile, and C++

Author: Kenan AY
Location: Kütahya, TÜRKİYE
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

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

logger = logging.getLogger(__name__)


class GPTInferenceWrapper(nn.Module):
    """
    Sadece logits tensorü üreten ve çıkarım motorları (ONNX Runtime, TorchScript JIT)
    ile %100 uyumlu hafif sarmalayıcı.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(token_ids)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class ModelExporter:
    """
    Orchestrates multi-format model export and quantization.
    Saves outputs into the model's export directory in the Model Registry.
    """

    def __init__(self, registry_dir: str = "models"):
        self.registry = ModelRegistry(registry_dir=registry_dir)
        self.registry_dir = Path(registry_dir)

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Dosyanın SHA-256 hash'ini hesaplar."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def generate_deployment_snippet(
        export_format: str,
        file_name: str,
        model_name: str
    ) -> str:
        """
        Dışa aktarılan model için hazır üretim / dağıtım çalıştırma kod bloğu oluşturur.
        """
        if export_format == "onnx":
            return f"""# ONNX Runtime ile Yüksek Hızlı Çıkarım (Python)
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("{file_name}", providers=["CPUExecutionProvider"])
input_ids = np.array([[1, 45, 128, 92]], dtype=np.int64)
outputs = session.run(["logits"], {{"token_ids": input_ids}})
logits = outputs[0]
next_token = np.argmax(logits[:, -1, :], axis=-1)
print("Sonraki Token:", next_token)
"""
        elif export_format == "torchscript":
            return f"""# TorchScript JIT Çıkarımı (Python & C++)
import torch

# Python ile yükleme
model = torch.jit.load("{file_name}")
model.eval()

with torch.no_grad():
    token_ids = torch.tensor([[1, 45, 128, 92]], dtype=torch.long)
    logits = model(token_ids)
    print("Logits Shape:", logits.shape)
"""
        elif export_format == "gguf":
            return f"""# Ollama Modelfile Oluşturma & Çalıştırma
# 1. 'Modelfile' adında bir dosya oluşturun:
FROM ./{file_name}
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# 2. Terminal komutları ile Ollama'ya kaydedin:
# ollama create {model_name} -f Modelfile
# ollama run {model_name} "Merhaba yapay zeka!"
"""
        return ""

    def export_onnx(
        self,
        model: nn.Module,
        config: GPTConfig,
        output_path: Path,
        dynamic_shapes: bool = True,
        opset: int = 17
    ) -> Dict[str, Any]:
        """
        Modeli ONNX formatına dönüştürür.
        """
        export_model = GPTInferenceWrapper(model)
        export_model.eval()
        dummy_input = torch.randint(0, config.vocab_size, (1, 16), dtype=torch.long)

        dynamic_axes = {
            "token_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        } if dynamic_shapes else None

        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            export_model,
            (dummy_input,),
            str(output_path),
            input_names=["token_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True
        )

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        sha256 = self._compute_sha256(output_path)

        return {
            "format": "onnx",
            "file_name": output_path.name,
            "path": str(output_path),
            "file_size_mb": round(file_size_mb, 2),
            "sha256": sha256,
            "opset_version": opset,
            "dynamic_shapes": dynamic_shapes
        }

    def export_torchscript(
        self,
        model: nn.Module,
        config: GPTConfig,
        output_path: Path,
        method: str = "trace"
    ) -> Dict[str, Any]:
        """
        Modeli TorchScript (.pt) formatına dönüştürür.
        """
        export_model = GPTInferenceWrapper(model)
        export_model.eval()
        dummy_input = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if method == "trace":
            traced = torch.jit.trace(export_model, dummy_input)
            traced_module = traced[0] if isinstance(traced, tuple) else traced
            torch.jit.save(traced_module, str(output_path))
        else:
            scripted_module = torch.jit.script(export_model)
            torch.jit.save(scripted_module, str(output_path))

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        sha256 = self._compute_sha256(output_path)

        return {
            "format": "torchscript",
            "file_name": output_path.name,
            "path": str(output_path),
            "file_size_mb": round(file_size_mb, 2),
            "sha256": sha256,
            "method": method
        }

    def export_gguf(
        self,
        model: nn.Module,
        config: GPTConfig,
        output_path: Path,
        quantization: str = "none",
        model_name: str = "gpt"
    ) -> Dict[str, Any]:
        """
        Modeli GGUF v3 formatına dönüştürür (Ollama / llama.cpp uyumlu).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = GGUFWriter(str(output_path), architecture="gpt2")

        # Architecture metadata
        writer.add_string("general.name", model_name)
        writer.add_uint32("gpt2.context_length", config.max_seq_len)
        writer.add_uint32("gpt2.embedding_length", config.d_model)
        writer.add_uint32("gpt2.block_count", config.n_layers)
        writer.add_uint32("gpt2.feed_forward_length", config.d_ff)
        writer.add_uint32("gpt2.attention.head_count", config.n_heads)

        # Tensor type selection
        if quantization == "int4":
            tensor_type = GGML_TYPE_Q4_0
        elif quantization == "int8":
            tensor_type = GGML_TYPE_Q8_0
        elif quantization == "fp16":
            tensor_type = GGML_TYPE_F16
        else:
            tensor_type = GGML_TYPE_F32

        # Serialize state_dict tensors
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            if not isinstance(param, torch.Tensor):
                continue
            if getattr(param, "is_quantized", False):
                param = param.dequantize()
            param_np = param.detach().cpu().float().numpy()
            writer.add_tensor(name, param_np, tensor_type=tensor_type)

        bytes_written = writer.write()
        file_size_mb = bytes_written / (1024 * 1024)
        sha256 = self._compute_sha256(output_path)

        return {
            "format": "gguf",
            "file_name": output_path.name,
            "path": str(output_path),
            "file_size_mb": round(file_size_mb, 2),
            "sha256": sha256,
            "tensor_count": len(state_dict),
            "quantization": quantization
        }

    def export_pipeline(
        self,
        model_name: str,
        version: Optional[str] = None,
        export_format: str = "onnx",
        quantization: str = "none"
    ) -> Dict[str, Any]:
        """
        Uçtan uca Model Export ve Kuantizasyon orkestrasyonu.
        Modeli yükler, kuantize eder, seçilen formata dönüştürür ve manifest dosyasına kaydeder.
        """
        start_time = time.time()
        export_format = export_format.lower()
        quantization = quantization.lower()

        # 1. Load model metadata & checkpoint from registry
        model_info = self.registry.load_model(model_name, version=version, load_weights=True)
        metadata = model_info["metadata"]
        resolved_version = metadata.get("version", "1.0.0")
        model_dir = Path(model_info["model_dir"])
        exports_dir = model_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        training_config = metadata.get("training_config", {})
        arch_config = training_config.get("model_config", {})

        # Reconstruct GPTConfig
        config = GPTConfig(
            vocab_size=arch_config.get("vocab_size", 8000),
            max_seq_len=arch_config.get("max_seq_len", 512),
            d_model=arch_config.get("d_model", 256),
            n_layers=arch_config.get("n_layers", 6),
            n_heads=arch_config.get("n_heads", 8),
            d_ff=arch_config.get("d_ff", 1024),
            dropout=0.0
        )

        # 2. Build model and load weights
        model = GPTModel(config)
        state_dict = model_info.get("state_dict")
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        model.eval()

        # 3. Apply Quantization if requested
        quant_metrics: Dict[str, Any] = {}
        if quantization == "fp16":
            model, quant_metrics = ModelQuantizer.quantize_fp16(model)
        elif quantization == "int8":
            model, quant_metrics = ModelQuantizer.quantize_int8_dynamic(model)
        elif quantization == "int4":
            model, quant_metrics = ModelQuantizer.quantize_int4_weight_only(model)

        # 4. Generate Export File
        suffix_map = {
            "onnx": ".onnx",
            "torchscript": ".pt",
            "gguf": ".gguf"
        }
        suffix = suffix_map.get(export_format, ".bin")
        quant_tag = f"_{quantization}" if quantization != "none" else ""
        out_filename = f"{model_name}_{resolved_version}{quant_tag}{suffix}"
        out_path = exports_dir / out_filename

        if export_format == "onnx":
            res = self.export_onnx(model, config, out_path)
        elif export_format == "torchscript":
            res = self.export_torchscript(model, config, out_path)
        elif export_format == "gguf":
            res = self.export_gguf(model, config, out_path, quantization=quantization, model_name=model_name)
        else:
            raise ValueError(f"Desteklenmeyen export formatı: {export_format}")

        export_duration = round(time.time() - start_time, 3)

        # Original size comparison
        orig_size_mb = metadata.get("file_size_mb") or res["file_size_mb"]
        compression_ratio = round(orig_size_mb / max(res["file_size_mb"], 0.01), 2)
        if compression_ratio < 1.0:
            compression_ratio = 1.0

        deployment_snippet = self.generate_deployment_snippet(
            export_format=export_format,
            file_name=out_filename,
            model_name=model_name
        )

        export_record = {
            "success": True,
            "model_name": model_name,
            "version": resolved_version,
            "export_format": export_format,
            "quantization": quantization,
            "file_name": out_filename,
            "file_path": str(out_path),
            "file_size_mb": res["file_size_mb"],
            "compression_ratio": compression_ratio,
            "sha256": res["sha256"],
            "quantization_metrics": quant_metrics,
            "export_duration_sec": export_duration,
            "download_url": f"/api/v1/models/{model_name}/download/{out_filename}",
            "deployment_snippet": deployment_snippet,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        # 5. Save/Update export manifest
        manifest_path = exports_dir / "manifest.json"
        existing_manifest: List[Dict[str, Any]] = []
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    existing_manifest = json.load(mf)
            except Exception:
                existing_manifest = []

        # Remove duplicate of same filename if exists
        existing_manifest = [m for m in existing_manifest if m.get("file_name") != out_filename]
        existing_manifest.append(export_record)

        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(existing_manifest, mf, indent=2, ensure_ascii=False)

        return export_record

    def list_exports(self, model_name: str, version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Model için kaydedilmiş tüm dışa aktarımları listeler.
        """
        try:
            model_info = self.registry.load_model(model_name, version=version)
            exports_dir = Path(model_info["model_dir"]) / "exports"
            manifest_path = exports_dir / "manifest.json"

            if not manifest_path.exists():
                return []

            with open(manifest_path, "r", encoding="utf-8") as mf:
                return json.load(mf)
        except Exception as e:
            logger.warning(f"Error listing exports for {model_name}: {e}")
            return []
