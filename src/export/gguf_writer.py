"""
src/export/gguf_writer.py

Pure-Python GGUF (v3) Binary Format Serializer:
Produces valid GGUF files for llama.cpp, Ollama, and local GGUF runtimes.

Spec reference:
https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

Author: Kenan AY
Location: Kütahya, TÜRKİYE
"""

import struct
import io
import os
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)

# GGUF Magic & Version Constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# GGUF Metadata Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Tensor Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 7


class GGUFWriter:
    """
    Pure-Python GGUF (v3) file generator.
    Creates valid GGUF files with architecture metadata, tokenizer info, and quantized tensors.
    """

    def __init__(self, path: str, architecture: str = "gpt2"):
        self.path = path
        self.architecture = architecture
        self.metadata: List[Tuple[str, int, Any]] = []
        self.tensors: List[Dict[str, Any]] = []
        self.alignment = DEFAULT_ALIGNMENT

        # Set default architecture metadata
        self.add_string("general.architecture", architecture)
        self.add_uint32("general.alignment", self.alignment)

    # -------------------------------------------------------------------------
    # Metadata Adders
    # -------------------------------------------------------------------------
    def add_string(self, key: str, value: str):
        self.metadata.append((key, GGUF_TYPE_STRING, value))

    def add_uint32(self, key: str, value: int):
        self.metadata.append((key, GGUF_TYPE_UINT32, int(value)))

    def add_int32(self, key: str, value: int):
        self.metadata.append((key, GGUF_TYPE_INT32, int(value)))

    def add_float32(self, key: str, value: float):
        self.metadata.append((key, GGUF_TYPE_FLOAT32, float(value)))

    def add_bool(self, key: str, value: bool):
        self.metadata.append((key, GGUF_TYPE_BOOL, bool(value)))

    def add_array(self, key: str, element_type_or_values: Any, values: Optional[List[Any]] = None):
        if values is None:
            # Called as add_array(key, values)
            vals = element_type_or_values
            if vals and isinstance(vals[0], str):
                elem_type = GGUF_TYPE_STRING
            elif vals and isinstance(vals[0], float):
                elem_type = GGUF_TYPE_FLOAT32
            elif vals and isinstance(vals[0], int):
                elem_type = GGUF_TYPE_INT32
            else:
                elem_type = GGUF_TYPE_STRING
            self.metadata.append((key, GGUF_TYPE_ARRAY, (elem_type, vals)))
        else:
            self.metadata.append((key, GGUF_TYPE_ARRAY, (element_type_or_values, values)))

    def add_string_array(self, key: str, values: List[str]):
        self.add_array(key, GGUF_TYPE_STRING, values)

    # -------------------------------------------------------------------------
    # Tensor Registration
    # -------------------------------------------------------------------------
    def add_tensor(
        self,
        name: str,
        tensor_data: np.ndarray,
        tensor_type: int = GGML_TYPE_F32
    ):
        """
        GGUF tensörü ekler.
        tensor_data: numpy array
        """
        # GGUF expects dimensions from innermost to outermost
        shape = list(tensor_data.shape)[::-1]

        if tensor_type == GGML_TYPE_F16:
            raw_data = tensor_data.astype(np.float16).tobytes()
        elif tensor_type == GGML_TYPE_F32:
            raw_data = tensor_data.astype(np.float32).tobytes()
        elif tensor_type == GGML_TYPE_Q8_0:
            raw_data = self._quantize_q8_0(tensor_data.astype(np.float32))
        elif tensor_type == GGML_TYPE_Q4_0:
            raw_data = self._quantize_q4_0(tensor_data.astype(np.float32))
        else:
            raw_data = tensor_data.astype(np.float32).tobytes()
            tensor_type = GGML_TYPE_F32

        self.tensors.append({
            "name": name,
            "shape": shape,
            "tensor_type": tensor_type,
            "raw_data": raw_data,
            "size": len(raw_data)
        })

    # -------------------------------------------------------------------------
    # Quantization Encoders (Q8_0 and Q4_0)
    # -------------------------------------------------------------------------
    @staticmethod
    def _quantize_q8_0(data: np.ndarray) -> bytes:
        """
        Q8_0 blok kuantizasyonu (her blok 32 eleman: float16 scale + 32 signed int8).
        """
        flat = data.flatten()
        pad_size = (32 - (len(flat) % 32)) % 32
        if pad_size > 0:
            flat = np.pad(flat, (0, pad_size), mode='edge')

        blocks = flat.reshape(-1, 32)
        out = io.BytesIO()

        for block in blocks:
            max_val = np.max(np.abs(block))
            scale = max_val / 127.0 if max_val > 0 else 1.0
            # Scale as float16
            out.write(struct.pack("<e", np.float16(scale)))
            # 32 int8 values
            q_vals = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
            out.write(q_vals.tobytes())

        return out.getvalue()

    @staticmethod
    def _quantize_q4_0(data: np.ndarray) -> bytes:
        """
        Q4_0 blok kuantizasyonu (her blok 32 eleman: float16 scale + 16 bytes packed 4-bit nibbles).
        """
        flat = data.flatten()
        pad_size = (32 - (len(flat) % 32)) % 32
        if pad_size > 0:
            flat = np.pad(flat, (0, pad_size), mode='edge')

        blocks = flat.reshape(-1, 32)
        out = io.BytesIO()

        for block in blocks:
            max_val = np.max(np.abs(block))
            scale = max_val / -8.0 if max_val > 0 else 1.0  # Q4_0 symmetric
            # Scale as float16
            out.write(struct.pack("<e", np.float16(scale)))

            q_vals = np.clip(np.round(block / scale) + 8, 0, 15).astype(np.uint8)
            # Pack pairs of 4-bit values into bytes
            low = q_vals[0:16]
            high = q_vals[16:32]
            packed = (low | (high << 4)).astype(np.uint8)
            out.write(packed.tobytes())

        return out.getvalue()

    # -------------------------------------------------------------------------
    # Binary Writer
    # -------------------------------------------------------------------------
    def write(self) -> int:
        """
        Tüm GGUF v3 dosyasını diske yazar.
        Döndürür: Toplam yazılan bayt sayısı.
        """
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

        with open(self.path, "wb") as f:
            # 1. Header: Magic, Version, Tensor Count, Metadata KV Count
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # 2. Metadata KV Section
            for key, val_type, value in self.metadata:
                self._write_string(f, key)
                f.write(struct.pack("<I", val_type))
                self._write_value(f, val_type, value)

            # 3. Calculate tensor offsets with alignment
            # First pass: write tensor infos and record positions
            tensor_info_pos = f.tell()
            # Calculate tensor info size to determine data start offset
            current_offset = 0
            for t in self.tensors:
                t["offset"] = current_offset
                current_offset += t["size"]
                # Align tensor data buffer
                padding = (self.alignment - (current_offset % self.alignment)) % self.alignment
                current_offset += padding

            # Write Tensor Info entries
            for t in self.tensors:
                self._write_string(f, t["name"])
                # Number of dimensions
                f.write(struct.pack("<I", len(t["shape"])))
                # Dimensions
                for dim in t["shape"]:
                    f.write(struct.pack("<Q", int(dim)))
                # Type
                f.write(struct.pack("<I", t["tensor_type"]))
                # Offset in tensor data buffer
                f.write(struct.pack("<Q", t["offset"]))

            # 4. Alignment padding before tensor data buffer
            cur_pos = f.tell()
            pad = (self.alignment - (cur_pos % self.alignment)) % self.alignment
            if pad > 0:
                f.write(b"\x00" * pad)

            # 5. Tensor Data Buffer
            for t in self.tensors:
                f.write(t["raw_data"])
                # Align after each tensor
                cur_len = len(t["raw_data"])
                tensor_pad = (self.alignment - (cur_len % self.alignment)) % self.alignment
                if tensor_pad > 0:
                    f.write(b"\x00" * tensor_pad)

            total_bytes = f.tell()

        logger.info(f"GGUF v3 file written successfully: {self.path} ({total_bytes / (1024*1024):.2f} MB)")
        return total_bytes

    def _write_string(self, f, s: str):
        utf8_bytes = s.encode("utf-8")
        f.write(struct.pack("<Q", len(utf8_bytes)))
        f.write(utf8_bytes)

    def _write_value(self, f, val_type: int, value: Any):
        if val_type == GGUF_TYPE_UINT8:
            f.write(struct.pack("<B", int(value)))
        elif val_type == GGUF_TYPE_INT8:
            f.write(struct.pack("<b", int(value)))
        elif val_type == GGUF_TYPE_UINT16:
            f.write(struct.pack("<H", int(value)))
        elif val_type == GGUF_TYPE_INT16:
            f.write(struct.pack("<h", int(value)))
        elif val_type == GGUF_TYPE_UINT32:
            f.write(struct.pack("<I", int(value)))
        elif val_type == GGUF_TYPE_INT32:
            f.write(struct.pack("<i", int(value)))
        elif val_type == GGUF_TYPE_FLOAT32:
            f.write(struct.pack("<f", float(value)))
        elif val_type == GGUF_TYPE_BOOL:
            f.write(struct.pack("<?", bool(value)))
        elif val_type == GGUF_TYPE_STRING:
            self._write_string(f, str(value))
        elif val_type == GGUF_TYPE_UINT64:
            f.write(struct.pack("<Q", int(value)))
        elif val_type == GGUF_TYPE_INT64:
            f.write(struct.pack("<q", int(value)))
        elif val_type == GGUF_TYPE_FLOAT64:
            f.write(struct.pack("<d", float(value)))
        elif val_type == GGUF_TYPE_ARRAY:
            elem_type, items = value
            f.write(struct.pack("<I", elem_type))
            f.write(struct.pack("<Q", len(items)))
            for item in items:
                self._write_value(f, elem_type, item)
