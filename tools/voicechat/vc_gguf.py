#!/usr/bin/env python3
"""
Minimal GGUF reader shared by the VoiceChat converters.

gguf.GGUFReader would also work, but the VoiceChat file is 6 GiB and every
converter needs only a slice of it; a plain seek/read keeps the peak RSS at one
tensor.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

GGML_TYPE_NAME = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 27: "I64"}

# elements per block, bytes per block
GGML_BLOCK = {
    "F32": (1, 4),
    "F16": (1, 2),
    "I64": (1, 8),
    "Q4_0": (32, 18),
    "Q8_0": (32, 34),
}

_SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
    5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8),
    12: ("<d", 8),
}


def dequant_q4_0(raw: bytes, n_elements: int) -> np.ndarray:
    """Q4_0: per block of 32, one f16 scale then 16 bytes of packed nibbles."""
    n_blocks = n_elements // 32
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)  # {n_blocks, 1}
    q = b[:, 2:]
    lo = (q & 0x0F).astype(np.int8) - 8
    hi = (q >> 4).astype(np.int8) - 8
    # llama.cpp stores element i in the low nibble and element i+16 in the high one
    out = np.concatenate([lo, hi], axis=1).astype(np.float32) * d
    return out.reshape(-1)


class GGUFSource:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "rb")

        if self._raw(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")

        self.version = self._u32()
        n_tensors = self._u64()
        n_kv = self._u64()

        self.kv: dict[str, Any] = {}
        for _ in range(n_kv):
            key = self._string()
            self.kv[key] = self._value(self._u32())

        self.tensors: dict[str, dict[str, Any]] = {}
        for _ in range(n_tensors):
            name = self._string()
            n_dims = self._u32()
            if n_dims > 4:
                raise SystemExit(f"{name}: ndim={n_dims}, header is desynced")
            dims = [self._u64() for _ in range(n_dims)]  # GGUF order: dims[0] is the row width
            ty = GGML_TYPE_NAME.get(self._u32())
            off = self._u64()

            n = 1
            for d in dims:
                n *= d

            self.tensors[name] = dict(name=name, dims=dims, ty=ty, off=off, elements=n)

        align = self.kv.get("general.alignment", 32)
        self.data_start = (self.f.tell() + align - 1) // align * align

    # -- primitives ----------------------------------------------------------

    def _raw(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"want {n} bytes, got {len(b)}")
        return b

    def _u32(self) -> int:
        return struct.unpack("<I", self._raw(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self._raw(8))[0]

    def _string(self) -> str:
        return self._raw(self._u64()).decode("utf-8", "replace")

    def _value(self, t: int) -> Any:
        if t in _SCALAR:
            fmt, n = _SCALAR[t]
            return struct.unpack(fmt, self._raw(n))[0]
        if t == 8:
            return self._string()
        if t == 9:
            et = self._u32()
            n = self._u64()
            if et == 8:
                return [self._string() for _ in range(n)]
            fmt, sz = _SCALAR[et]
            return list(struct.unpack("<" + fmt[1] * n, self._raw(sz * n)))
        raise ValueError(f"unknown metadata value type {t}")

    # -- tensor access -------------------------------------------------------

    def take(self, name: str) -> dict[str, Any]:
        t = self.tensors.get(name)
        if t is None:
            raise SystemExit(f"missing tensor in source: {name}")
        return t

    def nbytes(self, t: dict[str, Any]) -> int:
        if t["ty"] not in GGML_BLOCK:
            raise SystemExit(f"{t['name']}: unhandled dtype {t['ty']}")
        block, size = GGML_BLOCK[t["ty"]]
        return t["elements"] // block * size

    def raw(self, t: dict[str, Any]) -> bytes:
        self.f.seek(self.data_start + t["off"])
        return self._raw(self.nbytes(t))

    def f32(self, name: str) -> np.ndarray:
        """Read any supported tensor as F32 in numpy order (reverse of GGUF dims)."""
        t = self.take(name)
        if t["ty"] == "F16":
            a = np.frombuffer(self.raw(t), dtype=np.float16).astype(np.float32)
        elif t["ty"] == "F32":
            a = np.frombuffer(self.raw(t), dtype=np.float32)
        elif t["ty"] == "I64":
            a = np.frombuffer(self.raw(t), dtype=np.int64).astype(np.float32)
        elif t["ty"] == "Q4_0":
            a = dequant_q4_0(self.raw(t), t["elements"])
        else:
            raise SystemExit(f"{name}: cannot read {t['ty']} as float")
        return a.reshape(tuple(reversed(t["dims"])))

    def i64(self, name: str) -> np.ndarray:
        t = self.take(name)
        if t["ty"] != "I64":
            raise SystemExit(f"{name}: expected I64, got {t['ty']}")
        return np.frombuffer(self.raw(t), dtype=np.int64).reshape(tuple(reversed(t["dims"])))
