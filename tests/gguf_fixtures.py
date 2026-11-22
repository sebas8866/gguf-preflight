"""Builds minimal synthetic GGUF binaries for tests."""

from __future__ import annotations

import struct
from typing import Any

UINT32, FLOAT32, BOOL, STRING, ARRAY, UINT64 = 4, 6, 7, 8, 9, 10


def _count(value: int, version: int, endian: str) -> bytes:
    return struct.pack(endian + ("I" if version == 1 else "Q"), value)


def _string(text: str, version: int, endian: str) -> bytes:
    raw = text.encode()
    return _count(len(raw), version, endian) + raw


def _value(value: Any, version: int, endian: str) -> bytes:
    if isinstance(value, bool):
        return struct.pack(endian + "I?", BOOL, value)
    if isinstance(value, int):
        return struct.pack(endian + "II", UINT32, value)
    if isinstance(value, float):
        return struct.pack(endian + "If", FLOAT32, value)
    if isinstance(value, str):
        return struct.pack(endian + "I", STRING) + _string(value, version, endian)
    if isinstance(value, list):
        head = struct.pack(endian + "II", ARRAY, STRING) + _count(len(value), version, endian)
        return head + b"".join(_string(item, version, endian) for item in value)
    raise TypeError(type(value))


def build_gguf(
    kv: dict[str, Any],
    tensors: list[tuple[str, tuple[int, ...], int]],
    version: int = 3,
    endian: str = "<",
    trailing: bytes = b"",
) -> bytes:
    out = bytearray(b"GGUF")
    out += struct.pack(endian + "I", version)
    out += _count(len(tensors), version, endian)
    out += _count(len(kv), version, endian)
    for key, value in kv.items():
        out += _string(key, version, endian)
        out += _value(value, version, endian)
    offset = 0
    for name, shape, ggml_type in tensors:
        out += _string(name, version, endian)
        out += struct.pack(endian + "I", len(shape))
        for dim in shape:
            out += _count(dim, version, endian)
        out += struct.pack(endian + "IQ", ggml_type, offset)
        offset += 64
    return bytes(out) + trailing


HEALTHY_KV: dict[str, Any] = {
    "general.architecture": "llama",
    "general.name": "Tiny Test",
    "general.file_type": 15,
    "llama.block_count": 2,
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "llama-bpe",
    "tokenizer.ggml.tokens": ["<s>", "</s>", "a", "b"],
    "tokenizer.chat_template": "{{ messages }}",
}
HEALTHY_TENSORS: list[tuple[str, tuple[int, ...], int]] = [
    ("layers.0.weight", (256, 4), 12),
    ("layers.1.weight", (256, 4), 12),
    ("norm.weight", (256,), 0),
]
