"""Pure-Python GGUF v1-v3 header reader for local files and HTTP Range URLs.

Only the header is read: key-value metadata and tensor descriptors. Tensor
weights are never downloaded. Field widths and type tables follow llama.cpp's
ggml sources (`ggml.h`, `ggml-common.h`, `gguf-py/gguf/constants.py`).
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Protocol, Self
from urllib.parse import urlsplit, urlunsplit

import httpx

from gguf_preflight.models import (
    FailureCode,
    Finding,
    GgufMetadata,
    Severity,
    TensorTypeStat,
)

MAGIC = b"GGUF"
MAX_KV = 1_000_000
MAX_TENSORS = 10_000_000
MAX_STRING_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ITEMS = 200_000_000
MAX_DIMS = 8
MAX_ARRAY_DEPTH = 4
MAX_HEADER_BYTES = 1024 * 1024 * 1024
INITIAL_WINDOW = 256 * 1024
MAX_WINDOW = 8 * 1024 * 1024
LOW_BIT_THRESHOLD = 4.0

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32 = range(7)
T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(7, 13)

_SCALARS: dict[int, tuple[str, int]] = {
    T_UINT8: ("B", 1),
    T_INT8: ("b", 1),
    T_UINT16: ("H", 2),
    T_INT16: ("h", 2),
    T_UINT32: ("I", 4),
    T_INT32: ("i", 4),
    T_FLOAT32: ("f", 4),
    T_BOOL: ("?", 1),
    T_UINT64: ("Q", 8),
    T_INT64: ("q", 8),
    T_FLOAT64: ("d", 8),
}

# ggml type id -> (name, elements per block, bytes per block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
    43: ("F8_E4M3", 1, 1),
}
_UNQUANTIZED = {"F32", "F16", "F64", "BF16", "I8", "I16", "I32", "I64", "F8_E4M3"}

FILE_TYPES: dict[int, str] = {
    0: "ALL_F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
    39: "NVFP4",
    40: "Q1_0",
    41: "Q2_0",
    42: "F8_E4M3",
}
FILE_TYPE_GUESSED = 1024


class GgufError(ValueError):
    """The input is not a readable GGUF header."""


class ByteSource(Protocol):
    def read(self, offset: int, size: int) -> bytes: ...

    def close(self) -> None: ...


class LocalSource:
    def __init__(self, path: str | Path) -> None:
        try:
            self._file: BinaryIO = open(path, "rb")
        except OSError as exc:
            raise GgufError(f"Cannot open {path}: {exc.strerror or exc}") from exc

    def read(self, offset: int, size: int) -> bytes:
        self._file.seek(offset)
        return self._file.read(size)

    def close(self) -> None:
        self._file.close()


def normalize_url(url: str) -> str:
    """Turn a Hugging Face `/blob/` page URL into its downloadable `/resolve/` URL."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise GgufError(f"Unsupported URL scheme: {parts.scheme or '(none)'}")
    path = parts.path
    if _is_hf_host(parts.hostname) and "/blob/" in path:
        path = path.replace("/blob/", "/resolve/", 1)
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def _is_hf_host(host: str | None) -> bool:
    return host in {"huggingface.co", "www.huggingface.co", "hf.co"}


class HttpRangeSource:
    """Reads byte ranges over HTTP and refuses servers that would send the whole file."""

    def __init__(self, url: str, client: httpx.Client | None = None) -> None:
        self.url = normalize_url(url)
        self._owns_client = client is None
        self._client = client or httpx.Client(follow_redirects=True, timeout=30.0)
        self._headers = {"Accept-Encoding": "identity"}
        token = os.environ.get("HF_TOKEN")
        # httpx drops Authorization on cross-origin redirects, so the token stays on HF.
        if token and _is_hf_host(urlsplit(self.url).hostname):
            self._headers["Authorization"] = f"Bearer {token}"

    def read(self, offset: int, size: int) -> bytes:
        headers = {**self._headers, "Range": f"bytes={offset}-{offset + size - 1}"}
        try:
            with self._client.stream("GET", self.url, headers=headers) as response:
                if response.status_code == 416:
                    return b""
                if response.status_code in {401, 403}:
                    raise GgufError(
                        f"HTTP {response.status_code}: access denied. For gated Hugging Face "
                        "repos, accept the license and set HF_TOKEN."
                    )
                if response.status_code == 200:
                    raise GgufError("Server ignored the Range header; refusing a full download.")
                if response.status_code != 206:
                    raise GgufError(f"HTTP {response.status_code} while reading {self.url}")
                _check_content_range(response.headers.get("content-range"), offset)
                chunks: list[bytes] = []
                received = 0
                for chunk in response.iter_raw():
                    chunks.append(chunk)
                    received += len(chunk)
                    if received >= size:
                        break
                return b"".join(chunks)[:size]
        except httpx.HTTPError as exc:
            raise GgufError(f"Network error reading {self.url}: {exc}") from exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def _check_content_range(value: str | None, offset: int) -> None:
    if not value:
        return
    try:
        start = int(value.split()[1].split("-")[0])
    except (IndexError, ValueError):
        return
    if start != offset:
        raise GgufError(f"Server returned bytes from {start}, expected {offset}.")


class _Reader:
    """Sequential reader with a growing window, so vocab-heavy headers stay cheap."""

    def __init__(self, source: ByteSource) -> None:
        self._source = source
        self._buffer = b""
        self._start = 0
        self._window = INITIAL_WINDOW
        self.pos = 0
        self.endian = "<"

    def take(self, size: int) -> bytes:
        if size < 0 or self.pos + size > MAX_HEADER_BYTES:
            raise GgufError("GGUF header exceeds the supported size.")
        offset = self.pos - self._start
        if offset < 0 or offset + size > len(self._buffer):
            want = max(size, self._window)
            self._buffer = self._source.read(self.pos, want)
            self._start = self.pos
            self._window = min(self._window * 2, MAX_WINDOW)
            offset = 0
            if len(self._buffer) < size:
                raise GgufError(f"Truncated GGUF header at byte {self.pos}.")
        self.pos += size
        return self._buffer[offset : offset + size]

    def skip(self, size: int) -> None:
        if size < 0 or self.pos + size > MAX_HEADER_BYTES:
            raise GgufError("GGUF header exceeds the supported size.")
        self.pos += size

    def scalar(self, fmt: str, size: int) -> Any:
        return struct.unpack(self.endian + fmt, self.take(size))[0]

    def u32(self) -> int:
        return int(self.scalar("I", 4))

    def u64(self) -> int:
        return int(self.scalar("Q", 8))


@dataclass(frozen=True)
class ArraySummary:
    """Arrays such as vocabularies are counted, not loaded."""

    element_type: int
    length: int


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def elements(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def type_name(self) -> str:
        known = GGML_TYPES.get(self.ggml_type)
        return known[0] if known else f"UNKNOWN_{self.ggml_type}"

    @property
    def byte_size(self) -> int | None:
        known = GGML_TYPES.get(self.ggml_type)
        if known is None:
            return None
        _, block, block_bytes = known
        return self.elements * block_bytes // block


@dataclass
class GgufHeader:
    version: int
    endianness: str
    header_bytes: int
    kv: dict[str, Any] = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)


def _read_count(reader: _Reader, version: int) -> int:
    return reader.u32() if version == 1 else reader.u64()


def _read_string(reader: _Reader, version: int) -> str:
    length = _read_count(reader, version)
    if length > MAX_STRING_BYTES:
        raise GgufError(f"String of {length} bytes exceeds the limit.")
    raw = reader.take(length)
    # GGUFv1 writers could include a trailing NUL in the stored length.
    if version == 1:
        raw = raw.rstrip(b"\x00")
    return raw.decode("utf-8", errors="replace")


def _skip_string(reader: _Reader, version: int) -> None:
    length = _read_count(reader, version)
    if length > MAX_STRING_BYTES:
        raise GgufError(f"String of {length} bytes exceeds the limit.")
    reader.skip(length)


def _skip_array(reader: _Reader, version: int, depth: int) -> ArraySummary:
    if depth > MAX_ARRAY_DEPTH:
        raise GgufError("GGUF array nesting is too deep.")
    element_type = reader.u32()
    length = _read_count(reader, version)
    if length > MAX_ARRAY_ITEMS:
        raise GgufError(f"Array of {length} items exceeds the limit.")
    if element_type in _SCALARS:
        reader.skip(_SCALARS[element_type][1] * length)
    elif element_type == T_STRING:
        for _ in range(length):
            _skip_string(reader, version)
    elif element_type == T_ARRAY:
        for _ in range(length):
            _skip_array(reader, version, depth + 1)
    else:
        raise GgufError(f"Unknown array element type {element_type}.")
    return ArraySummary(element_type, length)


def _read_value(reader: _Reader, value_type: int, version: int) -> Any:
    if value_type in _SCALARS:
        return reader.scalar(*_SCALARS[value_type])
    if value_type == T_STRING:
        return _read_string(reader, version)
    if value_type == T_ARRAY:
        return _skip_array(reader, version, 1)
    raise GgufError(f"Unknown value type {value_type}.")


def parse_header(source: ByteSource) -> GgufHeader:
    reader = _Reader(source)
    if reader.take(4) != MAGIC:
        raise GgufError("Not a GGUF file (bad magic).")
    raw_version = reader.take(4)
    version = struct.unpack("<I", raw_version)[0]
    endianness = "little"
    if version not in {1, 2, 3}:
        swapped = struct.unpack(">I", raw_version)[0]
        if swapped not in {1, 2, 3}:
            raise GgufError(f"Unsupported GGUF version {version}.")
        version, endianness, reader.endian = swapped, "big", ">"

    tensor_count = _read_count(reader, version)
    kv_count = _read_count(reader, version)
    if tensor_count > MAX_TENSORS or kv_count > MAX_KV:
        raise GgufError("Implausible tensor or key-value count in GGUF header.")

    kv: dict[str, Any] = {}
    for _ in range(kv_count):
        key = _read_string(reader, version)
        value_type = reader.u32()
        kv[key] = _read_value(reader, value_type, version)

    tensors: list[TensorInfo] = []
    for _ in range(tensor_count):
        name = _read_string(reader, version)
        n_dims = reader.u32()
        if n_dims > MAX_DIMS:
            raise GgufError(f"Tensor {name!r} has {n_dims} dimensions.")
        shape = tuple(_read_count(reader, version) for _ in range(n_dims))
        ggml_type = reader.u32()
        offset = reader.u64()
        tensors.append(TensorInfo(name, shape, ggml_type, offset))

    return GgufHeader(version, endianness, reader.pos, kv, tensors)


def tensor_fingerprint(tensors: list[TensorInfo]) -> str:
    """Order-independent SHA-256 over tensor names, shapes and types (not weights)."""
    digest = hashlib.sha256()
    for tensor in sorted(tensors, key=lambda item: item.name):
        shape = ",".join(str(dim) for dim in tensor.shape)
        digest.update(f"{tensor.name}\0{shape}\0{tensor.type_name}\n".encode())
    return digest.hexdigest()


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _type_stats(tensors: list[TensorInfo]) -> list[TensorTypeStat]:
    counts: dict[str, int] = defaultdict(int)
    sizes: dict[str, int] = defaultdict(int)
    for tensor in tensors:
        counts[tensor.type_name] += 1
        sizes[tensor.type_name] += tensor.byte_size or 0
    stats = [TensorTypeStat(ggml_type=k, tensors=counts[k], bytes=sizes[k]) for k in counts]
    return sorted(stats, key=lambda stat: (-stat.bytes, stat.ggml_type))


def _dominant_quant_bits(stats: list[TensorTypeStat]) -> tuple[str, float] | None:
    by_name = {info[0]: info for info in GGML_TYPES.values()}
    for stat in stats:
        if stat.ggml_type in _UNQUANTIZED or stat.ggml_type not in by_name:
            continue
        _, block, block_bytes = by_name[stat.ggml_type]
        return stat.ggml_type, block_bytes * 8 / block
    return None


def _metadata_findings(meta: GgufMetadata, low_bit: tuple[str, float] | None) -> list[Finding]:
    findings: list[Finding] = []
    defect = FailureCode.METADATA_DEFECT
    if meta.tokenizer_model is None:
        findings.append(
            Finding(
                check="missing_tokenizer",
                severity=Severity.FAIL,
                code=defect,
                message="The file has no tokenizer.ggml.model key.",
            )
        )
    elif meta.tokenizer_model == "gpt2" and not meta.tokenizer_pre:
        # Only BPE ("gpt2") vocabularies use tokenizer.ggml.pre.
        findings.append(
            Finding(
                check="missing_pre_tokenizer",
                severity=Severity.FAIL,
                code=defect,
                message="BPE tokenizer has no tokenizer.ggml.pre, so text may tokenize wrongly.",
                evidence={"tokenizer_model": meta.tokenizer_model},
            )
        )
    if not meta.has_chat_template:
        findings.append(
            Finding(
                check="missing_chat_template",
                severity=Severity.FAIL,
                code=defect,
                message="No tokenizer.chat_template, so tool calls and roles may be misformatted.",
            )
        )
    if low_bit and low_bit[1] < LOW_BIT_THRESHOLD and not meta.imatrix_keys:
        findings.append(
            Finding(
                check="uncalibrated_low_bit_quant",
                severity=Severity.WARN,
                code=defect,
                message=(
                    f"Dominant tensor type {low_bit[0]} ({low_bit[1]:.2f} bits) has no "
                    "quantize.imatrix.* metadata. Low-bit builds without an importance "
                    "matrix often degrade."
                ),
                evidence={"dominant_type": low_bit[0], "bits": round(low_bit[1], 4)},
            )
        )
    if meta.split_count and meta.split_count > 1:
        findings.append(
            Finding(
                check="split_gguf",
                severity=Severity.INFO,
                message=(
                    f"This is one of {meta.split_count} shards. Tensor stats and the "
                    "fingerprint cover this shard only."
                ),
            )
        )
    unknown = [stat.ggml_type for stat in meta.tensor_types if stat.ggml_type.startswith("UNKNOWN")]
    if unknown:
        findings.append(
            Finding(
                check="unknown_tensor_type",
                severity=Severity.WARN,
                code=defect,
                message="Some tensor types are unknown to this version of gguf-preflight.",
                evidence={"types": unknown},
            )
        )
    return findings


def analyze(header: GgufHeader, source: str) -> GgufMetadata:
    kv = header.kv
    architecture = _as_str(kv.get("general.architecture"))
    template = _as_str(kv.get("tokenizer.chat_template"))
    named_template = any(key.startswith("tokenizer.chat_template.") for key in kv)
    file_type = _as_int(kv.get("general.file_type"))
    stats = _type_stats(header.tensors)
    dominant = _dominant_quant_bits(stats)

    quantization: str | None = None
    guessed = False
    if file_type is not None:
        guessed = bool(file_type & FILE_TYPE_GUESSED)
        base = file_type & ~FILE_TYPE_GUESSED
        quantization = FILE_TYPES.get(base, f"UNKNOWN_{base}")
    elif dominant:
        quantization, guessed = dominant[0], True
    elif stats:
        quantization, guessed = stats[0].ggml_type, True

    known = [tensor for tensor in header.tensors if tensor.byte_size is not None]
    parameters = sum(tensor.elements for tensor in header.tensors)
    known_elements = sum(tensor.elements for tensor in known)
    data_bytes = sum(tensor.byte_size or 0 for tensor in known)
    bits = round(data_bytes * 8 / known_elements, 4) if known_elements else None

    meta = GgufMetadata(
        source=source,
        gguf_version=header.version,
        endianness=header.endianness,
        header_bytes=header.header_bytes,
        kv_count=len(kv),
        tensor_count=len(header.tensors),
        architecture=architecture,
        model_name=_as_str(kv.get("general.name")),
        block_count=_as_int(kv.get(f"{architecture}.block_count")) if architecture else None,
        parameter_count=parameters,
        tensor_data_bytes=data_bytes,
        effective_bits_per_weight=bits,
        quantization=quantization,
        quantization_guessed=guessed,
        tensor_types=stats,
        tokenizer_model=_as_str(kv.get("tokenizer.ggml.model")),
        tokenizer_pre=_as_str(kv.get("tokenizer.ggml.pre")),
        has_chat_template=bool(template) or named_template,
        chat_template_sha256=hashlib.sha256(template.encode()).hexdigest() if template else None,
        imatrix_keys=sorted(key for key in kv if key.startswith("quantize.imatrix.")),
        split_count=_as_int(kv.get("split.count")),
        tensor_sha256=tensor_fingerprint(header.tensors),
    )
    meta.findings = _metadata_findings(meta, dominant)
    return meta


class _OpenedSource:
    def __init__(self, ref: str | Path, client: httpx.Client | None) -> None:
        text = str(ref)
        self.source: ByteSource
        if text.lower().startswith(("http://", "https://")):
            self.source = HttpRangeSource(text, client)
        else:
            self.source = LocalSource(text)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.source.close()


def read_header(ref: str | Path, client: httpx.Client | None = None) -> GgufHeader:
    with _OpenedSource(ref, client) as opened:
        return parse_header(opened.source)


def inspect_gguf(ref: str | Path, client: httpx.Client | None = None) -> GgufMetadata:
    """Read a local path or HTTP(S) URL and return analyzed GGUF metadata."""
    return analyze(read_header(ref, client), _safe_source_label(str(ref)))


def _safe_source_label(ref: str) -> str:
    """Keep receipts shareable: no local directories, credentials or signed queries."""
    if not ref.lower().startswith(("http://", "https://")):
        return Path(ref).name
    parts = urlsplit(ref)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))
