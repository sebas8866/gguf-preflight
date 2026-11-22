from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
import respx
from gguf_fixtures import HEALTHY_KV, HEALTHY_TENSORS, build_gguf

from gguf_preflight.inspector.gguf_header import (
    GGML_TYPES,
    GgufError,
    inspect_gguf,
    normalize_url,
    read_header,
)
from gguf_preflight.models import FailureCode, Severity


def write(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(data)
    return path


def checks(meta) -> set[str]:  # type: ignore[no-untyped-def]
    return {finding.check for finding in meta.findings}


@pytest.mark.parametrize("version", [1, 2, 3])
def test_reads_all_versions(tmp_path: Path, version: int) -> None:
    meta = inspect_gguf(write(tmp_path, build_gguf(HEALTHY_KV, HEALTHY_TENSORS, version=version)))
    assert meta.gguf_version == version
    assert meta.architecture == "llama"
    assert meta.block_count == 2
    assert meta.tensor_count == 3
    assert meta.parameter_count == 256 * 4 * 2 + 256
    assert meta.quantization == "Q4_K_M"
    assert meta.tokenizer_pre == "llama-bpe"
    assert meta.has_chat_template
    assert meta.findings == []


def test_big_endian_v3(tmp_path: Path) -> None:
    meta = inspect_gguf(write(tmp_path, build_gguf(HEALTHY_KV, HEALTHY_TENSORS, endian=">")))
    assert meta.endianness == "big"
    assert meta.parameter_count == 2304


def test_effective_bits_per_weight(tmp_path: Path) -> None:
    meta = inspect_gguf(write(tmp_path, build_gguf(HEALTHY_KV, [("w", (256, 8), 12)])))
    assert meta.effective_bits_per_weight == 4.5
    assert meta.tensor_data_bytes == 144 * 8


@pytest.mark.parametrize(
    ("name", "bits"),
    [("Q2_K", 2.625), ("Q3_K", 3.4375), ("Q5_K", 5.5), ("Q6_K", 6.5625), ("IQ4_XS", 4.25)],
)
def test_type_table_matches_ggml_block_sizes(name: str, bits: float) -> None:
    entry = next(item for item in GGML_TYPES.values() if item[0] == name)
    assert entry[2] * 8 / entry[1] == bits


def test_missing_pre_tokenizer_flagged_for_bpe(tmp_path: Path) -> None:
    kv = {k: v for k, v in HEALTHY_KV.items() if k != "tokenizer.ggml.pre"}
    meta = inspect_gguf(write(tmp_path, build_gguf(kv, HEALTHY_TENSORS)))
    finding = next(f for f in meta.findings if f.check == "missing_pre_tokenizer")
    assert finding.code is FailureCode.METADATA_DEFECT
    assert finding.severity is Severity.FAIL


def test_pre_tokenizer_not_required_for_spm(tmp_path: Path) -> None:
    kv = {k: v for k, v in HEALTHY_KV.items() if k != "tokenizer.ggml.pre"}
    kv["tokenizer.ggml.model"] = "llama"
    meta = inspect_gguf(write(tmp_path, build_gguf(kv, HEALTHY_TENSORS)))
    assert "missing_pre_tokenizer" not in checks(meta)


def test_missing_chat_template_flagged(tmp_path: Path) -> None:
    kv = {k: v for k, v in HEALTHY_KV.items() if k != "tokenizer.chat_template"}
    meta = inspect_gguf(write(tmp_path, build_gguf(kv, HEALTHY_TENSORS)))
    assert "missing_chat_template" in checks(meta)


def test_named_chat_template_counts(tmp_path: Path) -> None:
    kv = {k: v for k, v in HEALTHY_KV.items() if k != "tokenizer.chat_template"}
    kv["tokenizer.chat_template.tool_use"] = "{{ tools }}"
    meta = inspect_gguf(write(tmp_path, build_gguf(kv, HEALTHY_TENSORS)))
    assert meta.has_chat_template


@pytest.mark.parametrize("ggml_type", [10, 11, 16, 18])
def test_uncalibrated_low_bit_flagged(tmp_path: Path, ggml_type: int) -> None:
    kv = {**HEALTHY_KV, "general.file_type": 10}
    meta = inspect_gguf(write(tmp_path, build_gguf(kv, [("w", (256, 4), ggml_type)])))
    assert "uncalibrated_low_bit_quant" in checks(meta)


def test_imatrix_clears_low_bit_flag(tmp_path: Path) -> None:
    kv = {**HEALTHY_KV, "general.file_type": 10, "quantize.imatrix.file": "m.dat"}
    meta = inspect_gguf(write(tmp_path, build_gguf(kv, [("w", (256, 4), 10)])))
    assert "uncalibrated_low_bit_quant" not in checks(meta)
    assert meta.imatrix_keys == ["quantize.imatrix.file"]


def test_q4_without_imatrix_not_flagged(tmp_path: Path) -> None:
    meta = inspect_gguf(write(tmp_path, build_gguf(HEALTHY_KV, HEALTHY_TENSORS)))
    assert "uncalibrated_low_bit_quant" not in checks(meta)


def test_fingerprint_is_order_independent(tmp_path: Path) -> None:
    a = inspect_gguf(write(tmp_path, build_gguf(HEALTHY_KV, HEALTHY_TENSORS)))
    b = inspect_gguf(write(tmp_path, build_gguf(HEALTHY_KV, list(reversed(HEALTHY_TENSORS)))))
    assert a.tensor_sha256 == b.tensor_sha256


@pytest.mark.parametrize(
    "changed",
    [
        [("layers.0.renamed", (256, 4), 12), *HEALTHY_TENSORS[1:]],
        [("layers.0.weight", (256, 8), 12), *HEALTHY_TENSORS[1:]],
        [("layers.0.weight", (256, 4), 11), *HEALTHY_TENSORS[1:]],
    ],
)
def test_fingerprint_changes_with_layout(tmp_path: Path, changed: list) -> None:  # type: ignore[type-arg]
    a = inspect_gguf(write(tmp_path, build_gguf(HEALTHY_KV, HEALTHY_TENSORS)))
    b = inspect_gguf(write(tmp_path, build_gguf(HEALTHY_KV, changed)))
    assert a.tensor_sha256 != b.tensor_sha256


def test_guessed_file_type_flag(tmp_path: Path) -> None:
    kv = {**HEALTHY_KV, "general.file_type": 15 | 1024}
    meta = inspect_gguf(write(tmp_path, build_gguf(kv, HEALTHY_TENSORS)))
    assert meta.quantization == "Q4_K_M"
    assert meta.quantization_guessed


def test_split_shard_reported(tmp_path: Path) -> None:
    kv = {**HEALTHY_KV, "split.count": 3}
    meta = inspect_gguf(write(tmp_path, build_gguf(kv, HEALTHY_TENSORS)))
    assert meta.split_count == 3
    assert "split_gguf" in checks(meta)


@pytest.mark.parametrize(
    ("data", "pattern"),
    [
        (b"NOPE" + b"\0" * 32, "bad magic"),
        (b"GGUF" + (9).to_bytes(4, "little") + b"\0" * 16, "Unsupported"),
        (b"GGUF", "Truncated"),
    ],
)
def test_rejects_invalid(tmp_path: Path, data: bytes, pattern: str) -> None:
    with pytest.raises(GgufError, match=pattern):
        read_header(write(tmp_path, data))


def test_rejects_truncated_body(tmp_path: Path) -> None:
    data = build_gguf(HEALTHY_KV, HEALTHY_TENSORS)
    with pytest.raises(GgufError, match="Truncated"):
        read_header(write(tmp_path, data[:-5]))


def test_rejects_absurd_counts(tmp_path: Path) -> None:
    data = b"GGUF" + (3).to_bytes(4, "little") + (2**62).to_bytes(8, "little") + bytes(8)
    with pytest.raises(GgufError, match="Implausible"):
        read_header(write(tmp_path, data))


def test_missing_file_is_clean_error(tmp_path: Path) -> None:
    with pytest.raises(GgufError, match="Cannot open"):
        read_header(tmp_path / "absent.gguf")


def _range_handler(data: bytes, seen: list[str]):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        header = request.headers["range"]
        seen.append(header)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", header)
        assert match
        start, end = int(match[1]), int(match[2])
        if start >= len(data):
            return httpx.Response(416)
        body = data[start : end + 1]
        content_range = f"bytes {start}-{start + len(body) - 1}/{len(data)}"
        return httpx.Response(206, content=body, headers={"content-range": content_range})

    return handler


@respx.mock
def test_remote_reads_only_header_ranges() -> None:
    weights = b"\xaa" * (32 * 1024 * 1024)
    data = build_gguf(HEALTHY_KV, HEALTHY_TENSORS, trailing=weights)
    seen: list[str] = []
    url = "https://huggingface.co/org/repo/resolve/main/m.gguf"
    respx.get(url).mock(side_effect=_range_handler(data, seen))
    meta = inspect_gguf(url)
    assert meta.architecture == "llama"
    assert len(seen) == 1
    assert seen[0] == f"bytes=0-{256 * 1024 - 1}"


@respx.mock
def test_remote_large_vocab_grows_window() -> None:
    vocab = [f"token-{index:07d}" for index in range(120_000)]
    kv = {**HEALTHY_KV, "tokenizer.ggml.tokens": vocab}
    data = build_gguf(kv, HEALTHY_TENSORS)
    assert len(data) > 2 * 1024 * 1024
    seen: list[str] = []
    url = "https://example.test/large.gguf"
    respx.get(url).mock(side_effect=_range_handler(data, seen))
    meta = inspect_gguf(url)
    assert meta.tensor_count == 3
    assert 2 <= len(seen) <= 6


@respx.mock
def test_remote_rejects_server_without_range() -> None:
    url = "https://example.test/full.gguf"
    respx.get(url).mock(return_value=httpx.Response(200, content=b"GGUF"))
    with pytest.raises(GgufError, match="Range"):
        inspect_gguf(url)


@respx.mock
def test_remote_gated_message() -> None:
    url = "https://huggingface.co/org/gated/resolve/main/m.gguf"
    respx.get(url).mock(return_value=httpx.Response(401))
    with pytest.raises(GgufError, match="HF_TOKEN"):
        inspect_gguf(url)


@respx.mock
def test_hf_token_sent_only_to_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    data = build_gguf(HEALTHY_KV, HEALTHY_TENSORS)
    hf = respx.get("https://huggingface.co/o/r/resolve/main/m.gguf").mock(
        side_effect=_range_handler(data, [])
    )
    other = respx.get("https://example.test/m.gguf").mock(side_effect=_range_handler(data, []))
    inspect_gguf("https://huggingface.co/o/r/resolve/main/m.gguf")
    inspect_gguf("https://example.test/m.gguf")
    assert hf.calls[0].request.headers["authorization"] == "Bearer test-token"
    assert "authorization" not in other.calls[0].request.headers


@respx.mock
def test_source_label_strips_signed_query() -> None:
    data = build_gguf(HEALTHY_KV, HEALTHY_TENSORS)
    url = "https://user:secret@example.test/m.gguf?sig=private"
    respx.get(url__startswith="https://").mock(side_effect=_range_handler(data, []))
    meta = inspect_gguf(url)
    assert meta.source == "https://example.test/m.gguf"


def test_blob_url_normalized() -> None:
    url = "https://huggingface.co/org/repo/blob/main/model.gguf"
    assert normalize_url(url) == "https://huggingface.co/org/repo/resolve/main/model.gguf"


def test_rejects_non_http_scheme() -> None:
    with pytest.raises(GgufError, match="scheme"):
        normalize_url("file:///etc/passwd")
