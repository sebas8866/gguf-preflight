"""Pydantic v2 schemas shared by every probe, the CLI and the MCP server."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class FailureCode(StrEnum):
    METADATA_DEFECT = "METADATA_DEFECT"
    WIRE_BUG = "WIRE_BUG"
    PREFIX_CACHE_MISS = "PREFIX_CACHE_MISS"
    QUANT_COLLAPSE = "QUANT_COLLAPSE"
    LATENCY_BOTTLENECK = "LATENCY_BOTTLENECK"


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    FAIL = "fail"


class ProbeStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class Verdict(StrEnum):
    GO = "GO"
    CAUTION = "CAUTION"
    NO_GO = "NO_GO"
    INCOMPLETE = "INCOMPLETE"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Finding(StrictModel):
    """One observed problem. `check` is a stable id used by remediation rules."""

    check: str
    severity: Severity
    message: str
    code: FailureCode | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class TensorTypeStat(StrictModel):
    ggml_type: str
    tensors: int
    bytes: int


class GgufMetadata(StrictModel):
    source: str
    gguf_version: int
    endianness: str
    header_bytes: int
    kv_count: int
    tensor_count: int
    architecture: str | None = None
    model_name: str | None = None
    block_count: int | None = None
    parameter_count: int
    tensor_data_bytes: int
    effective_bits_per_weight: float | None = None
    quantization: str | None = None
    quantization_guessed: bool = False
    tensor_types: list[TensorTypeStat] = Field(default_factory=list)
    tokenizer_model: str | None = None
    tokenizer_pre: str | None = None
    has_chat_template: bool = False
    chat_template_sha256: str | None = None
    imatrix_keys: list[str] = Field(default_factory=list)
    split_count: int | None = None
    tensor_sha256: str = Field(
        description=(
            "Layout fingerprint over tensor names, shapes and types. It does not hash "
            "weight bytes, so fine-tunes with the same layout share this value."
        )
    )
    findings: list[Finding] = Field(default_factory=list)


class RuntimeInfo(StrictModel):
    adapter: str
    endpoint: str
    model: str
    version: str | None = None
    context_length: int | None = None
    openai_api: bool | None = None
    anthropic_api: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class GpuInfo(StrictModel):
    name: str
    vram_total_mb: int | None = None
    vram_used_mb: int | None = None


class HardwareInfo(StrictModel):
    os: str
    machine: str
    cpu: str | None = None
    logical_cpus: int | None = None
    ram_total_gb: float | None = None
    gpus: list[GpuInfo] = Field(default_factory=list)
    source: str = "builtin"
    llmfit: dict[str, Any] | None = None


class ProbeResult(StrictModel):
    name: str
    status: ProbeStatus
    duration_s: float = 0.0
    summary: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)


class RecommendedFix(StrictModel):
    check: str
    failure_code: FailureCode | None = None
    adapter: str
    title: str
    why: str
    commands: list[str] = Field(default_factory=list)
    source: str | None = None


class PreflightReceipt(StrictModel):
    schema_version: str = SCHEMA_VERSION
    tool_version: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    target_agent: str = "generic"
    verdict: Verdict
    duration_s: float = 0.0
    tensor_sha256: str | None = None
    gguf_metadata: GgufMetadata | None = None
    runtime_info: RuntimeInfo
    hardware_info: HardwareInfo
    probe_results: list[ProbeResult] = Field(default_factory=list)
    failure_codes: list[FailureCode] = Field(default_factory=list)
    recommended_fixes: list[RecommendedFix] = Field(default_factory=list)
    transcript_sha256: str | None = None
