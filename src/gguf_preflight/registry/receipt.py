"""Receipt verdicts, persistence, and comparison."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from gguf_preflight.models import (
    SCHEMA_VERSION,
    FailureCode,
    Finding,
    PreflightReceipt,
    ProbeResult,
    ProbeStatus,
    Severity,
    StrictModel,
    Verdict,
)

MAX_RECEIPT_BYTES = 5 * 1024 * 1024
SPEED_REGRESSION_RATIO = 0.15
CORE_PROBES = frozenset(
    {
        "prefix_cache",
        "context_canary",
        "tool_wire",
        "core_code_10",
        "core_tools_8",
        "core_context_4",
        "speed",
    }
)
_VERDICT_RANK = {Verdict.GO: 0, Verdict.CAUTION: 1, Verdict.INCOMPLETE: 2, Verdict.NO_GO: 3}
_STATUS_RANK = {
    ProbeStatus.PASS: 0,
    ProbeStatus.SKIP: 1,
    ProbeStatus.WARN: 2,
    ProbeStatus.ERROR: 3,
    ProbeStatus.FAIL: 4,
}


class ReceiptError(ValueError):
    """A receipt file is missing, too large, malformed, or an unsupported version."""


class DiffEntry(StrictModel):
    field: str
    a: Any = None
    b: Any = None
    regression: bool = False


class ReceiptDiff(StrictModel):
    same_layout: bool | None
    changes: list[DiffEntry]
    regressions: int


def all_findings(results: Iterable[ProbeResult]) -> list[Finding]:
    return [finding for result in results for finding in result.findings]


def decide_verdict(results: list[ProbeResult]) -> Verdict:
    """A confirmed failure outranks missing data, and missing data outranks warnings."""
    findings = all_findings(results)
    if any(finding.severity is Severity.FAIL for finding in findings):
        return Verdict.NO_GO
    for result in results:
        if result.status is ProbeStatus.ERROR:
            return Verdict.INCOMPLETE
        if result.status is ProbeStatus.SKIP and result.name in CORE_PROBES:
            return Verdict.INCOMPLETE
    if any(finding.severity is Severity.WARN for finding in findings):
        return Verdict.CAUTION
    return Verdict.GO


def failure_codes(results: Iterable[ProbeResult]) -> list[FailureCode]:
    codes = {
        finding.code
        for finding in all_findings(results)
        if finding.code is not None and finding.severity is not Severity.INFO
    }
    return sorted(codes)


def save_receipt(receipt: PreflightReceipt, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_receipt(path: str | Path) -> PreflightReceipt:
    """Load a receipt. Errors never echo file contents, since any path can be passed in."""
    target = Path(path)
    try:
        size = target.stat().st_size
        if size > MAX_RECEIPT_BYTES:
            raise ReceiptError(f"{target.name} is larger than the receipt size limit.")
        raw = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReceiptError(f"Could not read receipt: {target.name}") from exc
    try:
        receipt = PreflightReceipt.model_validate_json(raw)
    except ValidationError:
        raise ReceiptError(f"{target.name} is not a valid preflight receipt.") from None
    if receipt.schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ReceiptError(
            f"{target.name} uses schema {receipt.schema_version}; "
            f"this tool reads {SCHEMA_VERSION}."
        )
    return receipt


def _compare(entries: list[DiffEntry], field: str, a: Any, b: Any, regression: bool = False) -> None:
    if a != b:
        entries.append(DiffEntry(field=field, a=a, b=b, regression=regression))


def _speed_rows(receipt: PreflightReceipt) -> dict[str, dict[str, Any]]:
    for result in receipt.probe_results:
        if result.name == "speed":
            depths = result.metrics.get("depths")
            return depths if isinstance(depths, dict) else {}
    return {}


def _diff_speed(a: PreflightReceipt, b: PreflightReceipt, entries: list[DiffEntry]) -> None:
    rows_a, rows_b = _speed_rows(a), _speed_rows(b)
    metrics = (("ttft_s", True), ("prompt_tps", False), ("generation_tps", False))
    for depth in sorted(set(rows_a) & set(rows_b), key=int):
        for metric, lower_is_better in metrics:
            before = (rows_a[depth].get(metric) or {}).get("median")
            after = (rows_b[depth].get(metric) or {}).get("median")
            if not before or not after:
                continue
            change = (after - before) / before
            if abs(change) < SPEED_REGRESSION_RATIO:
                continue
            worse = change > 0 if lower_is_better else change < 0
            entries.append(
                DiffEntry(
                    field=f"speed.{depth}.{metric}.median", a=before, b=after, regression=worse
                )
            )


def diff_receipts(a: PreflightReceipt, b: PreflightReceipt) -> ReceiptDiff:
    """Report what changed from receipt A to receipt B."""
    entries: list[DiffEntry] = []
    _compare(
        entries, "verdict", a.verdict, b.verdict, _VERDICT_RANK[b.verdict] > _VERDICT_RANK[a.verdict]
    )
    _compare(entries, "tensor_sha256", a.tensor_sha256, b.tensor_sha256)
    quant_a = a.gguf_metadata.quantization if a.gguf_metadata else None
    quant_b = b.gguf_metadata.quantization if b.gguf_metadata else None
    _compare(entries, "quantization", quant_a, quant_b)
    for name in ("adapter", "model", "version", "context_length"):
        before, after = getattr(a.runtime_info, name), getattr(b.runtime_info, name)
        shrank = name == "context_length" and bool(before and after and after < before)
        _compare(entries, f"runtime.{name}", before, after, shrank)
    _compare(entries, "target_agent", a.target_agent, b.target_agent)

    codes_a, codes_b = set(a.failure_codes), set(b.failure_codes)
    for code in sorted(codes_b - codes_a):
        entries.append(DiffEntry(field=f"failure_code.{code}", a=None, b="new", regression=True))
    for code in sorted(codes_a - codes_b):
        entries.append(DiffEntry(field=f"failure_code.{code}", a="present", b="resolved"))

    probes_a = {result.name: result for result in a.probe_results}
    probes_b = {result.name: result for result in b.probe_results}
    for name in sorted(set(probes_a) & set(probes_b)):
        before_probe, after_probe = probes_a[name], probes_b[name]
        _compare(
            entries,
            f"probe.{name}.status",
            before_probe.status,
            after_probe.status,
            _STATUS_RANK[after_probe.status] > _STATUS_RANK[before_probe.status],
        )
        passed_a, passed_b = before_probe.metrics.get("passed"), after_probe.metrics.get("passed")
        if isinstance(passed_a, int) and isinstance(passed_b, int):
            _compare(entries, f"probe.{name}.passed", passed_a, passed_b, passed_b < passed_a)
    _diff_speed(a, b, entries)
    _compare(entries, "transcript_sha256", a.transcript_sha256, b.transcript_sha256)

    same_layout = None
    if a.tensor_sha256 and b.tensor_sha256:
        same_layout = a.tensor_sha256 == b.tensor_sha256
    return ReceiptDiff(
        same_layout=same_layout,
        changes=entries,
        regressions=sum(entry.regression for entry in entries),
    )
