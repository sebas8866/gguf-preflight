"""Runs every probe in order and assembles the receipt."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

import httpx

from gguf_preflight import __version__
from gguf_preflight.adapters import detect_adapter
from gguf_preflight.adapters.base import (
    ChatRequest,
    EndpointError,
    LlmClient,
    api_key_headers,
    normalize_endpoint,
)
from gguf_preflight.inspector.gguf_header import GgufError, _safe_source_label, inspect_gguf
from gguf_preflight.inspector.hardware import detect_hardware
from gguf_preflight.models import (
    Finding,
    GgufMetadata,
    PreflightReceipt,
    ProbeResult,
    ProbeStatus,
    Severity,
)
from gguf_preflight.probes import status_from_findings
from gguf_preflight.probes.agent_smoke import run_agent_smoke
from gguf_preflight.probes.speed_probe import DEFAULT_SPEED_DEPTHS, run_speed_probe
from gguf_preflight.probes.wire_probe import (
    DEFAULT_DEPTHS,
    TEMPLATE_DISABLED_CHECK,
    check_agent_context,
    is_template_disabled_error,
    run_wire_probes,
)
from gguf_preflight.registry.receipt import all_findings, decide_verdict, failure_codes
from gguf_preflight.remediate.rules import recommend

DEEP_DEPTH = 16000
# Leaves room for the template, the question and the reply inside the loaded context.
SPEED_CONTEXT_MARGIN = 0.9
_OUT_OF_MEMORY = re.compile(r"out of memory|\boom\b|failed to allocate", re.IGNORECASE)
# Without --jinja the chat template differs, so these say nothing about the model.
_TEMPLATE_CONFOUNDED_CHECKS = frozenset(
    {
        "canary_recall_failed",
        "smoke_code_collapse",
        "smoke_code_low",
        "smoke_tools_collapse",
        "smoke_tools_low",
        "smoke_context_collapse",
        "smoke_context_low",
    }
)
Status = Callable[[str], None]


def _new_http() -> httpx.Client:
    # trust_env=False keeps test prompts from being routed through an environment proxy.
    return httpx.Client(
        timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0),
        headers=api_key_headers(),
        trust_env=False,
    )


def _warm_up(client: LlmClient) -> None:
    """Load the model first, so load time is not measured and runtime details are real."""
    request = ChatRequest(messages=[{"role": "user", "content": "Reply ok."}], max_tokens=1)
    result = client.chat(request, "openai")
    if result.ok:
        return
    fallback = client.chat(request, "anthropic")
    if fallback.ok:
        return
    raise EndpointError(f"Model {client.model!r} did not respond: {result.error or fallback.error}")


def _safe_error(exc: Exception, ref: str) -> str:
    message = str(exc).replace(ref, _safe_source_label(ref))
    return re.sub(r"(https?://[^\s?]+)\?\S*", r"\1", message)[:300]


def _inspect(ref: str) -> tuple[GgufMetadata | None, ProbeResult]:
    started = time.perf_counter()
    try:
        metadata = inspect_gguf(ref)
    except (GgufError, OSError) as exc:
        return None, ProbeResult(
            name="gguf_header",
            status=ProbeStatus.ERROR,
            duration_s=round(time.perf_counter() - started, 3),
            summary=f"Could not read GGUF: {_safe_error(exc, ref)}",
        )
    return metadata, ProbeResult(
        name="gguf_header",
        status=status_from_findings(metadata.findings),
        duration_s=round(time.perf_counter() - started, 3),
        summary=f"{metadata.architecture or 'unknown'} {metadata.quantization or ''}".strip(),
        findings=metadata.findings,
    )


def _guarded(name: str, probe: Callable[[], list[ProbeResult]]) -> list[ProbeResult]:
    """One crashed probe must not discard the evidence from the others."""
    started = time.perf_counter()
    try:
        return probe()
    except Exception as exc:  # noqa: BLE001
        return [
            ProbeResult(
                name=name,
                status=ProbeStatus.ERROR,
                duration_s=round(time.perf_counter() - started, 3),
                summary=f"{type(exc).__name__}: {str(exc)[:240]}",
            )
        ]


def _runtime_health(errors: list[str]) -> ProbeResult | None:
    """Report server crashes separately, since they are not model or wire defects."""
    errors = [error for error in errors if not is_template_disabled_error(error)]
    if not errors:
        return None
    out_of_memory = any(_OUT_OF_MEMORY.search(error) for error in errors)
    message = (
        f"The server ran out of GPU memory; {len(errors)} request(s) failed."
        if out_of_memory
        else f"{len(errors)} request(s) failed on the server. Check its logs."
    )
    finding = Finding(
        check="runtime_out_of_memory" if out_of_memory else "runtime_request_failed",
        severity=Severity.WARN,
        message=message,
        evidence={"failed_requests": len(errors)},
    )
    return ProbeResult(
        name="runtime_health",
        status=ProbeStatus.ERROR,
        summary=message,
        findings=[finding],
    )


def _discount_template_confounded(results: list[ProbeResult]) -> list[ProbeResult]:
    """Keep model-quality scores on the receipt, but do not blame the model for --jinja."""
    if not any(finding.check == TEMPLATE_DISABLED_CHECK for finding in all_findings(results)):
        return results
    updated: list[ProbeResult] = []
    for result in results:
        if not any(finding.check in _TEMPLATE_CONFOUNDED_CHECKS for finding in result.findings):
            updated.append(result)
            continue
        findings = [
            finding.model_copy(
                update={
                    "severity": Severity.INFO,
                    "code": None,
                    "message": f"Not scored because --jinja is off. {finding.message}",
                    "evidence": {**finding.evidence, "confounded_by": TEMPLATE_DISABLED_CHECK},
                }
            )
            if finding.check in _TEMPLATE_CONFOUNDED_CHECKS
            else finding
            for finding in result.findings
        ]
        real = [finding for finding in findings if finding.severity is not Severity.INFO]
        updated.append(
            result.model_copy(
                update={
                    "findings": findings,
                    "status": status_from_findings(real) if real else ProbeStatus.SKIP,
                    "summary": f"Not scored because --jinja is off. {result.summary}",
                }
            )
        )
    return updated


def _speed(
    client: LlmClient, context_length: int | None, depths: tuple[int, ...]
) -> list[ProbeResult]:
    usable = depths
    if context_length:
        usable = tuple(d for d in depths if d <= context_length * SPEED_CONTEXT_MARGIN)
    if not usable:
        return [
            ProbeResult(
                name="speed",
                status=ProbeStatus.SKIP,
                summary=f"Loaded context {context_length} is below every test depth.",
                metrics={"skipped_depths": list(depths)},
            )
        ]
    result = run_speed_probe(client, usable)
    skipped = [depth for depth in depths if depth not in usable]
    if skipped:
        metrics: dict[str, Any] = {**result.metrics, "skipped_depths": skipped}
        result = result.model_copy(update={"metrics": metrics})
    return [result]


def run_preflight(
    endpoint: str,
    model: str,
    gguf_ref: str | None = None,
    agent: str = "generic",
    deep: bool = False,
    allow_remote: bool = False,
    http: httpx.Client | None = None,
    status: Status | None = None,
) -> PreflightReceipt:
    """Run the full preflight. Raises EndpointError when no usable server or model exists."""
    notify: Status = status or (lambda _message: None)
    base_url = normalize_endpoint(endpoint, allow_remote)
    started = time.perf_counter()
    wire_depths = (*DEFAULT_DEPTHS, DEEP_DEPTH) if deep else DEFAULT_DEPTHS
    speed_depths = (*DEFAULT_SPEED_DEPTHS, DEEP_DEPTH) if deep else DEFAULT_SPEED_DEPTHS
    owned = http is None
    client_http = http or _new_http()
    try:
        adapter = detect_adapter(client_http, base_url)
        client = LlmClient(client_http, base_url, model, adapter)
        notify(f"Loading {model} on {adapter.name}")
        _warm_up(client)
        runtime = adapter.runtime_info(model)
        hardware = detect_hardware()

        results: list[ProbeResult] = []
        metadata: GgufMetadata | None = None
        if gguf_ref:
            notify("Reading GGUF header")
            metadata, header_result = _inspect(gguf_ref)
            results.append(header_result)
        results.append(check_agent_context(runtime, agent))
        notify("Probing the wire: cache, truncation, tool calls")
        results.extend(
            _guarded("wire", lambda: run_wire_probes(client, runtime, wire_depths))
        )

        def task_done(task_id: str, passed: bool) -> None:
            notify(f"{task_id}: {'pass' if passed else 'fail'}")

        notify("Running the 22-task agent smoke suite")
        results.extend(_guarded("agent_smoke", lambda: run_agent_smoke(client, progress=task_done)))
        notify("Measuring speed")
        results.extend(
            _guarded("speed", lambda: _speed(client, runtime.context_length, speed_depths))
        )
        health = _runtime_health(client.server_errors)
        if health is not None:
            results.append(health)
    finally:
        if owned:
            client_http.close()

    results = _discount_template_confounded(results)
    return PreflightReceipt(
        tool_version=__version__,
        target_agent=agent,
        verdict=decide_verdict(results),
        duration_s=round(time.perf_counter() - started, 2),
        tensor_sha256=metadata.tensor_sha256 if metadata else None,
        gguf_metadata=metadata,
        runtime_info=runtime,
        hardware_info=hardware,
        probe_results=results,
        failure_codes=failure_codes(results),
        recommended_fixes=recommend(all_findings(results), runtime.adapter, agent),
        transcript_sha256=client.transcript.sha256(),
    )
