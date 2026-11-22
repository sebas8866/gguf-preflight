"""Context-depth speed probe reporting median and IQR over warm iterations."""

from __future__ import annotations

import secrets
import statistics
import time
from collections.abc import Sequence
from typing import Any

from gguf_preflight.adapters.base import ChatRequest, ChatResult, LlmClient, Protocol
from gguf_preflight.models import FailureCode, Finding, ProbeResult, ProbeStatus, Severity
from gguf_preflight.probes import status_from_findings
from gguf_preflight.probes.wire_probe import DEFAULT_TOKENS_PER_LINE, filler_lines

DEFAULT_SPEED_DEPTHS: tuple[int, ...] = (2000, 8000)
WARM_ITERATIONS = 3
GENERATION_TOKENS = 48
PREFILL_WARN_TPS = 300.0
PREFILL_FAIL_TPS = 60.0
GENERATION_WARN_TPS = 15.0
GENERATION_FAIL_TPS = 5.0


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    """Median and interquartile range, using inclusive quartiles for small samples."""
    if not values:
        return {"median": None, "iqr": None, "n": 0}
    if len(values) == 1:
        return {"median": round(values[0], 4), "iqr": 0.0, "n": 1}
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median": round(statistics.median(values), 4),
        "iqr": round(q3 - q1, 4),
        "n": len(values),
    }


def generation_tps(result: ChatResult) -> float | None:
    if result.predicted_ms and result.predicted_n:
        return result.predicted_n / (result.predicted_ms / 1000)
    tokens = result.usage.completion_tokens
    if tokens is None or tokens < 2 or result.ttft_s is None:
        return None
    elapsed = result.total_s - result.ttft_s
    # The first token arrives at TTFT, so only later tokens count against this window.
    return (tokens - 1) / elapsed if elapsed > 0 else None


def prefill_tps(result: ChatResult) -> float | None:
    if result.prompt_ms and result.prompt_n:
        return result.prompt_n / (result.prompt_ms / 1000)
    if not result.usage.prompt_tokens or not result.ttft_s:
        return None
    uncached = result.usage.prompt_tokens - (result.usage.cached_tokens or 0)
    return uncached / result.ttft_s if uncached > 0 else None


def effective_prompt_tps(result: ChatResult) -> float | None:
    if not result.usage.prompt_tokens or not result.ttft_s:
        return None
    return result.usage.prompt_tokens / result.ttft_s


def _request(system: str, stream: bool = True) -> ChatRequest:
    return ChatRequest(
        system=system,
        messages=[
            {
                "role": "user",
                "content": "Count upward from 1 to 40, separated by single spaces.",
            }
        ],
        max_tokens=GENERATION_TOKENS,
        stream=stream,
    )


def _present(values: Sequence[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _measure_depth(
    client: LlmClient, depth: int, nonce: str, iterations: int, protocol: Protocol
) -> dict[str, Any] | str:
    lines = max(1, int(depth / DEFAULT_TOKENS_PER_LINE))
    run_id = f"{nonce}-{depth}"
    client.transcript.redact(run_id, f"<NONCE-{depth}>")
    system = f"Run {run_id}. Benchmark context.\n{filler_lines(lines, f's{depth}')}"
    cold = client.chat(_request(system), protocol)
    if not cold.ok:
        return f"cold request failed: {cold.error}"
    warm: list[ChatResult] = []
    for _ in range(iterations):
        result = client.chat(_request(system), protocol)
        if not result.ok:
            return f"warm request failed: {result.error}"
        warm.append(result)
    ttft = _present([result.ttft_s for result in warm])
    warm_median = statistics.median(ttft) if ttft else None
    return {
        "prompt_tokens": cold.usage.prompt_tokens,
        "ttft_s": summarize(ttft),
        "prompt_tps": summarize(_present([effective_prompt_tps(result) for result in warm])),
        "generation_tps": summarize(_present([generation_tps(result) for result in warm])),
        "cold_ttft_s": round(cold.ttft_s, 4) if cold.ttft_s is not None else None,
        "cold_prefill_tps": _round(prefill_tps(cold)),
        "prefill_speedup": (
            round(cold.ttft_s / warm_median, 3) if cold.ttft_s and warm_median else None
        ),
    }


def _round(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _latency_findings(depths: dict[str, dict[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    if not depths:
        return findings
    deepest_key = max(depths, key=int)
    deepest = depths[deepest_key]
    prefill = deepest.get("cold_prefill_tps")
    if prefill is not None and prefill < PREFILL_WARN_TPS:
        findings.append(
            Finding(
                check="slow_prefill",
                severity=Severity.FAIL if prefill < PREFILL_FAIL_TPS else Severity.WARN,
                code=FailureCode.LATENCY_BOTTLENECK,
                message=f"Prompt processing is {prefill:.0f} tok/s at depth {deepest_key}.",
                evidence={"depth": int(deepest_key), "cold_prefill_tps": prefill},
            )
        )
    generation = deepest["generation_tps"]["median"]
    if generation is not None and generation < GENERATION_WARN_TPS:
        findings.append(
            Finding(
                check="slow_generation",
                severity=Severity.FAIL if generation < GENERATION_FAIL_TPS else Severity.WARN,
                code=FailureCode.LATENCY_BOTTLENECK,
                message=f"Generation is {generation:.1f} tok/s at depth {deepest_key}.",
                evidence={"depth": int(deepest_key), "generation_tps": generation},
            )
        )
    return findings


def run_speed_probe(
    client: LlmClient,
    depths: Sequence[int] = DEFAULT_SPEED_DEPTHS,
    iterations: int = WARM_ITERATIONS,
    nonce: str | None = None,
) -> ProbeResult:
    started = time.perf_counter()
    protocol: Protocol = "openai" if client.supports("openai") else "anthropic"
    if not client.supports(protocol):
        return ProbeResult(
            name="speed", status=ProbeStatus.SKIP, summary="No supported chat protocol."
        )
    run_nonce = nonce or secrets.token_hex(4)
    client.transcript.redact(run_nonce, "<NONCE>")
    measured: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for depth in sorted(depths):
        outcome = _measure_depth(client, depth, run_nonce, iterations, protocol)
        if isinstance(outcome, str):
            errors[str(depth)] = outcome
            break
        measured[str(depth)] = outcome
    findings = _latency_findings(measured)
    status = ProbeStatus.ERROR if errors else status_from_findings(findings)
    metrics: dict[str, Any] = {"iterations": iterations, "depths": measured}
    if errors:
        metrics["errors"] = errors
    summary = "; ".join(finding.message for finding in findings) or (
        f"Measured {len(measured)} depth(s)."
    )
    return ProbeResult(
        name="speed",
        status=status,
        duration_s=round(time.perf_counter() - started, 3),
        summary=summary,
        metrics=metrics,
        findings=findings,
    )
