"""Wire-level probes for bugs that break coding agents while HTTP still returns 200."""

from __future__ import annotations

import hashlib
import re
import secrets
import statistics
import time
from collections.abc import Sequence
from typing import Any

from gguf_preflight.adapters.base import (
    ChatRequest,
    ChatResult,
    LlmClient,
    Protocol,
    ToolSpec,
)
from gguf_preflight.models import (
    FailureCode,
    Finding,
    ProbeResult,
    ProbeStatus,
    RuntimeInfo,
    Severity,
)
from gguf_preflight.probes import status_from_findings

PROTOCOLS: tuple[Protocol, ...] = ("openai", "anthropic")
DEFAULT_DEPTHS: tuple[int, ...] = (4000, 8000)
PREFIX_TURNS = 4
PREFIX_LINES = 90
REUSE_PASS = 0.7
REUSE_FAIL = 0.3
LATENCY_MISS_RATIO = 0.75
MIN_COLD_TTFT_S = 0.15
TRUNCATION_RATIO = 0.8
CALIBRATION_LINES = 40
DEFAULT_TOKENS_PER_LINE = 25.0
CANARY_OVERHEAD_TOKENS = 120

# Minimums from each agent's guidance. Ollama documents 64k+ for Claude Code.
AGENT_MIN_CONTEXT: dict[str, int] = {"claude-code": 64000}
AGENT_HARD_MIN_CONTEXT = 16000

_LEAK_PATTERN = re.compile(
    r"<\|?tool_call\|?>|</tool_call>|\[TOOL_CALLS\]|<function=|<\|python_tag\|>",
    re.IGNORECASE,
)
_CONTEXT_ERROR = re.compile(r"context|exceed|too long|too many tokens", re.IGNORECASE)

PROBE_TOOL = ToolSpec(
    name="record_value",
    description="Store a value in the audit log.",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string", "description": "The value to store."}},
        "required": ["value"],
    },
)
PROBE_TOOL_VALUE = "alpha-7"


def filler_lines(count: int, label: str = "log") -> str:
    """Deterministic, low-repetition text used to reach a target context depth."""
    return "\n".join(
        f"{label} entry {index:05d}: service=node{(index * 7) % 13} status=ok "
        f"latency={(index * 37) % 211}ms region=r{(index * 11) % 5}"
        for index in range(count)
    )


def _finding(
    check: str,
    severity: Severity,
    code: FailureCode | None,
    message: str,
    **evidence: Any,
) -> Finding:
    return Finding(check=check, severity=severity, code=code, message=message, evidence=evidence)


def _available(client: LlmClient) -> list[Protocol]:
    return [protocol for protocol in PROTOCOLS if client.supports(protocol)]


def _reuse(result: ChatResult) -> float | None:
    usage = result.usage
    if usage.cached_tokens is None or not usage.prompt_tokens:
        return None
    return usage.cached_tokens / usage.prompt_tokens


def _evaluate_prefix(protocol: Protocol, rows: list[ChatResult]) -> tuple[dict[str, Any], Finding | None]:
    warm = rows[1:]
    ratios = [ratio for ratio in (_reuse(row) for row in warm) if ratio is not None]
    metrics: dict[str, Any] = {"turns": len(rows)}
    if ratios and len(ratios) == len(warm):
        reuse = statistics.median(ratios)
        metrics.update(signal="cached_tokens", warm_reuse_ratio=round(reuse, 3))
        if reuse >= REUSE_PASS:
            return metrics, None
        severity = Severity.FAIL if reuse < REUSE_FAIL else Severity.WARN
        return metrics, _finding(
            "prefix_cache_invalidated",
            severity,
            FailureCode.PREFIX_CACHE_MISS,
            f"{protocol}: only {reuse:.0%} of the prompt was reused on warm turns.",
            protocol=protocol,
            signal="cached_tokens",
            warm_reuse_ratio=round(reuse, 3),
        )

    cold = rows[0].ttft_s
    warm_ttft = [row.ttft_s for row in warm if row.ttft_s is not None]
    if cold is None or cold < MIN_COLD_TTFT_S or not warm_ttft:
        metrics["signal"] = "inconclusive"
        return metrics, None
    ratio = statistics.median(warm_ttft) / cold
    metrics.update(signal="ttft", warm_cold_ttft_ratio=round(ratio, 3))
    if ratio <= LATENCY_MISS_RATIO:
        return metrics, None
    return metrics, _finding(
        "prefix_cache_invalidated",
        Severity.WARN,
        FailureCode.PREFIX_CACHE_MISS,
        f"{protocol}: warm turns were not meaningfully faster (latency signal only).",
        protocol=protocol,
        signal="ttft",
        warm_cold_ttft_ratio=round(ratio, 3),
    )


def probe_prefix_cache(client: LlmClient, nonce: str) -> ProbeResult:
    """Keep a static system+tools prefix while a per-turn reminder changes after it."""
    started = time.perf_counter()
    findings: list[Finding] = []
    metrics: dict[str, Any] = {}
    protocols = _available(client)
    if not protocols:
        return ProbeResult(
            name="prefix_cache", status=ProbeStatus.SKIP, summary="No supported chat protocol."
        )
    for protocol in protocols:
        run_id = f"{nonce}-{protocol}"
        client.transcript.redact(run_id, f"<NONCE-{protocol}>")
        static = f"Session {run_id}. You are a coding agent.\n{filler_lines(PREFIX_LINES, 'rule')}"
        rows: list[ChatResult] = []
        for turn in range(PREFIX_TURNS):
            reminder = f"<system-reminder>total_tokens: {200000 - turn * 137}</system-reminder>"
            result = client.chat(
                ChatRequest(
                    system=f"{static}\n{reminder}",
                    messages=[{"role": "user", "content": "Reply with the single word ok."}],
                    tools=[PROBE_TOOL],
                    max_tokens=8,
                    stream=True,
                ),
                protocol,
            )
            if not result.ok:
                findings.append(
                    _finding(
                        "prefix_probe_request_failed",
                        Severity.WARN,
                        None if result.server_failed else FailureCode.WIRE_BUG,
                        f"{protocol}: request failed on turn {turn + 1}: {result.error}",
                        protocol=protocol,
                        status_code=result.status_code,
                    )
                )
                break
            rows.append(result)
        if len(rows) == PREFIX_TURNS:
            protocol_metrics, finding = _evaluate_prefix(protocol, rows)
            metrics[protocol] = protocol_metrics
            if finding:
                findings.append(finding)
    return ProbeResult(
        name="prefix_cache",
        status=status_from_findings(findings),
        duration_s=round(time.perf_counter() - started, 3),
        summary=_summarize(findings, "Prefix cache is retained across turns."),
        metrics=metrics,
        findings=findings,
    )


def _calibrate_tokens_per_line(client: LlmClient) -> float:
    base = client.chat(
        ChatRequest(system="x", messages=[{"role": "user", "content": "ok"}], max_tokens=1)
    )
    sample = client.chat(
        ChatRequest(
            system=filler_lines(CALIBRATION_LINES, "calibration"),
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
        )
    )
    if base.usage.prompt_tokens and sample.usage.prompt_tokens:
        delta = sample.usage.prompt_tokens - base.usage.prompt_tokens
        if delta > 0:
            return delta / CALIBRATION_LINES
    return DEFAULT_TOKENS_PER_LINE


def probe_context_canary(
    client: LlmClient,
    runtime: RuntimeInfo,
    nonce: str,
    depths: Sequence[int] = DEFAULT_DEPTHS,
) -> ProbeResult:
    """Place a canary at the start of the prompt and ask for it from the far end."""
    started = time.perf_counter()
    findings: list[Finding] = []
    tokens_per_line = _calibrate_tokens_per_line(client)
    results: dict[str, Any] = {}
    for depth in sorted(depths):
        digest = hashlib.sha256(f"{nonce}:{depth}".encode()).hexdigest()[:10].upper()
        canary = f"CANARY-{digest}"
        client.transcript.redact(canary, f"<CANARY-{depth}>")
        lines = max(1, int((depth - CANARY_OVERHEAD_TOKENS) / tokens_per_line))
        expected = int(lines * tokens_per_line + CANARY_OVERHEAD_TOKENS * 0.5)
        result = client.chat(
            ChatRequest(
                system=(
                    f"The security code is {canary}. You will be asked for it later.\n\n"
                    f"Reference log:\n{filler_lines(lines, f'd{depth}')}"
                ),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "What is the security code stated at the very start of the "
                            "system prompt? Reply with the code only."
                        ),
                    }
                ],
                max_tokens=32,
            )
        )
        row: dict[str, Any] = {
            "expected_prompt_tokens": expected,
            "reported_prompt_tokens": result.usage.prompt_tokens,
        }
        results[str(depth)] = row
        if not result.ok:
            row["outcome"] = "error"
            explicit = bool(result.error and _CONTEXT_ERROR.search(result.error))
            findings.append(
                _finding(
                    "context_limit_explicit" if explicit else "context_probe_request_failed",
                    Severity.WARN,
                    FailureCode.WIRE_BUG if explicit or not result.server_failed else None,
                    f"Depth {depth}: server returned an error: {result.error}",
                    depth=depth,
                    status_code=result.status_code,
                )
            )
            break
        if canary in result.text:
            row["outcome"] = "retained"
            continue
        reported = result.usage.prompt_tokens
        truncated = reported is not None and reported < expected * TRUNCATION_RATIO
        limited = runtime.context_length is not None and runtime.context_length < depth
        if truncated or limited:
            row["outcome"] = "truncated"
            findings.append(
                _finding(
                    "silent_context_truncation",
                    Severity.FAIL,
                    FailureCode.WIRE_BUG,
                    f"Depth {depth}: the start of the prompt was dropped with HTTP 200.",
                    depth=depth,
                    expected_prompt_tokens=expected,
                    reported_prompt_tokens=reported,
                    runtime_context_length=runtime.context_length,
                )
            )
            break
        row["outcome"] = "recall_failed"
        findings.append(
            _finding(
                "canary_recall_failed",
                Severity.WARN,
                FailureCode.QUANT_COLLAPSE,
                f"Depth {depth}: the prompt fit, but the model did not repeat the canary.",
                depth=depth,
                reported_prompt_tokens=reported,
            )
        )
    return ProbeResult(
        name="context_canary",
        status=status_from_findings(findings),
        duration_s=round(time.perf_counter() - started, 3),
        summary=_summarize(findings, "Canary survived at every tested depth."),
        metrics={"tokens_per_line": round(tokens_per_line, 2), "depths": results},
        findings=findings,
    )


def classify_tool_result(result: ChatResult, mode: str) -> Finding | None:
    """Classify one tool-call response. Returns None when the call is valid."""
    if result.server_failed:
        return _finding(
            "tool_probe_server_failed",
            Severity.WARN,
            None,
            f"{mode}: the server failed, so this mode was not verified.",
            mode=mode,
            status_code=result.status_code,
        )
    if not result.ok:
        return _finding(
            "tool_call_http_error",
            Severity.FAIL,
            FailureCode.WIRE_BUG,
            f"{mode}: request failed: {result.error or result.status_code}",
            mode=mode,
            status_code=result.status_code,
        )
    if result.malformed_events:
        return _finding(
            "tool_stream_malformed",
            Severity.FAIL,
            FailureCode.WIRE_BUG,
            f"{mode}: stream contained {result.malformed_events} malformed events.",
            mode=mode,
        )
    if result.tool_calls:
        call = result.tool_calls[0]
        if call.parse_error:
            return _finding(
                "tool_args_invalid_json",
                Severity.FAIL,
                FailureCode.WIRE_BUG,
                f"{mode}: tool arguments could not be parsed ({call.parse_error}).",
                mode=mode,
                raw=call.arguments_raw[:200],
            )
        if call.name != PROBE_TOOL.name or not (call.arguments or {}).get("value"):
            return _finding(
                "tool_call_wrong_shape",
                Severity.WARN,
                FailureCode.QUANT_COLLAPSE,
                f"{mode}: tool call had the wrong name or missing arguments.",
                mode=mode,
                name=call.name,
            )
        return None
    if _LEAK_PATTERN.search(result.text):
        return _finding(
            "tool_call_plaintext_leak",
            Severity.FAIL,
            FailureCode.WIRE_BUG,
            f"{mode}: tool-call markup reached the client as plain text.",
            mode=mode,
            sample=result.text[:160],
        )
    if result.budget_exhausted:
        return _finding(
            "tool_call_budget_exhausted",
            Severity.WARN,
            None,
            f"{mode}: the token limit was reached before a tool call was produced.",
            mode=mode,
        )
    if result.empty:
        return _finding(
            "tool_call_empty_response",
            Severity.FAIL,
            FailureCode.WIRE_BUG,
            f"{mode}: HTTP 200 with no text and no tool call (silent drop).",
            mode=mode,
            finish_reason=result.finish_reason,
        )
    return _finding(
        "tool_not_called",
        Severity.WARN,
        None,
        f"{mode}: the model answered in prose instead of calling the tool.",
        mode=mode,
    )


def probe_tool_wire(client: LlmClient) -> ProbeResult:
    started = time.perf_counter()
    findings: list[Finding] = []
    outcomes: dict[str, str] = {}
    protocols = _available(client)
    if not protocols:
        return ProbeResult(
            name="tool_wire", status=ProbeStatus.SKIP, summary="No supported chat protocol."
        )
    for protocol in protocols:
        for stream in (False, True):
            mode = f"{protocol}/{'stream' if stream else 'non-stream'}"
            result = client.chat(
                ChatRequest(
                    system="You must use the provided tool. Do not answer in text.",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Call record_value with value exactly {PROBE_TOOL_VALUE}.",
                        }
                    ],
                    tools=[PROBE_TOOL],
                    max_tokens=128,
                    stream=stream,
                ),
                protocol,
            )
            finding = classify_tool_result(result, mode)
            outcomes[mode] = finding.check if finding else "ok"
            if finding:
                findings.append(finding)
    return ProbeResult(
        name="tool_wire",
        status=status_from_findings(findings),
        duration_s=round(time.perf_counter() - started, 3),
        summary=_summarize(findings, "Tool calls arrive intact in every mode."),
        metrics={"modes": outcomes},
        findings=findings,
    )


def check_agent_context(runtime: RuntimeInfo, agent: str) -> ProbeResult:
    """Compare the runtime's loaded context with what the target agent needs."""
    minimum = AGENT_MIN_CONTEXT.get(agent)
    metrics = {"runtime_context": runtime.context_length, "agent_minimum": minimum}
    if minimum is None or runtime.context_length is None:
        return ProbeResult(
            name="agent_context",
            status=ProbeStatus.SKIP,
            summary="No context requirement or runtime value to compare.",
            metrics=metrics,
        )
    if runtime.context_length >= minimum:
        return ProbeResult(
            name="agent_context",
            status=ProbeStatus.PASS,
            summary=f"Context {runtime.context_length} meets the {agent} guidance.",
            metrics=metrics,
        )
    severity = Severity.FAIL if runtime.context_length < AGENT_HARD_MIN_CONTEXT else Severity.WARN
    model_max = runtime.details.get("model_max_context")
    if isinstance(model_max, int) and model_max < minimum and runtime.context_length >= model_max:
        finding = _finding(
            "model_context_limit",
            severity,
            None,
            f"This model's maximum context is {model_max}; {agent} guidance is {minimum}+. "
            "A higher context setting will not help.",
            runtime_context=runtime.context_length,
            model_max_context=model_max,
            agent_minimum=minimum,
        )
    else:
        finding = _finding(
            "context_below_agent_minimum",
            severity,
            FailureCode.WIRE_BUG,
            f"Loaded context is {runtime.context_length}; {agent} guidance is {minimum}+.",
            runtime_context=runtime.context_length,
            agent_minimum=minimum,
        )
    return ProbeResult(
        name="agent_context",
        status=status_from_findings([finding]),
        summary=finding.message,
        metrics=metrics,
        findings=[finding],
    )


def _summarize(findings: list[Finding], clean: str) -> str:
    if not findings:
        return clean
    return "; ".join(dict.fromkeys(finding.message for finding in findings))[:300]


def run_wire_probes(
    client: LlmClient,
    runtime: RuntimeInfo,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    nonce: str | None = None,
) -> list[ProbeResult]:
    run_nonce = nonce or secrets.token_hex(4)
    client.transcript.redact(run_nonce, "<NONCE>")
    return [
        probe_prefix_cache(client, run_nonce),
        probe_context_canary(client, runtime, run_nonce, depths),
        probe_tool_wire(client),
    ]
