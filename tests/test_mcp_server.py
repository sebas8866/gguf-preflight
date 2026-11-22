from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastmcp import Client
from fastmcp.exceptions import ToolError
from gguf_fixtures import HEALTHY_KV, HEALTHY_TENSORS, build_gguf
from typer.testing import CliRunner

from gguf_preflight.adapters.base import ChatRequest, ChatResult, ToolCall, Transcript
from gguf_preflight.adapters.ollama import OllamaAdapter
from gguf_preflight.cli import app
from gguf_preflight.mcp_server import mcp
from gguf_preflight.models import (
    FailureCode,
    Finding,
    HardwareInfo,
    PreflightReceipt,
    ProbeResult,
    ProbeStatus,
    RuntimeInfo,
    Severity,
    Verdict,
)
from gguf_preflight.probes.wire_probe import check_agent_context, classify_tool_result
from gguf_preflight.registry.receipt import (
    ReceiptError,
    decide_verdict,
    diff_receipts,
    load_receipt,
    save_receipt,
)
from gguf_preflight.remediate.rules import recommend
from gguf_preflight.runner import run_preflight

BASE = "http://127.0.0.1:9999"
runner = CliRunner()


def receipt(
    verdict: Verdict = Verdict.GO,
    passed: int = 10,
    status: ProbeStatus = ProbeStatus.PASS,
    context: int = 64000,
    generation: float = 40.0,
    codes: list[FailureCode] | None = None,
) -> PreflightReceipt:
    return PreflightReceipt(
        tool_version="0.1.0",
        verdict=verdict,
        tensor_sha256="abc",
        runtime_info=RuntimeInfo(
            adapter="ollama", endpoint=BASE, model="m", context_length=context
        ),
        hardware_info=HardwareInfo(os="test", machine="x86_64"),
        probe_results=[
            ProbeResult(
                name="core_code_10", status=status, metrics={"passed": passed, "total": 10}
            ),
            ProbeResult(
                name="speed",
                status=ProbeStatus.PASS,
                metrics={
                    "depths": {
                        "8000": {
                            "ttft_s": {"median": 1.0, "iqr": 0.1, "n": 3},
                            "prompt_tps": {"median": 800.0, "iqr": 5.0, "n": 3},
                            "generation_tps": {"median": generation, "iqr": 1.0, "n": 3},
                            "cold_prefill_tps": 400.0,
                            "prefill_speedup": 4.0,
                        }
                    }
                },
            ),
        ],
        failure_codes=codes or [],
        transcript_sha256="t",
    )


def probe(status: ProbeStatus, name: str = "tool_wire", severity: Severity | None = None) -> ProbeResult:
    findings = (
        [Finding(check="x", severity=severity, message="m", code=FailureCode.WIRE_BUG)]
        if severity
        else []
    )
    return ProbeResult(name=name, status=status, findings=findings)


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ([probe(ProbeStatus.PASS)], Verdict.GO),
        ([probe(ProbeStatus.WARN, severity=Severity.WARN)], Verdict.CAUTION),
        ([probe(ProbeStatus.FAIL, severity=Severity.FAIL)], Verdict.NO_GO),
        ([probe(ProbeStatus.ERROR)], Verdict.INCOMPLETE),
        ([probe(ProbeStatus.SKIP)], Verdict.INCOMPLETE),
        ([probe(ProbeStatus.SKIP, "agent_context")], Verdict.GO),
        (
            [probe(ProbeStatus.ERROR), probe(ProbeStatus.FAIL, severity=Severity.FAIL)],
            Verdict.NO_GO,
        ),
    ],
)
def test_verdict(results: list[ProbeResult], expected: Verdict) -> None:
    assert decide_verdict(results) is expected


def test_receipt_round_trip(tmp_path: Path) -> None:
    original = receipt()
    path = save_receipt(original, tmp_path / "nested" / "a.json")
    assert load_receipt(path) == original


def test_identical_receipts_have_no_diff() -> None:
    result = diff_receipts(receipt(), receipt())
    assert result.changes == []
    assert result.same_layout is True


def test_diff_flags_regressions() -> None:
    result = diff_receipts(
        receipt(),
        receipt(
            verdict=Verdict.NO_GO,
            passed=4,
            status=ProbeStatus.FAIL,
            context=4096,
            generation=10.0,
            codes=[FailureCode.QUANT_COLLAPSE],
        ),
    )
    flagged = {entry.field for entry in result.changes if entry.regression}
    assert {
        "verdict",
        "runtime.context_length",
        "failure_code.QUANT_COLLAPSE",
        "probe.core_code_10.status",
        "probe.core_code_10.passed",
        "speed.8000.generation_tps.median",
    } <= flagged


def test_diff_improvement_is_not_regression() -> None:
    result = diff_receipts(
        receipt(verdict=Verdict.NO_GO, passed=4, context=4096, generation=10.0),
        receipt(),
    )
    assert result.regressions == 0
    assert result.changes


def test_small_speed_noise_is_ignored() -> None:
    assert diff_receipts(receipt(generation=40.0), receipt(generation=42.0)).changes == []


def test_load_rejects_bad_files_without_echoing_content(tmp_path: Path) -> None:
    secret = tmp_path / "secret.json"
    secret.write_text('{"api_key": "TOP-SECRET-VALUE"}', encoding="utf-8")
    with pytest.raises(ReceiptError) as caught:
        load_receipt(secret)
    assert "TOP-SECRET-VALUE" not in str(caught.value)
    with pytest.raises(ReceiptError):
        load_receipt(tmp_path / "missing.json")


def test_load_rejects_future_schema(tmp_path: Path) -> None:
    path = save_receipt(receipt().model_copy(update={"schema_version": "9.0"}), tmp_path / "a.json")
    with pytest.raises(ReceiptError, match="schema"):
        load_receipt(path)


def call(tool: str, **arguments: Any) -> Any:
    async def _run() -> Any:
        async with Client(mcp) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(_run())


def test_mcp_lists_exact_tools() -> None:
    async def _names() -> set[str]:
        async with Client(mcp) as client:
            return {tool.name for tool in await client.list_tools()}

    assert asyncio.run(_names()) == {"preflight_check", "inspect_gguf", "compare_receipts"}


def test_mcp_inspect_local_gguf(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(build_gguf(HEALTHY_KV, HEALTHY_TENSORS))
    result = call("inspect_gguf", path_or_url=str(path))
    assert result.data["source"] == "model.gguf"
    assert str(tmp_path) not in json.dumps(result.structured_content)


def test_mcp_compare_receipts(tmp_path: Path) -> None:
    a = save_receipt(receipt(), tmp_path / "a.json")
    b = save_receipt(receipt(passed=3), tmp_path / "b.json")
    result = call("compare_receipts", receipt_a_path=str(a), receipt_b_path=str(b))
    assert result.data["regressions"] == 1


@pytest.mark.parametrize(
    "endpoint",
    ["https://api.openai.com", "http://8.8.8.8:11434", "http://169.254.169.254"],
)
def test_mcp_refuses_non_local_endpoints(endpoint: str) -> None:
    with pytest.raises(ToolError, match="not a local address"):
        call("preflight_check", endpoint=endpoint, model="m")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/model.gguf",
        "http://169.254.169.254/latest/meta-data",
        "https://example.com/model.gguf",
        "http://huggingface.co/a/b/resolve/main/m.gguf",
    ],
)
def test_mcp_refuses_non_hf_urls(url: str) -> None:
    with pytest.raises(ToolError, match="Hugging Face"):
        call("inspect_gguf", path_or_url=url)


def test_mcp_refuses_unknown_agent() -> None:
    with pytest.raises(ToolError, match="target_agent"):
        call("preflight_check", endpoint=BASE, model="m", target_agent="other")


def test_mcp_bad_receipt_error_is_clean(tmp_path: Path) -> None:
    secret = tmp_path / "private.json"
    secret.write_text('{"token": "TOP-SECRET-VALUE"}', encoding="utf-8")
    with pytest.raises(ToolError) as caught:
        call("compare_receipts", receipt_a_path=str(secret), receipt_b_path=str(secret))
    assert "TOP-SECRET-VALUE" not in str(caught.value)


def fake_server() -> None:
    respx.get(f"{BASE}/api/version").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/api/v1/models").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/api/v0/models").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "m"}]})
    )
    respx.post(f"{BASE}/v1/messages").mock(return_value=httpx.Response(404))

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            events = [
                {"choices": [{"delta": {"content": "fluent nonsense"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 5}},
            ]
            body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "fluent nonsense"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=handler)


@respx.mock
def test_full_run_builds_complete_receipt() -> None:
    fake_server()
    with httpx.Client(timeout=5) as http:
        result = run_preflight(BASE, "m", agent="generic", http=http)
    names = {probe_result.name for probe_result in result.probe_results}
    assert {
        "prefix_cache",
        "context_canary",
        "tool_wire",
        "core_code_10",
        "core_tools_8",
        "core_context_4",
        "speed",
    } <= names
    assert result.verdict is Verdict.NO_GO
    assert FailureCode.QUANT_COLLAPSE in result.failure_codes
    assert result.recommended_fixes
    assert result.transcript_sha256


@respx.mock
@pytest.mark.parametrize(
    ("requested", "expected"),
    [("my-model", 40960), ("my-model:latest", 40960), ("my-model:other", None), ("my", None)],
)
def test_ollama_matches_implicit_latest_tag(requested: str, expected: int | None) -> None:
    respx.get(f"{BASE}/api/version").mock(
        return_value=httpx.Response(200, json={"version": "0.40.12"})
    )
    respx.post(f"{BASE}/api/show").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{BASE}/api/ps").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "my-model:latest",
                        "model": "my-model:latest",
                        "size": 100,
                        "size_vram": 69,
                        "context_length": 40960,
                    }
                ]
            },
        )
    )
    with httpx.Client() as http:
        info = OllamaAdapter(http, BASE).runtime_info(requested)
    assert info.context_length == expected


@respx.mock
def test_gpu_crash_is_incomplete_not_a_wire_bug() -> None:
    fake_server()
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                },
            )
        return httpx.Response(500, json={"error": {"message": "CUDA error: out of memory"}})

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=handler)
    with httpx.Client(timeout=5) as http:
        result = run_preflight(BASE, "m", http=http)
    assert result.verdict is Verdict.INCOMPLETE
    assert result.failure_codes == []
    health = next(p for p in result.probe_results if p.name == "runtime_health")
    assert health.findings[0].check == "runtime_out_of_memory"
    assert any("GPU memory" in fix.title for fix in result.recommended_fixes)


def test_model_context_limit_does_not_suggest_raising_context() -> None:
    capped = RuntimeInfo(
        adapter="ollama",
        endpoint=BASE,
        model="m",
        context_length=40960,
        details={"model_max_context": 40960},
    )
    finding = check_agent_context(capped, "claude-code").findings[0]
    assert finding.check == "model_context_limit"
    assert finding.code is None
    fixes = recommend([finding], "ollama", "claude-code")
    assert fixes[0].title == "Use a model with a larger context window"
    assert not any("OLLAMA_CONTEXT_LENGTH" in command for command in fixes[0].commands)

    raisable = capped.model_copy(update={"context_length": 4096})
    assert check_agent_context(raisable, "claude-code").findings[0].check == (
        "context_below_agent_minimum"
    )


def test_client_error_is_still_a_wire_bug() -> None:
    finding = classify_tool_result(ChatResult("openai", False, 400, error="bad request"), "m")
    assert finding is not None
    assert finding.code is FailureCode.WIRE_BUG
    crash = classify_tool_result(ChatResult("openai", False, 500, error="crashed"), "m")
    assert crash is not None
    assert crash.code is None


def test_transcript_hash_ignores_json_key_order() -> None:
    def digest(arguments: str) -> str:
        transcript = Transcript()
        result = ChatResult("openai", False, 200, tool_calls=[ToolCall.parse("t", arguments)])
        transcript.add(ChatRequest(messages=[{"role": "user", "content": "x"}]), result)
        return transcript.sha256()

    assert digest('{"a":1,"b":{"c":2,"d":3}}') == digest('{"b":{"d":3,"c":2},"a":1}')
    assert digest('{"a":1}') != digest('{"a":2}')
    assert digest("{broken") != digest("{broken2")


def test_cli_refuses_public_host_without_flag() -> None:
    outcome = runner.invoke(app, ["check", "https://api.openai.com", "--model", "m"])
    assert outcome.exit_code == 2
    assert "not a local address" in outcome.output


def test_cli_inspect_json(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(build_gguf(HEALTHY_KV, HEALTHY_TENSORS))
    outcome = runner.invoke(app, ["inspect", str(path), "--json"])
    assert outcome.exit_code == 0
    assert json.loads(outcome.stdout)["source"] == "model.gguf"


def test_cli_inspect_bad_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"not a gguf")
    assert runner.invoke(app, ["inspect", str(path)]).exit_code == 2


def test_cli_diff_exit_codes(tmp_path: Path) -> None:
    a = save_receipt(receipt(), tmp_path / "a.json")
    b = save_receipt(receipt(passed=3), tmp_path / "b.json")
    assert runner.invoke(app, ["diff", str(a), str(b)]).exit_code == 0
    assert runner.invoke(app, ["diff", str(a), str(b), "--fail-on-regression"]).exit_code == 1
    assert runner.invoke(app, ["diff", str(a), str(a), "--fail-on-regression"]).exit_code == 0


@pytest.mark.parametrize(
    ("verdict", "code"),
    [(Verdict.GO, 0), (Verdict.CAUTION, 0), (Verdict.NO_GO, 1), (Verdict.INCOMPLETE, 2)],
)
def test_cli_check_exit_codes(
    verdict: Verdict, code: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("gguf_preflight.cli.run_preflight", lambda *a, **k: receipt(verdict))
    out = tmp_path / "r.json"
    outcome = runner.invoke(app, ["check", BASE, "--model", "m", "--json", "--out", str(out)])
    assert outcome.exit_code == code
    assert json.loads(outcome.stdout)["verdict"] == verdict.value
    assert load_receipt(out).verdict is verdict
