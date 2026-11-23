from __future__ import annotations

from gguf_preflight.adapters.base import ChatResult
from gguf_preflight.models import FailureCode, Finding, ProbeResult, ProbeStatus, Severity
from gguf_preflight.probes.wire_probe import TEMPLATE_DISABLED_CHECK, classify_tool_result
from gguf_preflight.registry.receipt import all_findings, decide_verdict, failure_codes
from gguf_preflight.remediate.rules import recommend
from gguf_preflight.runner import _discount_template_confounded, _runtime_health

JINJA_ERROR = "tools param requires --jinja flag"


def _collapse() -> Finding:
    return Finding(
        check="smoke_code_collapse",
        severity=Severity.FAIL,
        code=FailureCode.QUANT_COLLAPSE,
        message="core_code_10: 0% passed.",
    )


def _disabled() -> Finding:
    return Finding(
        check=TEMPLATE_DISABLED_CHECK,
        severity=Severity.FAIL,
        code=FailureCode.WIRE_BUG,
        message="started without --jinja",
    )


def test_jinja_500_is_a_specific_failure() -> None:
    result = ChatResult(protocol="openai", stream=False, status_code=500, error=JINJA_ERROR)
    finding = classify_tool_result(result, "openai/non-stream")
    assert finding is not None
    assert finding.check == TEMPLATE_DISABLED_CHECK
    assert finding.severity is Severity.FAIL
    assert finding.code is FailureCode.WIRE_BUG


def test_other_500_stays_unverified() -> None:
    result = ChatResult(protocol="openai", stream=False, status_code=500, error="boom")
    finding = classify_tool_result(result, "openai/non-stream")
    assert finding is not None
    assert finding.check == "tool_probe_server_failed"


def test_jinja_fault_removes_quant_blame_and_keeps_no_go() -> None:
    results = _discount_template_confounded(
        [
            ProbeResult(name="tool_wire", status=ProbeStatus.FAIL, findings=[_disabled()]),
            ProbeResult(
                name="core_code_10",
                status=ProbeStatus.FAIL,
                summary="0/10 tasks passed.",
                findings=[_collapse()],
            ),
        ]
    )
    assert decide_verdict(results).value == "NO_GO"
    assert failure_codes(results) == [FailureCode.WIRE_BUG]
    assert results[1].status is ProbeStatus.SKIP
    assert results[1].findings[0].evidence["confounded_by"] == TEMPLATE_DISABLED_CHECK
    fixes = recommend(all_findings(results), "llama_server")
    assert [fix.title for fix in fixes] == ["Restart llama-server with --jinja"]
    assert "--jinja" in fixes[0].commands[0]


def test_collapse_without_jinja_fault_is_unchanged() -> None:
    original = [ProbeResult(name="core_code_10", status=ProbeStatus.FAIL, findings=[_collapse()])]
    assert _discount_template_confounded(original) == original


def test_jinja_errors_are_not_runtime_crashes() -> None:
    assert _runtime_health([JINJA_ERROR, JINJA_ERROR]) is None
    health = _runtime_health([JINJA_ERROR, "HTTP 500"])
    assert health is not None
    assert health.findings[0].evidence["failed_requests"] == 1
