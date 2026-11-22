"""Live probes."""

from __future__ import annotations

from gguf_preflight.models import Finding, ProbeStatus, Severity


def status_from_findings(findings: list[Finding]) -> ProbeStatus:
    if any(finding.severity is Severity.FAIL for finding in findings):
        return ProbeStatus.FAIL
    if any(finding.severity is Severity.WARN for finding in findings):
        return ProbeStatus.WARN
    return ProbeStatus.PASS
