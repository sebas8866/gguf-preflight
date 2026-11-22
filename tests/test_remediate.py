from __future__ import annotations

import pytest

from gguf_preflight.models import FailureCode, Finding, Severity
from gguf_preflight.remediate.rules import ALIASES, ANY, RULES, recommend


def finding(check: str, severity: Severity = Severity.FAIL) -> Finding:
    return Finding(check=check, severity=severity, message=check, code=FailureCode.WIRE_BUG)


def all_commands(check: str, adapter: str, agent: str = "generic") -> str:
    return "\n".join(c for fix in recommend([finding(check)], adapter, agent) for c in fix.commands)


def test_ollama_context_fix() -> None:
    assert "OLLAMA_CONTEXT_LENGTH" in all_commands("silent_context_truncation", "ollama")


def test_llama_server_context_fix() -> None:
    assert "-c 64000" in all_commands("silent_context_truncation", "llama_server")


def test_lmstudio_context_fix() -> None:
    assert "--context-length" in all_commands("context_below_agent_minimum", "lmstudio")


def test_llama_server_kv_cache_fix() -> None:
    commands = all_commands("prefix_cache_invalidated", "llama_server")
    assert "-ctk f16 -ctv f16" in commands
    assert "--cache-reuse" in commands


def test_ollama_kv_cache_fix() -> None:
    assert "OLLAMA_KV_CACHE_TYPE" in all_commands("prefix_cache_invalidated", "ollama")


def test_claude_code_reminder_only_for_claude_code() -> None:
    assert "CLAUDE_CODE_TOTAL_TOKENS_REMINDER" in all_commands(
        "prefix_cache_invalidated", "ollama", "claude-code"
    )
    assert "CLAUDE_CODE_ATTRIBUTION_HEADER" in all_commands(
        "prefix_cache_invalidated", "ollama", "claude-code"
    )
    assert "CLAUDE_CODE" not in all_commands("prefix_cache_invalidated", "ollama", "cline")


def test_commands_cover_powershell_and_posix() -> None:
    commands = all_commands("silent_context_truncation", "ollama")
    assert "$env:OLLAMA_CONTEXT_LENGTH" in commands
    assert "OLLAMA_CONTEXT_LENGTH=64000 ollama serve" in commands


def test_jinja_fix_for_tool_leak() -> None:
    assert "--jinja" in all_commands("tool_call_plaintext_leak", "llama_server")


def test_imatrix_fix() -> None:
    assert "--imatrix" in all_commands("uncalibrated_low_bit_quant", "ollama")


def test_deduplicates_shared_fixes() -> None:
    fixes = recommend(
        [finding("silent_context_truncation"), finding("context_below_agent_minimum")], "ollama"
    )
    assert len(fixes) == 1


def test_fail_ranks_before_warn() -> None:
    fixes = recommend(
        [
            finding("uncalibrated_low_bit_quant", Severity.WARN),
            finding("silent_context_truncation", Severity.FAIL),
        ],
        "ollama",
    )
    assert fixes[0].check == "silent_context_truncation"


def test_info_findings_get_no_fix() -> None:
    assert recommend([finding("missing_tokenizer", Severity.INFO)], "ollama") == []


def test_unknown_check_is_ignored() -> None:
    assert recommend([finding("does_not_exist")], "ollama") == []


@pytest.mark.parametrize("check", sorted(RULES))
def test_every_rule_has_a_fallback_for_unknown_adapters(check: str) -> None:
    assert ANY in RULES[check]


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_aliases_resolve(alias: str) -> None:
    assert ALIASES[alias] in RULES
