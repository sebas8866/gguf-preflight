"""Deterministic agent smoke suite driven by declarative YAML task packs.

Model-written code runs in a separate, time-limited Python process with a blocking
audit hook. This reduces risk but is not a security sandbox.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from gguf_preflight.adapters.base import ChatRequest, ChatResult, LlmClient, Protocol, ToolSpec
from gguf_preflight.models import FailureCode, Finding, ProbeResult, ProbeStatus, Severity
from gguf_preflight.probes import status_from_findings

DEFAULT_SUITES = ("core_code_10", "core_tools_8", "core_context_4")
Progress = Callable[[str, bool], None]


class _SuiteModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolDefinition(_SuiteModel):
    name: str
    description: str
    parameters: dict[str, Any]

    def to_spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class ToolExpectation(_SuiteModel):
    tool: str | None
    args: dict[str, Any] = Field(default_factory=dict)
    text_matches: str | None = None


class CodeTask(_SuiteModel):
    id: str
    prompt: str
    tests: str


class ToolTask(_SuiteModel):
    id: str
    prompt: str
    tools: list[ToolDefinition]
    expect: ToolExpectation
    forbidden: list[str] = Field(default_factory=list)


class ContextTask(_SuiteModel):
    id: str
    turns: list[str] = Field(min_length=2)
    must_match: str
    must_not_match: str | None = None


class Suite(_SuiteModel):
    id: str
    kind: Literal["code", "tools", "context"]
    version: int
    description: str
    system: str
    max_tokens: int
    warn_below: float
    fail_below: float
    timeout_s: float = 5.0
    tasks: list[dict[str, Any]]

    def code_tasks(self) -> list[CodeTask]:
        return [CodeTask.model_validate(task) for task in self.tasks]

    def tool_tasks(self) -> list[ToolTask]:
        return [ToolTask.model_validate(task) for task in self.tasks]

    def context_tasks(self) -> list[ContextTask]:
        return [ContextTask.model_validate(task) for task in self.tasks]


def _suite_text(name: str) -> str:
    filename = f"{name}.yaml"
    packaged = resources.files("gguf_preflight").joinpath("suites", filename)
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    # Running from a source checkout, where suites/ sits at the repo root.
    source = Path(__file__).resolve().parents[3] / "suites" / filename
    if source.is_file():
        return source.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Suite not found: {name}")


def load_suite(name: str) -> Suite:
    suite = Suite.model_validate(yaml.safe_load(_suite_text(name)))
    # Parse tasks now so a bad pack fails before any model call.
    {"code": suite.code_tasks, "tools": suite.tool_tasks, "context": suite.context_tasks}[
        suite.kind
    ]()
    return suite


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """Pick the first Python fence that defines something, then any fence, then raw code."""
    cleaned = _THINK.sub("", text).strip()
    blocks: list[tuple[str, str]] = [
        (str(lang).lower(), str(body).strip()) for lang, body in _FENCE.findall(cleaned)
    ]
    python = [body for lang, body in blocks if lang in {"python", "py", "python3"}]
    candidates = python or [body for lang, body in blocks if not lang]
    for body in candidates:
        if re.search(r"^\s*(def|class)\s", body, re.MULTILINE):
            return body
    if candidates:
        return candidates[0]
    if re.search(r"^\s*(def|class)\s", cleaned, re.MULTILINE):
        return cleaned
    return None


_RUNNER = r"""
import json, sys
token, result_path = sys.argv[1], sys.argv[2]
result = open(result_path, "w", encoding="utf-8")
candidate = open("candidate.py", encoding="utf-8").read()
tests = open("checks.py", encoding="utf-8").read()
BLOCKED = ("socket.", "subprocess.", "os.system", "os.exec", "os.fork", "os.forkpty",
           "os.spawn", "os.posix_spawn", "os.startfile", "os.remove", "os.rename", "os.rmdir",
           "os.mkdir", "os.link", "os.symlink", "os.truncate", "os.chmod", "os.chown",
           "shutil.", "ctypes.", "winreg.", "webbrowser.", "ftplib.", "smtplib.", "urllib.")
def guard(event, args):
    if event.startswith(BLOCKED):
        raise PermissionError("blocked by preflight: " + event)
    if event == "open":
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(flags & 3)
        if writing:
            raise PermissionError("blocked by preflight: file write")
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
except Exception:
    pass
sys.addaudithook(guard)
outcome = {"token": token, "passed": False, "error": None}
try:
    scope = {"__name__": "candidate"}
    exec(compile(candidate, "candidate.py", "exec"), scope)
    exec(compile(tests, "checks.py", "exec"), scope)
    outcome["passed"] = True
except BaseException as exc:
    outcome["error"] = (type(exc).__name__ + ": " + str(exc))[:300]
result.write(json.dumps(outcome))
result.flush()
"""


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    reason: str


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        import signal

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _child_env(workdir: str) -> dict[str, str]:
    env = {"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8", "TEMP": workdir, "TMP": workdir}
    if sys.platform == "win32":
        env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
    return env


def run_python(code: str, tests: str, timeout_s: float = 5.0) -> ExecutionResult:
    """Run candidate code plus assertions in an isolated interpreter."""
    token = secrets.token_hex(16)
    with tempfile.TemporaryDirectory(prefix="gguf-preflight-") as workdir:
        root = Path(workdir)
        (root / "runner.py").write_text(_RUNNER, encoding="utf-8")
        (root / "candidate.py").write_text(code, encoding="utf-8")
        (root / "checks.py").write_text(tests, encoding="utf-8")
        result_path = root / "result.json"
        command = [sys.executable, "-I", "-S", "-B", "runner.py", token, str(result_path)]
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(
            command,
            cwd=workdir,
            env=_child_env(workdir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(process)
            return ExecutionResult(False, f"timed out after {timeout_s:g}s")
        try:
            outcome = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ExecutionResult(False, f"process exited with code {process.returncode}")
        if outcome.get("token") != token:
            return ExecutionResult(False, "result token mismatch")
        if outcome.get("passed") is True:
            return ExecutionResult(True, "passed")
        return ExecutionResult(False, str(outcome.get("error") or "failed"))


_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "object": dict,
    "array": list,
    "boolean": bool,
    "number": (int, float),
    "integer": int,
}


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate the JSON-schema subset used by tool definitions."""
    errors: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str):
        allowed = _TYPES.get(expected)
        wrong_bool = isinstance(value, bool) and expected in {"number", "integer"}
        if allowed is not None and (not isinstance(value, allowed) or wrong_bool):
            return [f"{path}: expected {expected}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not an allowed value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}.{key}: required")
        for key, item in value.items():
            if key in properties:
                errors.extend(validate_schema(item, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: unexpected")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: too few items")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                errors.extend(validate_schema(item, schema["items"], f"{path}[{index}]"))
    return errors


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) >= set(expected)
            and all(_values_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_values_equal(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return bool(actual == expected)


def evaluate_tool_task(task: ToolTask, result: ChatResult) -> tuple[bool, str]:
    names = [call.name for call in result.tool_calls]
    forbidden = sorted(set(names) & set(task.forbidden))
    if forbidden:
        return False, f"called forbidden tool: {', '.join(forbidden)}"
    expect = task.expect
    if expect.tool is None:
        if names:
            return False, f"called a tool when none was needed: {names[0]}"
        if expect.text_matches and not re.search(expect.text_matches, result.text):
            return False, "answer did not match"
        return True, "passed"
    if len(names) != 1:
        return False, f"expected one tool call, got {len(names)}"
    call = result.tool_calls[0]
    if call.name != expect.tool:
        return False, f"wrong tool: {call.name}"
    if call.parse_error or call.arguments is None:
        return False, f"bad arguments: {call.parse_error}"
    definition = next(tool for tool in task.tools if tool.name == expect.tool)
    errors = validate_schema(call.arguments, definition.parameters)
    if errors:
        return False, "; ".join(errors[:3])
    if not _values_equal(call.arguments, expect.args):
        return False, "argument values did not match"
    return True, "passed"


@dataclass
class TaskOutcome:
    task_id: str
    passed: bool
    reason: str
    infrastructure: bool = False
    fluent: bool = False
    server_error: bool = False


def _request_failure(task_id: str, result: ChatResult) -> TaskOutcome | None:
    if result.server_failed:
        return TaskOutcome(task_id, False, f"server failed: {result.error}", server_error=True)
    if not result.ok:
        return TaskOutcome(task_id, False, f"request failed: {result.error}", infrastructure=True)
    if result.empty:
        return TaskOutcome(task_id, False, "HTTP 200 with an empty response", infrastructure=True)
    if result.budget_exhausted:
        return TaskOutcome(task_id, False, "token limit reached before an answer")
    return None


def _run_code(client: LlmClient, suite: Suite, protocol: Protocol, progress: Progress | None) -> list[TaskOutcome]:
    outcomes: list[TaskOutcome] = []
    for task in suite.code_tasks():
        result = client.chat(
            ChatRequest(
                system=suite.system,
                messages=[{"role": "user", "content": task.prompt}],
                max_tokens=suite.max_tokens,
            ),
            protocol,
        )
        outcome = _request_failure(task.id, result)
        if outcome is None:
            code = extract_code(result.text)
            if code is None:
                outcome = TaskOutcome(task.id, False, "no code in response", fluent=True)
            else:
                run = run_python(code, task.tests, suite.timeout_s)
                outcome = TaskOutcome(task.id, run.passed, run.reason, fluent=not run.passed)
        outcomes.append(outcome)
        if progress:
            progress(task.id, outcome.passed)
    return outcomes


def _run_tools(client: LlmClient, suite: Suite, protocol: Protocol, progress: Progress | None) -> list[TaskOutcome]:
    outcomes: list[TaskOutcome] = []
    for task in suite.tool_tasks():
        result = client.chat(
            ChatRequest(
                system=suite.system,
                messages=[{"role": "user", "content": task.prompt}],
                tools=[tool.to_spec() for tool in task.tools],
                max_tokens=suite.max_tokens,
            ),
            protocol,
        )
        outcome = _request_failure(task.id, result)
        if outcome is None:
            passed, reason = evaluate_tool_task(task, result)
            outcome = TaskOutcome(task.id, passed, reason)
        outcomes.append(outcome)
        if progress:
            progress(task.id, outcome.passed)
    return outcomes


def _run_context(client: LlmClient, suite: Suite, protocol: Protocol, progress: Progress | None) -> list[TaskOutcome]:
    outcomes: list[TaskOutcome] = []
    for task in suite.context_tasks():
        messages: list[dict[str, str]] = []
        outcome: TaskOutcome | None = None
        result: ChatResult | None = None
        for turn in task.turns:
            messages.append({"role": "user", "content": turn})
            result = client.chat(
                ChatRequest(
                    system=suite.system,
                    messages=list(messages),
                    max_tokens=suite.max_tokens,
                ),
                protocol,
            )
            outcome = _request_failure(task.id, result)
            if outcome is not None:
                break
            messages.append({"role": "assistant", "content": result.text})
        if outcome is None and result is not None:
            text = _THINK.sub("", result.text).strip()
            if not re.search(task.must_match, text):
                outcome = TaskOutcome(task.id, False, "turn-1 constraint was lost")
            elif task.must_not_match and re.search(task.must_not_match, text):
                outcome = TaskOutcome(task.id, False, "reply broke a turn-1 prohibition")
            else:
                outcome = TaskOutcome(task.id, True, "passed")
        assert outcome is not None
        outcomes.append(outcome)
        if progress:
            progress(task.id, outcome.passed)
    return outcomes


def evaluate_suite(suite: Suite, outcomes: list[TaskOutcome]) -> list[Finding]:
    """Separate transport failures from model failures before classifying."""
    findings: list[Finding] = []
    infrastructure = [outcome for outcome in outcomes if outcome.infrastructure]
    # A crashed server tells us nothing about the model, so those tasks are not scored.
    scored = [
        outcome for outcome in outcomes if not outcome.infrastructure and not outcome.server_error
    ]
    if infrastructure:
        findings.append(
            Finding(
                check="tool_call_empty_response"
                if any("empty" in outcome.reason for outcome in infrastructure)
                else "tool_call_http_error",
                severity=Severity.FAIL,
                code=FailureCode.WIRE_BUG,
                message=f"{suite.id}: {len(infrastructure)} task(s) failed at the transport level.",
                evidence={"tasks": [outcome.task_id for outcome in infrastructure]},
            )
        )
    if not scored:
        return findings
    rate = sum(outcome.passed for outcome in scored) / len(scored)
    if rate >= suite.warn_below:
        return findings
    failed = [outcome for outcome in scored if not outcome.passed]
    evidence = {
        "pass_rate": round(rate, 3),
        "failed": {outcome.task_id: outcome.reason[:120] for outcome in failed},
    }
    if rate < suite.fail_below:
        findings.append(
            Finding(
                check=f"smoke_{suite.kind}_collapse",
                severity=Severity.FAIL,
                code=FailureCode.QUANT_COLLAPSE,
                message=(
                    f"{suite.id}: {rate:.0%} passed. This can mean a damaged quant, or a model "
                    "too small for these tasks."
                ),
                evidence=evidence,
            )
        )
    else:
        # A healthy quant can land here, so a borderline score must not blame the quant.
        findings.append(
            Finding(
                check=f"smoke_{suite.kind}_low",
                severity=Severity.WARN,
                message=f"{suite.id}: {rate:.0%} passed, below the {suite.warn_below:.0%} target.",
                evidence=evidence,
            )
        )
    return findings


def run_suite(
    client: LlmClient,
    suite: Suite,
    protocol: Protocol = "openai",
    progress: Progress | None = None,
) -> ProbeResult:
    started = time.perf_counter()
    runner = {"code": _run_code, "tools": _run_tools, "context": _run_context}[suite.kind]
    outcomes = runner(client, suite, protocol, progress)
    findings = evaluate_suite(suite, outcomes)
    passed = sum(outcome.passed for outcome in outcomes)
    status = status_from_findings(findings)
    if status is not ProbeStatus.FAIL and any(outcome.server_error for outcome in outcomes):
        status = ProbeStatus.ERROR
    return ProbeResult(
        name=suite.id,
        status=status,
        duration_s=round(time.perf_counter() - started, 3),
        summary=f"{passed}/{len(outcomes)} tasks passed.",
        metrics={
            "passed": passed,
            "total": len(outcomes),
            "suite_version": suite.version,
            "tasks": {
                outcome.task_id: "passed" if outcome.passed else outcome.reason[:120]
                for outcome in outcomes
            },
        },
        findings=findings,
    )


def run_agent_smoke(
    client: LlmClient,
    suites: tuple[str, ...] = DEFAULT_SUITES,
    progress: Progress | None = None,
) -> list[ProbeResult]:
    protocol: Protocol = "openai" if client.supports("openai") else "anthropic"
    if not client.supports(protocol):
        return [
            ProbeResult(name=name, status=ProbeStatus.SKIP, summary="No supported chat protocol.")
            for name in suites
        ]
    return [run_suite(client, load_suite(name), protocol, progress) for name in suites]
