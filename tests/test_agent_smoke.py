from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

import httpx
import pytest
import respx

from gguf_preflight.adapters.base import (
    ChatResult,
    LlmClient,
    RuntimeAdapter,
    ToolCall,
    Usage,
)
from gguf_preflight.inspector.hardware import detect_hardware, parse_nvidia_smi
from gguf_preflight.models import FailureCode, ProbeStatus, Severity
from gguf_preflight.probes.agent_smoke import (
    DEFAULT_SUITES,
    TaskOutcome,
    ToolDefinition,
    ToolExpectation,
    ToolTask,
    evaluate_suite,
    evaluate_tool_task,
    extract_code,
    load_suite,
    run_python,
    run_suite,
    validate_schema,
)
from gguf_preflight.probes.speed_probe import (
    generation_tps,
    prefill_tps,
    run_speed_probe,
    summarize,
)

BASE = "http://127.0.0.1:9999"

# Known-good solutions, so the hidden tests are proven solvable.
SOLUTIONS = {
    "code_01_two_sum": """
def two_sum(nums, target):
    seen = {}
    for j, value in enumerate(nums):
        need = target - value
        if need in seen:
            return (seen[need], j)
        seen.setdefault(value, j)
    return None
""",
    "code_02_balanced": """
def is_balanced(text):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in text:
        if ch in '([{':
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
""",
    "code_03_roman": """
def roman_to_int(s):
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    for i, ch in enumerate(s):
        v = values[ch]
        total += -v if i + 1 < len(s) and v < values[s[i + 1]] else v
    return total
""",
    "code_04_merge_intervals": """
def merge_intervals(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
""",
    "code_05_flatten": """
def flatten(items):
    out = []
    for item in items:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
""",
    "code_06_top_words": """
import re
from collections import Counter
def top_words(text, k):
    counts = Counter(re.findall(r'[a-z]+', text.lower()))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
""",
    "code_07_rle": """
import re
from itertools import groupby
def rle_encode(s):
    return ''.join(f'{len(list(g))}{c}' for c, g in groupby(s))
def rle_decode(s):
    return ''.join(c * int(n) for n, c in re.findall(r'(\\d+)(\\D)', s))
""",
    "code_08_lower_bound": """
def first_index(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(nums) and nums[lo] == target else -1
""",
    "code_09_lru": """
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()
    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]
    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
""",
    "code_10_parse_query": """
def parse_pairs(text):
    out = {}
    for segment in text.split('&'):
        if not segment:
            continue
        key, _, value = segment.partition('=')
        out.setdefault(key, []).append(value)
    return out
""",
}


def llm() -> LlmClient:
    http = httpx.Client(timeout=5)
    return LlmClient(http, BASE, "test-model", RuntimeAdapter(http, BASE))


def reply(text: str = "", tools: list[Any] | None = None) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tools:
        message["tool_calls"] = tools
    return httpx.Response(
        200,
        json={
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        },
    )


def test_suite_shapes() -> None:
    assert [len(load_suite(name).tasks) for name in DEFAULT_SUITES] == [10, 8, 4]


def test_task_ids_are_unique() -> None:
    ids = [task["id"] for name in DEFAULT_SUITES for task in load_suite(name).tasks]
    assert len(ids) == len(set(ids)) == 22


def test_probe_logic_has_no_hardcoded_model_names() -> None:
    from pathlib import Path

    root = Path(__file__).parents[1] / "src" / "gguf_preflight" / "probes"
    pattern = re.compile(r"qwen|llama-?3|mistral|gemma|deepseek|phi-?\d", re.IGNORECASE)
    for path in root.glob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name


@pytest.mark.parametrize("task", load_suite("core_code_10").code_tasks(), ids=lambda t: t.id)
def test_reference_solution_passes(task: Any) -> None:
    assert run_python(SOLUTIONS[task.id], task.tests, timeout_s=15).passed


@pytest.mark.parametrize("task", load_suite("core_code_10").code_tasks(), ids=lambda t: t.id)
def test_stub_solution_fails(task: Any) -> None:
    names = re.findall(r"^(?:def|class) (\w+)", SOLUTIONS[task.id], re.MULTILINE)
    stub = "\n".join(f"def {name}(*args, **kwargs):\n    return None" for name in names)
    assert not run_python(stub, task.tests, timeout_s=15).passed


def test_execution_failure_reason() -> None:
    result = run_python("def f():\n    return 1\n", "assert f() == 2", 10)
    assert not result.passed
    assert "AssertionError" in result.reason


def test_infinite_loop_times_out_quickly() -> None:
    started = time.perf_counter()
    result = run_python("while True:\n    pass\n", "assert True", 1.0)
    assert not result.passed
    assert "timed out" in result.reason
    assert time.perf_counter() - started < 8


@pytest.mark.parametrize(
    "code",
    [
        "import sys\nsys.exit(0)",
        "import os\nos._exit(0)",
        "raise SystemExit(0)",
    ],
)
def test_early_exit_cannot_fake_a_pass(code: str) -> None:
    assert not run_python(code, "assert True", 10).passed


@pytest.mark.parametrize(
    "code",
    [
        "import socket\nsocket.create_connection(('127.0.0.1', 9))",
        "import subprocess\nsubprocess.run(['whoami'])",
        "import os\nos.system('echo unsafe')",
        "open('written.txt', 'w').write('x')",
        "import os\nos.remove('runner.py')",
    ],
)
def test_guard_blocks_side_effects(code: str) -> None:
    result = run_python(code, "assert True", 10)
    assert not result.passed
    assert "blocked by preflight" in result.reason


def test_guard_does_not_leak_parent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREFLIGHT_SECRET_TEST", "secret")
    code = "import os\nvalue = os.environ.get('PREFLIGHT_SECRET_TEST')"
    assert run_python(code, "assert value is None", 10).passed


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("```python\ndef f():\n    return 1\n```", "def f"),
        ("Here:\n```py\nclass A:\n    pass\n```\nDone", "class A"),
        ("```\ndef g():\n    pass\n```", "def g"),
        ("def raw():\n    return 2", "def raw"),
        ("<think>```python\ndef wrong(): pass\n```</think>```python\ndef right(): pass\n```", "right"),
        ("```python\nprint('demo')\n```\n```python\ndef real(): pass\n```", "real"),
    ],
)
def test_extract_code(text: str, needle: str) -> None:
    code = extract_code(text)
    assert code is not None and needle in code


def test_extract_code_none_for_prose() -> None:
    assert extract_code("I cannot help with that.") is None


SCHEMA = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "minimum": 1, "maximum": 3},
        "flag": {"type": "boolean"},
        "kind": {"type": "string", "enum": ["a", "b"]},
        "items": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["count"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ({"count": 2}, True),
        ({"count": 2, "flag": True, "kind": "a", "items": ["x"]}, True),
        ({}, False),
        ({"count": "2"}, False),
        ({"count": True}, False),
        ({"count": 9}, False),
        ({"count": 1, "kind": "c"}, False),
        ({"count": 1, "items": []}, False),
        ({"count": 1, "items": [1]}, False),
        ({"count": 1, "extra": 1}, False),
    ],
)
def test_validate_schema(value: dict[str, Any], valid: bool) -> None:
    assert (not validate_schema(value, SCHEMA)) is valid


def tool_task(forbidden: list[str] | None = None, tool: str | None = "good") -> ToolTask:
    return ToolTask(
        id="t",
        prompt="p",
        tools=[
            ToolDefinition(
                name="good",
                description="d",
                parameters={
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                },
            ),
            ToolDefinition(name="bad", description="d", parameters={"type": "object"}),
        ],
        expect=ToolExpectation(tool=tool, args={"n": 3} if tool else {}),
        forbidden=forbidden or [],
    )


def tool_result(*calls: ToolCall, text: str = "") -> ChatResult:
    return ChatResult("openai", False, 200, text=text, tool_calls=list(calls))


@pytest.mark.parametrize(
    ("calls", "passed", "reason"),
    [
        ([ToolCall.parse("good", '{"n":3}')], True, "passed"),
        ([ToolCall.parse("good", '{"n":4}')], False, "values"),
        ([ToolCall.parse("good", '{"n":"3"}')], False, "expected integer"),
        ([ToolCall.parse("good", "{bad")], False, "bad arguments"),
        ([ToolCall.parse("bad", "{}")], False, "forbidden"),
        ([], False, "expected one"),
        ([ToolCall.parse("good", '{"n":3}'), ToolCall.parse("good", '{"n":3}')], False, "one tool"),
    ],
)
def test_tool_evaluation(calls: list[ToolCall], passed: bool, reason: str) -> None:
    ok, why = evaluate_tool_task(tool_task(["bad"]), tool_result(*calls))
    assert ok is passed
    assert reason in why


def test_no_tool_expected() -> None:
    task = tool_task(tool=None)
    assert evaluate_tool_task(task, tool_result(text="def"))[0]
    assert not evaluate_tool_task(task, tool_result(ToolCall.parse("good", '{"n":1}')))[0]


@respx.mock
def test_code_suite_all_pass() -> None:
    suite = load_suite("core_code_10")
    tasks = suite.code_tasks()
    by_prompt = {task.prompt: SOLUTIONS[task.id] for task in tasks}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["temperature"] == 0 and body["top_p"] == 1.0 and body["seed"] == 42
        return reply(f"```python\n{by_prompt[body['messages'][-1]['content']]}\n```")

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=handler)
    result = run_suite(llm(), suite)
    assert result.status is ProbeStatus.PASS
    assert result.metrics["passed"] == 10


@respx.mock
def test_fluent_but_broken_code_is_quant_collapse() -> None:
    suite = load_suite("core_code_10")
    suite.tasks = suite.tasks[:3]
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=reply("```python\ndef helper():\n    return 'looks fine'\n```")
    )
    result = run_suite(llm(), suite)
    assert result.status is ProbeStatus.FAIL
    assert result.findings[0].code is FailureCode.QUANT_COLLAPSE
    assert result.findings[0].check == "smoke_code_collapse"


@respx.mock
def test_transport_failure_is_wire_bug_not_collapse() -> None:
    suite = load_suite("core_code_10")
    suite.tasks = suite.tasks[:2]
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=reply(""))
    result = run_suite(llm(), suite)
    assert [finding.code for finding in result.findings] == [FailureCode.WIRE_BUG]


@respx.mock
def test_context_suite_uses_full_history() -> None:
    suite = load_suite("core_context_4")
    suite.tasks = [suite.tasks[3]]
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        sizes.append(len(messages))
        return reply("85" if len(sizes) == 4 else "OK")

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=handler)
    result = run_suite(llm(), suite)
    assert result.status is ProbeStatus.PASS
    assert sizes == [2, 4, 6, 8]


@respx.mock
def test_context_constraint_loss_detected() -> None:
    suite = load_suite("core_context_4")
    suite.tasks = [suite.tasks[2]]
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=reply("requests"))
    result = run_suite(llm(), suite)
    assert result.metrics["passed"] == 0
    assert result.findings[0].check == "smoke_context_collapse"


def test_warn_and_fail_thresholds() -> None:
    suite = load_suite("core_code_10")
    def outcomes(passed: int) -> list[TaskOutcome]:
        return [TaskOutcome(str(i), i < passed, "x") for i in range(10)]
    assert evaluate_suite(suite, outcomes(8)) == []
    borderline = evaluate_suite(suite, outcomes(7))[0]
    assert borderline.severity is Severity.WARN
    assert borderline.code is None
    assert borderline.check == "smoke_code_low"
    collapsed = evaluate_suite(suite, outcomes(5))[0]
    assert collapsed.severity is Severity.FAIL
    assert collapsed.code is FailureCode.QUANT_COLLAPSE


def test_summary_median_and_iqr() -> None:
    assert summarize([1.0, 2.0, 9.0]) == {"median": 2.0, "iqr": 4.0, "n": 3}
    assert summarize([5.0]) == {"median": 5.0, "iqr": 0.0, "n": 1}
    assert summarize([]) == {"median": None, "iqr": None, "n": 0}


def test_tps_prefers_server_timings() -> None:
    result = ChatResult("openai", True, 200, ttft_s=1.0, total_s=3.0)
    result.prompt_ms, result.prompt_n, result.predicted_ms, result.predicted_n = 500, 1000, 2000, 40
    assert prefill_tps(result) == 2000
    assert generation_tps(result) == 20


def test_tps_fallback_excludes_cached_tokens() -> None:
    result = ChatResult(
        "openai",
        True,
        200,
        ttft_s=2.0,
        total_s=4.0,
        usage=Usage(prompt_tokens=1000, completion_tokens=41, cached_tokens=600),
    )
    assert prefill_tps(result) == 200
    assert generation_tps(result) == 20


@respx.mock
def test_speed_probe_reports_all_metrics() -> None:
    respx.post(f"{BASE}/v1/messages").mock(return_value=httpx.Response(404))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not body.get("stream"):
            return reply("ok")
        events = [
            {"choices": [{"delta": {"content": "1 2 3"}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "timings": {"prompt_n": 100, "prompt_ms": 100, "predicted_n": 40, "predicted_ms": 1000},
                "usage": {"prompt_tokens": 2000, "completion_tokens": 40},
            },
        ]
        text = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=text)

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=handler)
    result = run_speed_probe(llm(), depths=(2000, 8000))
    assert set(result.metrics["depths"]) == {"2000", "8000"}
    row = result.metrics["depths"]["8000"]
    assert row["ttft_s"]["n"] == 3
    assert row["generation_tps"]["median"] == 40
    assert row["cold_prefill_tps"] == 1000


@respx.mock
def test_slow_stack_flags_latency_bottleneck() -> None:
    respx.post(f"{BASE}/v1/messages").mock(return_value=httpx.Response(404))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not body.get("stream"):
            return reply("ok")
        events = [
            {"choices": [{"delta": {"content": "1"}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "timings": {"prompt_n": 100, "prompt_ms": 2000, "predicted_n": 8, "predicted_ms": 4000},
                "usage": {"prompt_tokens": 100, "completion_tokens": 8},
            },
        ]
        text = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=text)

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=handler)
    result = run_speed_probe(llm(), depths=(2000,), iterations=3)
    assert {finding.check for finding in result.findings} == {"slow_prefill", "slow_generation"}
    assert all(finding.code is FailureCode.LATENCY_BOTTLENECK for finding in result.findings)


def test_parse_nvidia_smi() -> None:
    gpus = parse_nvidia_smi("NVIDIA GeForce RTX 5060, 8151, 4746\n")
    assert gpus[0].name == "NVIDIA GeForce RTX 5060"
    assert gpus[0].vram_total_mb == 8151
    assert gpus[0].vram_used_mb == 4746


def test_detect_hardware_runs_on_this_platform() -> None:
    info = detect_hardware()
    assert info.os
    assert info.logical_cpus
    if sys.platform in {"win32", "linux"}:
        assert info.ram_total_gb and info.ram_total_gb > 0
