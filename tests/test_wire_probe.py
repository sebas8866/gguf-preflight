from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from gguf_preflight.adapters import detect_adapter
from gguf_preflight.adapters.base import (
    ChatRequest,
    EndpointError,
    LlmClient,
    RuntimeAdapter,
    Transcript,
    normalize_endpoint,
)
from gguf_preflight.adapters.llama_server import LlamaServerAdapter
from gguf_preflight.adapters.lmstudio import LmStudioAdapter
from gguf_preflight.adapters.ollama import OllamaAdapter
from gguf_preflight.models import FailureCode, ProbeStatus, RuntimeInfo, Severity
from gguf_preflight.probes.wire_probe import (
    PROBE_TOOL,
    check_agent_context,
    probe_context_canary,
    probe_prefix_cache,
    probe_tool_wire,
)

BASE = "http://127.0.0.1:9999"
FIXTURES = Path(__file__).parent / "fixtures"
Handler = Callable[[httpx.Request], httpx.Response]


def fixture_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def sse(*events: dict[str, Any]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def client(adapter: type[RuntimeAdapter] = RuntimeAdapter) -> LlmClient:
    http = httpx.Client()
    return LlmClient(http, BASE, "test-model", adapter(http, BASE))


def openai_message(content: str = "", tool_calls: list[Any] | None = None, **usage: int) -> Any:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 2},
    }


GOOD_TOOL = {"function": {"name": "record_value", "arguments": '{"value":"alpha-7"}'}}


def anthropic_missing() -> None:
    respx.post(f"{BASE}/v1/messages").mock(return_value=httpx.Response(404, text="not found"))


def cache_stream(cached: Callable[[int], int]) -> Handler:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not body.get("stream"):
            return httpx.Response(200, json=openai_message("ok"))
        turn = calls["n"]
        calls["n"] += 1
        return sse(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 2000,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": cached(turn)},
                },
            },
        )

    return handler


@respx.mock
def test_prefix_cache_retained_passes() -> None:
    anthropic_missing()
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=cache_stream(lambda turn: 0 if turn == 0 else 1950)
    )
    result = probe_prefix_cache(client(), "nonce")
    assert result.status is ProbeStatus.PASS
    assert result.metrics["openai"]["warm_reuse_ratio"] == 0.975


@respx.mock
def test_prefix_cache_invalidated_every_turn() -> None:
    anthropic_missing()
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=cache_stream(lambda turn: 4))
    result = probe_prefix_cache(client(), "nonce")
    assert result.status is ProbeStatus.FAIL
    assert result.findings[0].code is FailureCode.PREFIX_CACHE_MISS
    assert result.findings[0].check == "prefix_cache_invalidated"


@respx.mock
def test_anthropic_invalidation_uses_cache_read_tokens() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=cache_stream(lambda t: 1950))

    def anthropic(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            )
        return sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 2000}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 1998, "cache_read_input_tokens": 2, "output_tokens": 1},
            },
        )

    respx.post(f"{BASE}/v1/messages").mock(side_effect=anthropic)
    result = probe_prefix_cache(client(), "nonce")
    assert result.status is ProbeStatus.FAIL
    assert [f.evidence["protocol"] for f in result.findings] == ["anthropic"]


def canary_handler(keep: bool, prompt_tokens: Callable[[int], int]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        chars = sum(len(message["content"]) for message in body["messages"])
        tokens = prompt_tokens(chars)
        text = "ok"
        if system.startswith("The security code is "):
            code = system.split()[4].rstrip(".")
            text = code if keep else "I do not see a code."
        return httpx.Response(
            200, json=openai_message(text, prompt_tokens=tokens, completion_tokens=3)
        )

    return handler


def runtime(context: int | None = None) -> RuntimeInfo:
    return RuntimeInfo(adapter="generic", endpoint=BASE, model="m", context_length=context)


@respx.mock
def test_canary_survives() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=canary_handler(True, lambda chars: chars // 3)
    )
    result = probe_context_canary(client(), runtime(), "n", (4000, 8000))
    assert result.status is ProbeStatus.PASS
    assert {row["outcome"] for row in result.metrics["depths"].values()} == {"retained"}


@respx.mock
def test_silent_truncation_detected_from_usage() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=canary_handler(False, lambda chars: min(chars // 3, 3000))
    )
    result = probe_context_canary(client(), runtime(), "n", (4000, 8000))
    assert result.status is ProbeStatus.FAIL
    finding = result.findings[0]
    assert finding.check == "silent_context_truncation"
    assert finding.code is FailureCode.WIRE_BUG


@respx.mock
def test_truncation_detected_from_runtime_context() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=canary_handler(False, lambda chars: chars // 3)
    )
    result = probe_context_canary(client(), runtime(context=4096), "n", (8000,))
    assert result.findings[0].check == "silent_context_truncation"


@respx.mock
def test_recall_failure_is_not_called_truncation() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=canary_handler(False, lambda chars: chars // 3)
    )
    result = probe_context_canary(client(), runtime(context=32000), "n", (4000,))
    assert result.findings[0].check == "canary_recall_failed"
    assert result.findings[0].severity is Severity.WARN


@respx.mock
def test_explicit_context_error_is_not_silent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if len(body["messages"][0]["content"]) > 5000:
            return httpx.Response(400, json={"error": {"message": "request exceeds context size"}})
        return httpx.Response(200, json=openai_message("ok"))

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=handler)
    result = probe_context_canary(client(), runtime(), "n", (8000,))
    assert result.findings[0].check == "context_limit_explicit"


def tool_handler(full: Any, stream_events: list[dict[str, Any]]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools") and body.get("stream"):
            return sse(*stream_events)
        if body.get("tools"):
            return httpx.Response(200, json=full)
        return httpx.Response(200, json=openai_message("ok"))

    return handler


GOOD_STREAM = [
    {"choices": [{"delta": {"tool_calls": [{"index": 0, **GOOD_TOOL}]}, "finish_reason": None}]},
    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
]


@respx.mock
def test_tool_wire_clean() -> None:
    anthropic_missing()
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=tool_handler(openai_message("", [GOOD_TOOL]), GOOD_STREAM)
    )
    result = probe_tool_wire(client())
    assert result.status is ProbeStatus.PASS
    assert result.metrics["modes"] == {"openai/non-stream": "ok", "openai/stream": "ok"}


@respx.mock
def test_split_stream_arguments_reassembled() -> None:
    anthropic_missing()
    split = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "record_value", "arguments": '{"va'}}
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'lue":"alpha-7"}'}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=tool_handler(openai_message("", [GOOD_TOOL]), split)
    )
    assert probe_tool_wire(client()).status is ProbeStatus.PASS


@respx.mock
def test_silent_empty_stream_drop() -> None:
    anthropic_missing()
    empty = [
        {"choices": [{"delta": {"content": ""}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=tool_handler(openai_message("", [GOOD_TOOL]), empty)
    )
    result = probe_tool_wire(client())
    assert result.status is ProbeStatus.FAIL
    assert result.metrics["modes"]["openai/stream"] == "tool_call_empty_response"
    assert result.metrics["modes"]["openai/non-stream"] == "ok"


@respx.mock
@pytest.mark.parametrize(
    "markup",
    [
        '<tool_call>{"name":"record_value","arguments":{"value":"alpha-7"}}</tool_call>',
        '[TOOL_CALLS] [{"name":"record_value"}]',
        '<function=record_value>{"value":"alpha-7"}</function>',
    ],
)
def test_plaintext_tool_leak(markup: str) -> None:
    anthropic_missing()
    stream = [{"choices": [{"delta": {"content": markup}, "finish_reason": "stop"}]}]
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=tool_handler(openai_message(markup), stream)
    )
    result = probe_tool_wire(client())
    assert {f.check for f in result.findings} == {"tool_call_plaintext_leak"}


@respx.mock
def test_invalid_json_arguments() -> None:
    anthropic_missing()
    bad = {"function": {"name": "record_value", "arguments": '{"value": alpha'}}
    stream = [{"choices": [{"delta": {"tool_calls": [{"index": 0, **bad}]}}]}]
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=tool_handler(openai_message("", [bad]), stream)
    )
    result = probe_tool_wire(client())
    assert {f.check for f in result.findings} == {"tool_args_invalid_json"}


@respx.mock
def test_thinking_budget_exhaustion_is_not_a_wire_bug() -> None:
    anthropic_missing()
    full = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "", "reasoning": "still thinking"},
                "finish_reason": "length",
            }
        ]
    }
    stream = [
        {"choices": [{"delta": {"reasoning": "still thinking"}, "finish_reason": "length"}]},
    ]
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=tool_handler(full, stream))
    result = probe_tool_wire(client())
    assert {f.check for f in result.findings} == {"tool_call_budget_exhausted"}
    assert all(f.code is None for f in result.findings)
    assert result.status is ProbeStatus.WARN


@respx.mock
def test_error_event_inside_http_200_stream() -> None:
    anthropic_missing()
    stream = [{"error": {"message": "slot unavailable"}}]
    respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=tool_handler(openai_message("", [GOOD_TOOL]), stream)
    )
    result = probe_tool_wire(client())
    assert result.metrics["modes"]["openai/stream"] == "tool_call_http_error"


@respx.mock
def test_malformed_sse_is_flagged() -> None:
    anthropic_missing()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools") and body.get("stream"):
            good = json.dumps(GOOD_STREAM[0])
            return httpx.Response(200, text=f"data: {{broken\n\ndata: {good}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json=openai_message("", [GOOD_TOOL]))

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=handler)
    result = probe_tool_wire(client())
    assert result.metrics["modes"]["openai/stream"] == "tool_stream_malformed"


@respx.mock
def test_real_ollama_openai_tool_response() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=fixture_json("ollama_oai_tool.json"))
    )
    result = client().chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"city": "Paris"}
    assert result.reasoning


@respx.mock
def test_real_ollama_openai_stream() -> None:
    text = (FIXTURES / "ollama_oai_tool_stream.sse").read_text(encoding="utf-8")
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=httpx.Response(200, text=text))
    result = client().chat(ChatRequest(messages=[{"role": "user", "content": "x"}], stream=True))
    assert result.tool_calls[0].arguments == {"city": "Paris"}
    assert result.finish_reason == "tool_calls"
    assert result.usage.prompt_tokens == 149
    assert result.usage.cached_tokens == 148


@respx.mock
def test_real_ollama_anthropic_response() -> None:
    respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(200, json=fixture_json("ollama_anth_tool.json"))
    )
    result = client().chat(ChatRequest(messages=[{"role": "user", "content": "x"}]), "anthropic")
    assert result.tool_calls[0].arguments == {"city": "Paris"}
    assert result.usage.prompt_tokens == 149
    assert result.usage.cached_tokens == 148


@respx.mock
def test_real_ollama_anthropic_stream() -> None:
    text = (FIXTURES / "ollama_anth_tool_stream.sse").read_text(encoding="utf-8")
    respx.post(f"{BASE}/v1/messages").mock(return_value=httpx.Response(200, text=text))
    result = client().chat(
        ChatRequest(messages=[{"role": "user", "content": "x"}], stream=True), "anthropic"
    )
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"city": "Paris"}
    assert result.usage.prompt_tokens == 149
    assert result.usage.cached_tokens == 148
    assert result.finish_reason == "tool_use"


@respx.mock
def test_llama_server_timings_supply_cache_counts() -> None:
    payload = openai_message("ok")
    payload.pop("usage")
    payload["timings"] = {
        "cache_n": 236,
        "prompt_n": 4,
        "prompt_ms": 12.5,
        "predicted_n": 8,
        "predicted_ms": 40.0,
    }
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=httpx.Response(200, json=payload))
    result = client().chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
    assert result.usage.cached_tokens == 236
    assert result.usage.prompt_tokens == 240
    assert result.prompt_ms == 12.5


def _html(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="<!doctype html><html></html>")


@respx.mock
def test_detect_rejects_html_app_on_shared_port() -> None:
    respx.route(host="127.0.0.1").mock(side_effect=_html)
    with httpx.Client() as http, pytest.raises(EndpointError, match="No inference server"):
        detect_adapter(http, BASE)


@respx.mock
def test_detect_ollama() -> None:
    respx.get(f"{BASE}/api/version").mock(
        return_value=httpx.Response(200, json={"version": "0.40.12"})
    )
    with httpx.Client() as http:
        assert isinstance(detect_adapter(http, BASE), OllamaAdapter)


@respx.mock
def test_detect_llama_server() -> None:
    respx.get(f"{BASE}/api/version").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/props").mock(
        return_value=httpx.Response(
            200, json={"default_generation_settings": {"n_ctx": 8192}, "build_info": "b1"}
        )
    )
    with httpx.Client() as http:
        adapter = detect_adapter(http, BASE)
        assert isinstance(adapter, LlamaServerAdapter)
        assert adapter.runtime_info("m").context_length == 8192


@respx.mock
def test_detect_lmstudio_v1() -> None:
    respx.get(f"{BASE}/api/version").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    payload = {
        "models": [
            {
                "key": "m",
                "max_context_length": 131072,
                "quantization": {"name": "Q4_K_M"},
                "loaded_instances": [{"id": "m", "config": {"context_length": 4096}}],
            }
        ]
    }
    respx.get(f"{BASE}/api/v1/models").mock(return_value=httpx.Response(200, json=payload))
    with httpx.Client() as http:
        adapter = detect_adapter(http, BASE)
        assert isinstance(adapter, LmStudioAdapter)
        info = adapter.runtime_info("m")
    assert info.context_length == 4096
    assert info.details["model_max_context"] == 131072


@respx.mock
def test_ollama_runtime_info_from_real_ps() -> None:
    respx.get(f"{BASE}/api/version").mock(
        return_value=httpx.Response(200, json={"version": "0.40.12"})
    )
    respx.post(f"{BASE}/api/show").mock(
        return_value=httpx.Response(
            200,
            json={
                "model_info": {"qwen3.context_length": 40960},
                "capabilities": ["completion", "tools"],
            },
        )
    )
    respx.get(f"{BASE}/api/ps").mock(
        return_value=httpx.Response(200, json=fixture_json("ollama_ps.json"))
    )
    with httpx.Client() as http:
        info = OllamaAdapter(http, BASE).runtime_info("qwen3:8b")
    assert info.context_length == 4096
    assert info.details["model_max_context"] == 40960
    assert info.details["gpu_offload_ratio"] == 1.0


def test_agent_context_gate() -> None:
    result = check_agent_context(runtime(4096), "claude-code")
    assert result.status is ProbeStatus.FAIL
    assert result.findings[0].check == "context_below_agent_minimum"
    assert check_agent_context(runtime(32768), "claude-code").status is ProbeStatus.WARN
    assert check_agent_context(runtime(64000), "claude-code").status is ProbeStatus.PASS
    assert check_agent_context(runtime(4096), "generic").status is ProbeStatus.SKIP


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("localhost:11434", "http://localhost:11434"),
        ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080"),
        ("http://192.168.1.20:1234/v1/", "http://192.168.1.20:1234"),
        ("http://gpu-box.local:8000", "http://gpu-box.local:8000"),
    ],
)
def test_normalize_endpoint(url: str, expected: str) -> None:
    assert normalize_endpoint(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "http://8.8.8.8:80",
        "http://169.254.169.254/latest",
        "ftp://localhost",
        "http://user:pass@localhost:1",
    ],
)
def test_endpoint_rejects_non_local(url: str) -> None:
    with pytest.raises(EndpointError):
        normalize_endpoint(url)


def test_allow_remote_opt_in() -> None:
    assert normalize_endpoint("https://api.example.com", True) == "https://api.example.com"


@respx.mock
def test_transcript_hash_stable_across_nonces() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=openai_message("ok"))
    )

    def run(nonce: str) -> str:
        transcript = Transcript()
        http = httpx.Client()
        llm = LlmClient(http, BASE, "m", RuntimeAdapter(http, BASE), transcript)
        transcript.redact(nonce, "<NONCE>")
        llm.chat(ChatRequest(system=f"run {nonce}", messages=[{"role": "user", "content": "x"}]))
        return transcript.sha256()

    assert run("aaaa1111") == run("bbbb2222")


def test_probe_tool_schema_requires_value() -> None:
    assert PROBE_TOOL.parameters["required"] == ["value"]
