"""Protocol client and the base runtime adapter."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx

from gguf_preflight.models import RuntimeInfo

Protocol = Literal["openai", "anthropic"]
DETERMINISTIC_SEED = 42
UNSUPPORTED_STATUS = {404, 405, 501}
MAX_ERROR_CHARS = 400


class EndpointError(ValueError):
    """The endpoint is invalid, not allowed, or not an inference server."""


def _is_local_host(host: str) -> bool:
    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith((".localhost", ".local")):
        return True
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    # Link-local covers cloud metadata addresses such as 169.254.169.254.
    return not address.is_link_local and (address.is_loopback or address.is_private)


def normalize_endpoint(url: str, allow_remote: bool = False) -> str:
    """Return a base URL without `/v1`, refusing public hosts unless explicitly allowed."""
    candidate = url.strip()
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parts = urlsplit(candidate)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise EndpointError(f"Invalid endpoint: {url}")
    if parts.username or parts.password:
        raise EndpointError("Put credentials in GGUF_PREFLIGHT_API_KEY, not in the URL.")
    if not allow_remote and not _is_local_host(parts.hostname):
        raise EndpointError(
            f"{parts.hostname} is not a local address. Pass --allow-remote to send "
            "test prompts to it."
        )
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    name: str
    arguments_raw: str
    arguments: dict[str, Any] | None = None
    parse_error: str | None = None

    @classmethod
    def parse(cls, name: str, raw: Any) -> ToolCall:
        if isinstance(raw, dict):
            return cls(name, json.dumps(raw, sort_keys=True), raw)
        text = raw if isinstance(raw, str) else ""
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            return cls(name, text, None, f"invalid JSON: {exc.msg}")
        if not isinstance(value, dict):
            return cls(name, text, None, "arguments are not a JSON object")
        return cls(name, text, value)


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None


@dataclass
class ChatRequest:
    messages: list[dict[str, str]]
    system: str | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    max_tokens: int = 256
    stream: bool = False


@dataclass
class ChatResult:
    protocol: Protocol
    stream: bool
    status_code: int = 0
    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    ttft_s: float | None = None
    total_s: float = 0.0
    prompt_ms: float | None = None
    prompt_n: int | None = None
    predicted_ms: float | None = None
    predicted_n: int | None = None
    malformed_events: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status_code == 200 and self.error is None

    @property
    def unsupported(self) -> bool:
        return self.status_code in UNSUPPORTED_STATUS

    @property
    def server_failed(self) -> bool:
        """The server crashed or was unreachable, so this says nothing about the wire format."""
        return not self.ok and (self.status_code == 0 or self.status_code >= 500)

    @property
    def empty(self) -> bool:
        return not self.text.strip() and not self.tool_calls and not self.reasoning.strip()

    @property
    def budget_exhausted(self) -> bool:
        return (
            self.finish_reason in {"length", "max_tokens"}
            and not self.text.strip()
            and not self.tool_calls
        )


class Transcript:
    """Canonical model I/O, so identical behavior produces an identical hash."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._redactions: dict[str, str] = {}

    def redact(self, value: str, token: str) -> None:
        self._redactions[value] = token

    def add(self, request: ChatRequest, result: ChatResult) -> None:
        self._entries.append(
            {
                "protocol": result.protocol,
                "stream": request.stream,
                "system": request.system,
                "messages": request.messages,
                "tools": [tool.name for tool in request.tools],
                "max_tokens": request.max_tokens,
                "status": result.status_code,
                "text": result.text,
                # Runtimes can emit JSON keys in any order, so hash parsed arguments.
                "tool_calls": [
                    [c.name, c.arguments_raw if c.arguments is None else c.arguments]
                    for c in result.tool_calls
                ],
                "finish_reason": result.finish_reason,
            }
        )

    def sha256(self) -> str:
        payload = json.dumps(self._entries, sort_keys=True, ensure_ascii=False)
        for value, token in sorted(self._redactions.items(), key=lambda kv: -len(kv[0])):
            payload = payload.replace(value, token)
        return hashlib.sha256(payload.encode()).hexdigest()


def _get_json(http: httpx.Client, url: str) -> Any | None:
    """GET JSON, returning None for anything else (HTML apps on a shared port, errors)."""
    try:
        response = http.get(url)
        if response.status_code != 200:
            return None
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


class RuntimeAdapter:
    name: ClassVar[str] = "generic"

    def __init__(self, http: httpx.Client, base_url: str) -> None:
        self.http = http
        self.base_url = base_url

    @classmethod
    def matches(cls, http: httpx.Client, base_url: str) -> bool:
        data = _get_json(http, f"{base_url}/v1/models")
        return isinstance(data, dict) and isinstance(data.get("data"), list)

    def get_json(self, path: str) -> Any | None:
        return _get_json(self.http, f"{self.base_url}{path}")

    def runtime_info(self, model: str) -> RuntimeInfo:
        return RuntimeInfo(adapter=self.name, endpoint=self.base_url, model=model)

    def request_extras(self, protocol: Protocol) -> dict[str, Any]:
        """Runtime-specific body fields. Used to turn off thinking for short probes."""
        return {}


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _apply_timings(result: ChatResult, timings: Any) -> None:
    if not isinstance(timings, dict):
        return
    result.prompt_ms = _float(timings.get("prompt_ms"))
    result.prompt_n = _int(timings.get("prompt_n"))
    result.predicted_ms = _float(timings.get("predicted_ms"))
    result.predicted_n = _int(timings.get("predicted_n"))
    cache_n = _int(timings.get("cache_n"))
    if cache_n is not None:
        if result.usage.cached_tokens is None:
            result.usage.cached_tokens = cache_n
        if result.usage.prompt_tokens is None and result.prompt_n is not None:
            result.usage.prompt_tokens = result.prompt_n + cache_n


def _apply_openai_usage(result: ChatResult, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    result.usage.prompt_tokens = _int(usage.get("prompt_tokens"))
    result.usage.completion_tokens = _int(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        result.usage.cached_tokens = _int(details.get("cached_tokens"))


class _AnthropicUsage:
    """Anthropic reports uncached, cache-read and cache-write input separately."""

    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def update(self, usage: Any) -> None:
        if isinstance(usage, dict):
            for key, value in usage.items():
                number = _int(value)
                if number is not None:
                    self.values[key] = number

    def apply(self, result: ChatResult) -> None:
        if "input_tokens" in self.values:
            cached = self.values.get("cache_read_input_tokens")
            written = self.values.get("cache_creation_input_tokens", 0)
            result.usage.prompt_tokens = self.values["input_tokens"] + (cached or 0) + written
            result.usage.cached_tokens = cached
        result.usage.completion_tokens = self.values.get("output_tokens")


def _error_text(payload: Any) -> str | None:
    if not isinstance(payload, dict) or "error" not in payload:
        return None
    error = payload["error"]
    if isinstance(error, dict):
        return str(error.get("message") or error)[:MAX_ERROR_CHARS]
    return str(error)[:MAX_ERROR_CHARS]


class LlmClient:
    """Sends deterministic requests over the OpenAI or Anthropic wire format."""

    def __init__(
        self,
        http: httpx.Client,
        base_url: str,
        model: str,
        adapter: RuntimeAdapter,
        transcript: Transcript | None = None,
    ) -> None:
        self.http = http
        self.base_url = base_url
        self.model = model
        self.adapter = adapter
        self.transcript = transcript or Transcript()
        self._supported: dict[Protocol, bool] = {}
        self.server_errors: list[str] = []

    def supports(self, protocol: Protocol) -> bool:
        if protocol not in self._supported:
            result = self.chat(
                ChatRequest(messages=[{"role": "user", "content": "Reply ok."}], max_tokens=1),
                protocol,
            )
            self._supported[protocol] = not result.unsupported or "model" in (
                result.error or ""
            ).lower()
        return self._supported[protocol]

    def _openai_body(self, request: ChatRequest) -> dict[str, Any]:
        messages = list(request.messages)
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1.0,
            "seed": DETERMINISTIC_SEED,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }
        if request.stream:
            body["stream_options"] = {"include_usage": True}
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
        return {**body, **self.adapter.request_extras("openai")}

    def _anthropic_body(self, request: ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": request.messages,
            "temperature": 0,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }
        if request.system:
            body["system"] = request.system
        if request.tools:
            body["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in request.tools
            ]
        return {**body, **self.adapter.request_extras("anthropic")}

    def chat(self, request: ChatRequest, protocol: Protocol = "openai") -> ChatResult:
        if protocol == "openai":
            path, body = "/v1/chat/completions", self._openai_body(request)
        else:
            path, body = "/v1/messages", self._anthropic_body(request)
        result = ChatResult(protocol=protocol, stream=request.stream)
        started = time.perf_counter()
        try:
            if request.stream:
                self._run_stream(path, body, result, started)
            else:
                response = self.http.post(f"{self.base_url}{path}", json=body)
                result.status_code = response.status_code
                result.ttft_s = time.perf_counter() - started
                self._parse_full(response, result)
        except httpx.HTTPError as exc:
            result.error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        result.total_s = time.perf_counter() - started
        if result.server_failed:
            self.server_errors.append(result.error or f"HTTP {result.status_code}")
        self.transcript.add(request, result)
        return result

    def _parse_full(self, response: httpx.Response, result: ChatResult) -> None:
        try:
            payload = response.json()
        except ValueError:
            result.error = f"HTTP {response.status_code}: response was not JSON"
            return
        error = _error_text(payload)
        if error or response.status_code != 200:
            result.error = error or f"HTTP {response.status_code}"
            return
        if result.protocol == "openai":
            self._parse_openai_full(payload, result)
        else:
            self._parse_anthropic_full(payload, result)

    def _parse_openai_full(self, payload: dict[str, Any], result: ChatResult) -> None:
        choices = payload.get("choices") or [{}]
        choice = choices[0]
        message = choice.get("message") or {}
        result.text = message.get("content") or ""
        result.reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        result.finish_reason = choice.get("finish_reason")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            result.tool_calls.append(
                ToolCall.parse(function.get("name") or "", function.get("arguments"))
            )
        _apply_openai_usage(result, payload.get("usage"))
        _apply_timings(result, payload.get("timings"))

    def _parse_anthropic_full(self, payload: dict[str, Any], result: ChatResult) -> None:
        for block in payload.get("content") or []:
            kind = block.get("type")
            if kind == "text":
                result.text += block.get("text") or ""
            elif kind == "thinking":
                result.reasoning += block.get("thinking") or ""
            elif kind == "tool_use":
                result.tool_calls.append(
                    ToolCall.parse(block.get("name") or "", block.get("input"))
                )
        result.finish_reason = payload.get("stop_reason")
        usage = _AnthropicUsage()
        usage.update(payload.get("usage"))
        usage.apply(result)

    def _run_stream(
        self, path: str, body: dict[str, Any], result: ChatResult, started: float
    ) -> None:
        with self.http.stream("POST", f"{self.base_url}{path}", json=body) as response:
            result.status_code = response.status_code
            if response.status_code != 200:
                response.read()
                self._parse_full(response, result)
                return
            events = _sse_events(response.iter_lines(), result)
            if result.protocol == "openai":
                _consume_openai_stream(events, result, started)
            else:
                _consume_anthropic_stream(events, result, started)


def _sse_events(lines: Iterable[str], result: ChatResult) -> Iterable[dict[str, Any]]:
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            result.malformed_events += 1
            continue
        if isinstance(event, dict):
            yield event
        else:
            result.malformed_events += 1


def _mark_first(result: ChatResult, started: float) -> None:
    if result.ttft_s is None:
        result.ttft_s = time.perf_counter() - started


def _consume_openai_stream(
    events: Iterable[dict[str, Any]], result: ChatResult, started: float
) -> None:
    calls: dict[int, dict[str, str]] = {}
    for event in events:
        error = _error_text(event)
        if error:
            result.error = error
            break
        if event.get("usage"):
            _apply_openai_usage(result, event["usage"])
        if event.get("timings"):
            _apply_timings(result, event["timings"])
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                result.finish_reason = choice["finish_reason"]
            content = delta.get("content")
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if content:
                _mark_first(result, started)
                result.text += content
            if reasoning:
                _mark_first(result, started)
                result.reasoning += reasoning
            for position, call in enumerate(delta.get("tool_calls") or []):
                _mark_first(result, started)
                index = call.get("index", position)
                slot = calls.setdefault(index, {"name": "", "arguments": ""})
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                arguments = function.get("arguments")
                if isinstance(arguments, dict):
                    slot["arguments"] += json.dumps(arguments)
                elif arguments:
                    slot["arguments"] += arguments
    for index in sorted(calls):
        result.tool_calls.append(ToolCall.parse(calls[index]["name"], calls[index]["arguments"]))


def _consume_anthropic_stream(
    events: Iterable[dict[str, Any]], result: ChatResult, started: float
) -> None:
    usage = _AnthropicUsage()
    blocks: dict[int, dict[str, str]] = {}
    for event in events:
        kind = event.get("type")
        if kind == "error":
            result.error = _error_text(event) or "stream error"
            break
        if kind == "message_start":
            usage.update((event.get("message") or {}).get("usage"))
        elif kind == "message_delta":
            usage.update(event.get("usage"))
            stop = (event.get("delta") or {}).get("stop_reason")
            if stop:
                result.finish_reason = stop
        elif kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                _mark_first(result, started)
                blocks[event.get("index", len(blocks))] = {
                    "name": block.get("name") or "",
                    "arguments": "",
                }
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                _mark_first(result, started)
                result.text += delta.get("text") or ""
            elif delta_type == "thinking_delta":
                _mark_first(result, started)
                result.reasoning += delta.get("thinking") or ""
            elif delta_type == "input_json_delta":
                _mark_first(result, started)
                slot = blocks.setdefault(event.get("index", 0), {"name": "", "arguments": ""})
                slot["arguments"] += delta.get("partial_json") or ""
    usage.apply(result)
    for index in sorted(blocks):
        raw = blocks[index]["arguments"] or "{}"
        result.tool_calls.append(ToolCall.parse(blocks[index]["name"], raw))


def api_key_headers() -> dict[str, str]:
    """Optional key for servers started with an API key. Never written to receipts."""
    key = os.environ.get("GGUF_PREFLIGHT_API_KEY")
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}", "x-api-key": key}
