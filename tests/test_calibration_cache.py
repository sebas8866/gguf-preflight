from __future__ import annotations

import json
import re

import httpx
import respx

from gguf_preflight.adapters.base import LlmClient, RuntimeAdapter
from gguf_preflight.models import RuntimeInfo
from gguf_preflight.probes.wire_probe import probe_context_canary

BASE = "http://127.0.0.1:9999"


class WarmServer:
    """Reports only the tokens it had to process, reusing the longest prefix it has seen."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = "\n".join(str(message["content"]) for message in body["messages"])
        tokens = text.split()
        reused = max((self._common(tokens, old) for old in self.seen), default=0)
        self.seen.append(tokens)
        canary = re.search(r"CANARY-[0-9A-F]+", text)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": canary.group(0) if canary else "ok",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": len(tokens) - reused, "completion_tokens": 1},
            },
        )

    @staticmethod
    def _common(left: list[str], right: list[str]) -> int:
        count = 0
        for first, second in zip(left, right, strict=False):
            if first != second:
                break
            count += 1
        return count


def canary_prompt_size(server: WarmServer, nonce: str) -> int:
    """The real size of the prompt sent for the depth test."""
    http = httpx.Client()
    llm = LlmClient(http, BASE, "test-model", RuntimeAdapter(http, BASE))
    runtime = RuntimeInfo(adapter="test", endpoint=BASE, model="test-model")
    probe_context_canary(llm, runtime, nonce, (4000,))
    return len(server.seen[-1])


@respx.mock
def test_warm_server_does_not_change_calibration() -> None:
    server = WarmServer()
    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=server)
    cold = canary_prompt_size(server, "run-one")
    warm = canary_prompt_size(server, "run-two")
    assert abs(warm - cold) <= cold * 0.02
