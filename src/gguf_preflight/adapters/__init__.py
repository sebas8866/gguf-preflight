"""Runtime adapters and detection."""

from __future__ import annotations

import httpx

from gguf_preflight.adapters.base import EndpointError, RuntimeAdapter
from gguf_preflight.adapters.llama_server import LlamaServerAdapter
from gguf_preflight.adapters.lmstudio import LmStudioAdapter
from gguf_preflight.adapters.ollama import OllamaAdapter

ADAPTERS: tuple[type[RuntimeAdapter], ...] = (
    OllamaAdapter,
    LlamaServerAdapter,
    LmStudioAdapter,
    RuntimeAdapter,
)


def detect_adapter(http: httpx.Client, base_url: str) -> RuntimeAdapter:
    """Identify the runtime from JSON APIs, so other apps on a shared port are rejected."""
    for adapter in ADAPTERS:
        if adapter.matches(http, base_url):
            return adapter(http, base_url)
    raise EndpointError(
        f"No inference server found at {base_url}. Expected Ollama, llama-server, "
        "LM Studio, or an OpenAI-compatible /v1/models API."
    )
