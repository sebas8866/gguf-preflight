"""Ollama runtime adapter."""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from gguf_preflight.adapters.base import Protocol, RuntimeAdapter, _get_json
from gguf_preflight.models import RuntimeInfo


class OllamaAdapter(RuntimeAdapter):
    name: ClassVar[str] = "ollama"

    @classmethod
    def matches(cls, http: httpx.Client, base_url: str) -> bool:
        data = _get_json(http, f"{base_url}/api/version")
        return isinstance(data, dict) and isinstance(data.get("version"), str)

    def request_extras(self, protocol: Protocol) -> dict[str, Any]:
        if protocol == "openai":
            return {"reasoning_effort": "none"}
        return {"thinking": {"type": "disabled"}}

    def _show(self, model: str) -> dict[str, Any]:
        try:
            response = self.http.post(f"{self.base_url}/api/show", json={"model": model})
            payload = response.json() if response.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def runtime_info(self, model: str) -> RuntimeInfo:
        version = self.get_json("/api/version") or {}
        details: dict[str, Any] = {}
        context_length: int | None = None

        show = self._show(model)
        model_info = show.get("model_info")
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                if key.endswith(".context_length") and isinstance(value, int):
                    details["model_max_context"] = value
        if isinstance(show.get("capabilities"), list):
            details["capabilities"] = show["capabilities"]

        running = self.get_json("/api/ps") or {}
        for entry in running.get("models") or []:
            # Ollama lists an untagged name with its implicit :latest tag.
            names = {entry.get("name"), entry.get("model")}
            if model in names or (":" not in model and f"{model}:latest" in names):
                value = entry.get("context_length")
                context_length = value if isinstance(value, int) else None
                size, vram = entry.get("size"), entry.get("size_vram")
                if isinstance(size, int) and isinstance(vram, int) and size > 0:
                    details["gpu_offload_ratio"] = round(vram / size, 3)
                entry_details = entry.get("details") or {}
                details["quantization"] = entry_details.get("quantization_level")
                details["digest"] = entry.get("digest")
                break

        return RuntimeInfo(
            adapter=self.name,
            endpoint=self.base_url,
            model=model,
            version=version.get("version"),
            context_length=context_length,
            details=details,
        )
