"""LM Studio runtime adapter (native REST v1, with a v0 fallback)."""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from gguf_preflight.adapters.base import RuntimeAdapter, _get_json
from gguf_preflight.models import RuntimeInfo


def _v1_entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [entry for entry in data["models"] if isinstance(entry, dict)]
    return []


def _v0_entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [
            entry
            for entry in data["data"]
            if isinstance(entry, dict) and "max_context_length" in entry
        ]
    return []


class LmStudioAdapter(RuntimeAdapter):
    name: ClassVar[str] = "lmstudio"

    @classmethod
    def matches(cls, http: httpx.Client, base_url: str) -> bool:
        return bool(
            _v1_entries(_get_json(http, f"{base_url}/api/v1/models"))
            or _v0_entries(_get_json(http, f"{base_url}/api/v0/models"))
        )

    def runtime_info(self, model: str) -> RuntimeInfo:
        context: int | None = None
        details: dict[str, Any] = {}
        for entry in _v1_entries(self.get_json("/api/v1/models")):
            instances = [i for i in entry.get("loaded_instances") or [] if isinstance(i, dict)]
            names = {entry.get("key"), *(instance.get("id") for instance in instances)}
            if model not in names:
                continue
            details["model_max_context"] = entry.get("max_context_length")
            details["quantization"] = (entry.get("quantization") or {}).get("name")
            if instances:
                value = (instances[0].get("config") or {}).get("context_length")
                context = value if isinstance(value, int) else None
            break
        else:
            for entry in _v0_entries(self.get_json("/api/v0/models")):
                if entry.get("id") == model:
                    details["model_max_context"] = entry.get("max_context_length")
                    details["quantization"] = entry.get("quantization")
                    value = entry.get("loaded_context_length")
                    context = value if isinstance(value, int) else None
                    break
        return RuntimeInfo(
            adapter=self.name,
            endpoint=self.base_url,
            model=model,
            context_length=context,
            details=details,
        )
