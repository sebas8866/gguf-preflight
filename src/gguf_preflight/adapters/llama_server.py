"""llama.cpp llama-server runtime adapter."""

from __future__ import annotations

from pathlib import PurePath
from typing import Any, ClassVar

import httpx

from gguf_preflight.adapters.base import Protocol, RuntimeAdapter, _get_json
from gguf_preflight.models import RuntimeInfo


class LlamaServerAdapter(RuntimeAdapter):
    name: ClassVar[str] = "llama_server"

    @classmethod
    def matches(cls, http: httpx.Client, base_url: str) -> bool:
        data = _get_json(http, f"{base_url}/props")
        return isinstance(data, dict) and (
            "default_generation_settings" in data or "build_info" in data
        )

    def request_extras(self, protocol: Protocol) -> dict[str, Any]:
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def runtime_info(self, model: str) -> RuntimeInfo:
        props = self.get_json("/props") or {}
        settings = props.get("default_generation_settings") or {}
        context = settings.get("n_ctx")
        details: dict[str, Any] = {
            "total_slots": props.get("total_slots"),
            "has_chat_template": bool(props.get("chat_template")),
        }
        model_path = props.get("model_path")
        if isinstance(model_path, str) and model_path:
            # Basename only: receipts should not expose local directory names.
            details["model_file"] = PurePath(model_path.replace("\\", "/")).name
        return RuntimeInfo(
            adapter=self.name,
            endpoint=self.base_url,
            model=model,
            version=props.get("build_info"),
            context_length=context if isinstance(context, int) else None,
            details=details,
        )
