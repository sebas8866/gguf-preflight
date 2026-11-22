"""FastMCP stdio server.

An agent can call these tools without a person reviewing each argument, so this surface
is stricter than the CLI: endpoints must be local and remote GGUFs must be on Hugging Face.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlsplit

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from gguf_preflight.adapters.base import EndpointError
from gguf_preflight.inspector.gguf_header import GgufError, _is_hf_host
from gguf_preflight.inspector.gguf_header import inspect_gguf as read_gguf
from gguf_preflight.registry.receipt import ReceiptError, diff_receipts, load_receipt
from gguf_preflight.runner import run_preflight

TARGET_AGENTS = {"claude-code", "cline", "generic"}

mcp: FastMCP = FastMCP(
    "gguf-preflight",
    instructions=(
        "Checks a local GGUF inference stack before an AI coding agent uses it. "
        "preflight_check sends test prompts to a local server and runs the model's "
        "Python answers in a restricted subprocess."
    ),
)


def _check_gguf_ref(ref: str) -> None:
    if "://" not in ref:
        return
    parts = urlsplit(ref)
    if parts.scheme != "https" or not _is_hf_host(parts.hostname):
        raise ToolError("Remote GGUF references must be https Hugging Face URLs.")


@mcp.tool
async def preflight_check(
    endpoint: str,
    model: str,
    gguf_ref: str | None = None,
    target_agent: str = "claude-code",
) -> dict[str, Any]:
    """Run the full preflight against a local server and return the receipt."""
    if target_agent not in TARGET_AGENTS:
        raise ToolError(f"target_agent must be one of: {', '.join(sorted(TARGET_AGENTS))}")
    if gguf_ref:
        _check_gguf_ref(gguf_ref)
    try:
        receipt = await asyncio.to_thread(
            run_preflight, endpoint, model, gguf_ref=gguf_ref, agent=target_agent
        )
    except EndpointError as exc:
        raise ToolError(str(exc)) from exc
    return receipt.model_dump(mode="json")


@mcp.tool
async def inspect_gguf(path_or_url: str) -> dict[str, Any]:
    """Read GGUF metadata from a local file or Hugging Face URL without the weights."""
    _check_gguf_ref(path_or_url)
    try:
        metadata = await asyncio.to_thread(read_gguf, path_or_url)
    except (GgufError, OSError) as exc:
        raise ToolError(f"Could not read GGUF: {type(exc).__name__}") from exc
    return metadata.model_dump(mode="json")


@mcp.tool
async def compare_receipts(receipt_a_path: str, receipt_b_path: str) -> dict[str, Any]:
    """Compare two saved receipts and list changes and regressions."""
    try:
        a = await asyncio.to_thread(load_receipt, receipt_a_path)
        b = await asyncio.to_thread(load_receipt, receipt_b_path)
    except ReceiptError as exc:
        raise ToolError(str(exc)) from exc
    return diff_receipts(a, b).model_dump(mode="json")


def serve() -> None:
    # stdout carries the MCP protocol, so the startup banner must stay off.
    mcp.run(transport="stdio", show_banner=False)
