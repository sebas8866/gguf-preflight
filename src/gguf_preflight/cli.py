"""Command line interface."""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from gguf_preflight.adapters.base import EndpointError
from gguf_preflight.inspector.gguf_header import GgufError, inspect_gguf
from gguf_preflight.models import GgufMetadata, PreflightReceipt, ProbeStatus, Verdict
from gguf_preflight.registry.receipt import (
    ReceiptDiff,
    ReceiptError,
    diff_receipts,
    load_receipt,
    save_receipt,
)
from gguf_preflight.runner import run_preflight

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Deterministic preflight for local GGUF stacks, before you connect a coding agent.",
)
EXIT_CODES = {Verdict.GO: 0, Verdict.CAUTION: 0, Verdict.NO_GO: 1, Verdict.INCOMPLETE: 2}
_VERDICT_STYLE = {
    Verdict.GO: "bold green",
    Verdict.CAUTION: "bold yellow",
    Verdict.NO_GO: "bold red",
    Verdict.INCOMPLETE: "bold magenta",
}
_STATUS_STYLE = {
    ProbeStatus.PASS: "green",
    ProbeStatus.WARN: "yellow",
    ProbeStatus.FAIL: "red",
    ProbeStatus.SKIP: "dim",
    ProbeStatus.ERROR: "magenta",
}


class Agent(StrEnum):
    claude_code = "claude-code"
    cline = "cline"
    generic = "generic"


def _print_json(data: Any) -> None:
    # ASCII-only JSON survives legacy Windows code pages and shell redirection.
    sys.stdout.write(json.dumps(data, indent=2) + "\n")


def _fail(message: str, code: int = 2) -> typer.Exit:
    Console(stderr=True).print(f"[bold red]Error:[/] {message}", markup=True, highlight=False)
    return typer.Exit(code)


def _fmt(value: Any, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _render_speed(console: Console, receipt: PreflightReceipt) -> None:
    speed = next((r for r in receipt.probe_results if r.name == "speed"), None)
    depths = speed.metrics.get("depths") if speed else None
    skipped = speed.metrics.get("skipped_depths") if speed else None
    if skipped:
        listed = ", ".join(str(depth) for depth in skipped)
        console.print(f"Speed not measured at depth {listed}: above loaded context.", highlight=False)
    if not depths:
        return
    table = Table(title="Speed (warm median +/- IQR)")
    columns = ("Depth", "TTFT s", "Cached prompt tok/s", "Gen tok/s", "Cold prefill tok/s", "Speedup")
    for column in columns:
        table.add_column(column, justify="right")
    for depth, row in sorted(depths.items(), key=lambda item: int(item[0])):
        ttft, prompt, generation = row["ttft_s"], row["prompt_tps"], row["generation_tps"]
        table.add_row(
            depth,
            f"{_fmt(ttft['median'], 2)} +/- {_fmt(ttft['iqr'], 2)}",
            f"{_fmt(prompt['median'], 0)} +/- {_fmt(prompt['iqr'], 0)}",
            f"{_fmt(generation['median'])} +/- {_fmt(generation['iqr'])}",
            _fmt(row.get("cold_prefill_tps"), 0),
            f"{_fmt(row.get('prefill_speedup'))}x",
        )
    console.print(table)


def _render_receipt(console: Console, receipt: PreflightReceipt) -> None:
    runtime = receipt.runtime_info
    style = _VERDICT_STYLE[receipt.verdict]
    console.print(f"[{style}]{receipt.verdict.value}[/]  {runtime.model} on {runtime.adapter}")
    context = runtime.context_length if runtime.context_length is not None else "unknown"
    console.print(
        f"Agent: {receipt.target_agent}   Context: {context}   Run: {receipt.duration_s:.0f}s",
        highlight=False,
    )
    table = Table(title="Probes")
    table.add_column("Probe")
    table.add_column("Status")
    table.add_column("Result", overflow="fold")
    for result in receipt.probe_results:
        status_style = _STATUS_STYLE[result.status]
        table.add_row(
            result.name, f"[{status_style}]{result.status.value}[/]", result.summary
        )
    console.print(table)
    _render_speed(console, receipt)
    if receipt.failure_codes:
        console.print("Failure codes: " + ", ".join(code.value for code in receipt.failure_codes))
    for index, fix in enumerate(receipt.recommended_fixes, start=1):
        console.print(f"\n[bold]Fix {index}: {fix.title}[/]", highlight=False)
        console.print(fix.why, highlight=False, markup=False)
        for command in fix.commands:
            console.print(f"  {command}", markup=False, highlight=False, soft_wrap=True)


def _render_metadata(console: Console, metadata: GgufMetadata) -> None:
    table = Table(title=metadata.source, show_header=False)
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", overflow="fold")
    rows: list[tuple[str, Any]] = [
        ("GGUF version", metadata.gguf_version),
        ("Architecture", metadata.architecture),
        ("Name", metadata.model_name),
        ("Blocks", metadata.block_count),
        ("Parameters", f"{metadata.parameter_count:,}"),
        ("Tensors", metadata.tensor_count),
        ("Quantization", metadata.quantization),
        ("Bits per weight", metadata.effective_bits_per_weight),
        ("Tokenizer", metadata.tokenizer_model),
        ("Pre-tokenizer", metadata.tokenizer_pre),
        ("Chat template", "yes" if metadata.has_chat_template else "no"),
        ("imatrix", "yes" if metadata.imatrix_keys else "no"),
        ("Layout hash", metadata.tensor_sha256),
    ]
    for key, value in rows:
        table.add_row(key, "-" if value is None else str(value))
    console.print(table)
    for finding in metadata.findings:
        console.print(f"{finding.severity.value}: {finding.message}", markup=False)


def _render_diff(console: Console, diff: ReceiptDiff) -> None:
    if not diff.changes:
        console.print("No differences.")
        return
    table = Table(title=f"Receipt diff ({diff.regressions} regression(s))")
    table.add_column("Field", overflow="fold")
    table.add_column("A", overflow="fold")
    table.add_column("B", overflow="fold")
    table.add_column("Regression")
    for entry in diff.changes:
        table.add_row(
            entry.field, str(entry.a), str(entry.b), "[red]yes[/]" if entry.regression else ""
        )
    console.print(table)


@app.command()
def check(
    endpoint: Annotated[str, typer.Argument(help="Server URL, e.g. http://localhost:11434")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model name on the server.")],
    gguf: Annotated[
        str | None, typer.Option("--gguf", help="GGUF path or Hugging Face URL to inspect.")
    ] = None,
    agent: Annotated[Agent, typer.Option("--agent", help="Agent you plan to connect.")] = (
        Agent.generic
    ),
    as_json: Annotated[bool, typer.Option("--json", help="Print the receipt as JSON.")] = False,
    out: Annotated[Path | None, typer.Option("--out", help="Save the receipt here.")] = None,
    deep: Annotated[bool, typer.Option("--deep", help="Also test 16k depth (slower).")] = False,
    allow_remote: Annotated[
        bool, typer.Option("--allow-remote", help="Allow sending test prompts to a public host.")
    ] = False,
) -> None:
    """Run the full preflight. Exit code: 0 GO or CAUTION, 1 NO_GO, 2 INCOMPLETE or error."""
    progress = Console(stderr=True)
    try:
        with progress.status("Starting preflight") as spinner:
            receipt = run_preflight(
                endpoint,
                model,
                gguf_ref=gguf,
                agent=agent.value,
                deep=deep,
                allow_remote=allow_remote,
                status=lambda message: spinner.update(message),
            )
    except EndpointError as exc:
        raise _fail(str(exc)) from exc
    if out is not None:
        try:
            save_receipt(receipt, out)
        except OSError as exc:
            raise _fail(f"Could not save receipt: {exc}") from exc
    if as_json:
        _print_json(receipt.model_dump(mode="json"))
    else:
        _render_receipt(Console(), receipt)
        if out is not None:
            Console().print(f"\nReceipt saved: {out}", markup=False, highlight=False)
    raise typer.Exit(EXIT_CODES[receipt.verdict])


@app.command("inspect")
def inspect_command(
    source: Annotated[str, typer.Argument(help="Local GGUF path or Hugging Face URL.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Read GGUF metadata without downloading the weights."""
    try:
        metadata = inspect_gguf(source)
    except (GgufError, OSError) as exc:
        raise _fail(str(exc)) from exc
    if as_json:
        _print_json(metadata.model_dump(mode="json"))
    else:
        _render_metadata(Console(), metadata)


@app.command("diff")
def diff_command(
    receipt_a: Annotated[Path, typer.Argument(help="Baseline receipt.")],
    receipt_b: Annotated[Path, typer.Argument(help="New receipt.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
    fail_on_regression: Annotated[
        bool, typer.Option("--fail-on-regression", help="Exit 1 if B regressed from A.")
    ] = False,
) -> None:
    """Compare two receipts."""
    try:
        result = diff_receipts(load_receipt(receipt_a), load_receipt(receipt_b))
    except ReceiptError as exc:
        raise _fail(str(exc)) from exc
    if as_json:
        _print_json(result.model_dump(mode="json"))
    else:
        _render_diff(Console(), result)
    if fail_on_regression and result.regressions:
        raise typer.Exit(1)


@app.command("mcp")
def mcp_command() -> None:
    """Start the MCP server on stdio."""
    from gguf_preflight.mcp_server import serve

    serve()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
