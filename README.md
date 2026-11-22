# gguf-preflight

Check a local GGUF model stack before you point an AI coding agent at it.

A local model can look healthy in a chat window and still break an agent. The server may silently cut off your context, drop tool calls, miss the prompt cache on every turn, or run a quant that writes fluent, broken code. `gguf-preflight` tests for those failures, gives a verdict, and prints the exact fix for your runtime.

- Works with **Ollama**, **llama-server**, **LM Studio**, and other local OpenAI- or Anthropic-compatible servers.
- Ships as a **CLI**, a **Python library**, and an **MCP server**.
- No account, no telemetry, no GPU kernel code. MIT licensed.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and a running local server.

```powershell
uvx gguf-preflight check http://localhost:11434 --model qwen3:8b --agent claude-code
```

Inspect any GGUF without downloading its weights:

```powershell
uvx gguf-preflight inspect https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf
```

## What it checks

| Layer | Check | Failure code |
|---|---|---|
| GGUF header | Missing tokenizer, pre-tokenizer, or chat template. Low-bit quant with no importance matrix | `METADATA_DEFECT` |
| Wire | Silent context truncation at 4k and 8k (16k with `--deep`) | `WIRE_BUG` |
| Wire | Tool calls in streaming and non-streaming mode: empty drops, plaintext leaks, invalid JSON | `WIRE_BUG` |
| Cache | Whether an identical prefix is reused on the next request | `PREFIX_CACHE_MISS` |
| Agent smoke | 10 coding tasks run against hidden tests | `QUANT_COLLAPSE` |
| Agent smoke | 8 strict tool tasks: right tool, types, arguments, forbidden tools | `QUANT_COLLAPSE` |
| Agent smoke | 4 multi-turn tasks: does a turn 1 rule survive to turn 4 | `QUANT_COLLAPSE` |
| Speed | TTFT, prompt tok/s, generation tok/s, cold vs warm prefill (median and IQR) | `LATENCY_BOTTLENECK` |

`QUANT_COLLAPSE` means the stack answers fluently but scores below a suite's failure threshold. That can be a damaged quant. It can also be a model that is simply too small for the tasks. Borderline scores are warnings and do not blame the quant.

## Verdicts and exit codes

| Verdict | Meaning | Exit code |
|---|---|---|
| `GO` | Nothing failed | 0 |
| `CAUTION` | Usable, with warnings | 0 |
| `NO_GO` | A check failed | 1 |
| `INCOMPLETE` | A core check could not run, for example the GPU ran out of memory | 2 |

## Commands

```text
gguf-preflight check <endpoint> --model <name> [--gguf <path-or-url>]
                     [--agent claude-code|cline|generic] [--json] [--out receipt.json]
                     [--deep] [--allow-remote]
gguf-preflight inspect <path-or-hf-url> [--json]
gguf-preflight diff <receipt_a.json> <receipt_b.json> [--json] [--fail-on-regression]
gguf-preflight mcp
```

Public hosts are refused unless you pass `--allow-remote`. If your server needs a key, set `GGUF_PREFLIGHT_API_KEY`. It is never written to a receipt. Set `HF_TOKEN` for gated Hugging Face repos.

## Receipts

`--out` saves a JSON receipt with the verdict, GGUF metadata, runtime, hardware, every probe result, the fixes, and a hash of the test transcript. Compare two receipts after you change a quant, a flag, or a runtime version:

```powershell
gguf-preflight check http://localhost:11434 --model my-model --out before.json
gguf-preflight check http://localhost:11434 --model my-model --out after.json
gguf-preflight diff before.json after.json --fail-on-regression
```

Receipts keep only the file name of local GGUFs, not their folders.

## MCP server

The server exposes `preflight_check`, `inspect_gguf`, and `compare_receipts`. It is stricter than the CLI: endpoints must be local, and remote GGUFs must be `https` Hugging Face URLs.

**Claude Code**

```powershell
claude mcp add gguf-preflight -- uvx gguf-preflight mcp
```

**Gemini CLI** (`~/.gemini/settings.json`)

```json
{
  "mcpServers": {
    "gguf-preflight": {
      "command": "uvx",
      "args": ["gguf-preflight", "mcp"]
    }
  }
}
```

## Python library

```python
from gguf_preflight.inspector.gguf_header import inspect_gguf
from gguf_preflight.runner import run_preflight

metadata = inspect_gguf("model.gguf")
receipt = run_preflight("http://localhost:11434", "qwen3:8b", agent="claude-code")
print(receipt.verdict, receipt.failure_codes)
```

## Safety and limits

- **Model-written code runs on your machine.** The coding tasks execute the model's Python in a separate process with a timeout, a stripped environment, and an audit hook that blocks network, subprocess, and file-write calls. **This is not a security sandbox.** Only test models you are willing to run code from, or use a VM or container.
- **`tensor_sha256` is a layout fingerprint.** It hashes tensor names, shapes, and types, not the weights. Two fine-tunes with the same layout share it.
- **Determinism is best effort.** Requests use `temperature=0`, `top_p=1.0`, and `seed=42`. Output can still differ between runs, even on the same GPU, so `transcript_sha256` can change without a real regression. Treat a matching hash as strong evidence and a different hash as a prompt to read the diff.
- **Task scores are a smoke test.** They catch broken stacks. They do not rank models.

## What has been tested

Live runs used Windows 11, Ollama 0.40.12, an 8 GB RTX 5060, and an 8B Q4_K_M model:

| Setup | Verdict | Main finding | Full run |
|---|---|---|---|
| Ollama default, 4,096 context | `NO_GO` | A 3,933-token prompt was silently cut to 2,050 tokens | 126 to 194 s |
| Model maximum, 40,960 context, 69% on GPU | `CAUTION` | Below the 64k guidance for Claude Code, and slow | 404 to 428 s |

Run time depends mostly on generation speed, so slower or memory-constrained GPUs take several minutes.

**Not live-tested yet:** llama-server and LM Studio (covered by mocked HTTP tests only), Linux, and macOS.

## Development

```powershell
uv sync
uv run pytest
uv run mypy
```

Task packs live in `suites/` as YAML, so new tasks need no probe code.

## License

MIT
