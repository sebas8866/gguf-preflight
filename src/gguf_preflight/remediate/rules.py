"""Maps finding checks plus runtime to copy-paste fixes.

Flags and variables are taken from the upstream docs linked in `source`. The two
Claude Code variables were confirmed in the Claude Code 2.1.325 binary.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gguf_preflight.models import Finding, RecommendedFix, Severity

LLAMA_SERVER_DOCS = "https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md"
LLAMA_QUANTIZE_DOCS = "https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md"
LLAMA_IMATRIX_DOCS = "https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md"
OLLAMA_CONTEXT_DOCS = "https://docs.ollama.com/context-length"
OLLAMA_CONFIG_SOURCE = "https://github.com/ollama/ollama/blob/main/envconfig/config.go"
LMSTUDIO_LOAD_DOCS = "https://lmstudio.ai/docs/cli/load"
ATTRIBUTION_HEADER_NOTE = (
    "https://mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix/"
)
ANY = "*"
DEFAULT_CONTEXT = 64000


def env_commands(name: str, value: str, then: str = "") -> list[str]:
    """One command for PowerShell and one for POSIX shells."""
    suffix = f"; {then}" if then else ""
    posix = f"{name}={value} {then}" if then else f"export {name}={value}"
    return [f'$env:{name}="{value}"{suffix}  # PowerShell', f"{posix}  # bash/zsh"]


@dataclass(frozen=True)
class FixTemplate:
    title: str
    why: str
    commands: tuple[str, ...] = ()
    source: str | None = None


def _context_fixes() -> dict[str, FixTemplate]:
    why = (
        "The loaded context is smaller than the prompts an agent sends. Larger contexts use "
        "more memory, so lower the value if the model stops fitting on your GPU."
    )
    return {
        "ollama": FixTemplate(
            "Raise Ollama's context length",
            why,
            tuple(env_commands("OLLAMA_CONTEXT_LENGTH", str(DEFAULT_CONTEXT), "ollama serve")),
            OLLAMA_CONTEXT_DOCS,
        ),
        "llama_server": FixTemplate(
            "Raise llama-server's context size",
            why,
            (f"llama-server -m <model.gguf> -c {DEFAULT_CONTEXT} --jinja",),
            LLAMA_SERVER_DOCS,
        ),
        "lmstudio": FixTemplate(
            "Reload the model with a larger context",
            why,
            (f"lms load <model> --context-length {DEFAULT_CONTEXT}",),
            LMSTUDIO_LOAD_DOCS,
        ),
        ANY: FixTemplate("Increase the server's context size", why),
    }


_TEMPLATE_WHY = "Tool calls depend on the model's chat template being applied correctly."

RULES: dict[str, dict[str, FixTemplate]] = {
    "prefix_cache_invalidated": {
        "ollama": FixTemplate(
            "Keep Ollama's KV cache stable",
            "A quantized KV cache or a model reload can force full prompt reprocessing.",
            (
                *env_commands("OLLAMA_KV_CACHE_TYPE", "f16"),
                *env_commands("OLLAMA_NUM_PARALLEL", "1"),
                *env_commands("OLLAMA_KEEP_ALIVE", "30m", "ollama serve"),
            ),
            OLLAMA_CONFIG_SOURCE,
        ),
        "llama_server": FixTemplate(
            "Enable prompt reuse and an unquantized KV cache",
            "These settings let llama-server reuse the unchanged part of each prompt.",
            ("llama-server -m <model.gguf> --cache-prompt --cache-reuse 256 -ctk f16 -ctv f16",),
            LLAMA_SERVER_DOCS,
        ),
        "lmstudio": FixTemplate(
            "Keep the model loaded between turns",
            "Reloading or changing load settings discards LM Studio's prompt cache.",
            ("lms ps",),
            LMSTUDIO_LOAD_DOCS,
        ),
        ANY: FixTemplate(
            "Enable prompt caching on your server",
            "Without prefix reuse, every agent turn reprocesses the full prompt.",
        ),
    },
    "silent_context_truncation": _context_fixes(),
    "context_below_agent_minimum": _context_fixes(),
    "context_limit_explicit": _context_fixes(),
    "missing_chat_template": {
        "ollama": FixTemplate(
            "Add a chat template with a Modelfile",
            _TEMPLATE_WHY,
            ('Add a TEMPLATE block to a Modelfile, then run: ollama create <name> -f Modelfile',),
            "https://docs.ollama.com/modelfile",
        ),
        "llama_server": FixTemplate(
            "Provide a chat template",
            _TEMPLATE_WHY,
            ("llama-server -m <model.gguf> --jinja --chat-template-file <template.jinja>",),
            LLAMA_SERVER_DOCS,
        ),
        ANY: FixTemplate(
            "Use a GGUF that embeds a chat template",
            _TEMPLATE_WHY,
        ),
    },
    "missing_pre_tokenizer": {
        ANY: FixTemplate(
            "Re-convert or replace this GGUF",
            "Without the right pre-tokenizer, text can be split into the wrong tokens.",
            ("python convert_hf_to_gguf.py <hf-model-dir> --outfile <model.gguf>",),
            "https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py",
        ),
    },
    "missing_tokenizer": {
        ANY: FixTemplate(
            "Replace this GGUF",
            "The file has no tokenizer metadata, so runtimes cannot encode text reliably.",
        ),
    },
    "unknown_tensor_type": {
        ANY: FixTemplate(
            "Update gguf-preflight and your runtime",
            "The file uses a tensor type this version does not recognize.",
        ),
    },
    "uncalibrated_low_bit_quant": {
        ANY: FixTemplate(
            "Use an imatrix-calibrated or higher-bit GGUF",
            "An importance matrix helps low-bit quantization keep the most important weights.",
            (
                "llama-imatrix -m <model-f16.gguf> -f <calibration.txt> -o imatrix.gguf",
                "llama-quantize --imatrix imatrix.gguf <model-f16.gguf> <out.gguf> Q3_K_M",
            ),
            LLAMA_IMATRIX_DOCS,
        ),
    },
    "tool_call_plaintext_leak": {
        "ollama": FixTemplate(
            "Use a tool-capable Ollama model",
            "Ollama only parses tool calls when the model's template supports tools.",
            ("ollama show <model>",),
            "https://docs.ollama.com/capabilities/tool-calling",
        ),
        "llama_server": FixTemplate(
            "Enable Jinja tool-call parsing",
            "The server returned tool markup as text instead of structured tool calls.",
            ("llama-server -m <model.gguf> --jinja",),
            LLAMA_SERVER_DOCS,
        ),
        ANY: FixTemplate(
            "Use a model and template with native tool calling",
            "The server returned tool markup as text instead of structured tool calls.",
        ),
    },
    "quant_collapse": {
        ANY: FixTemplate(
            "Try a higher-bit or imatrix-calibrated quant",
            "The model produced fluent output but failed verifiable tasks.",
            ("gguf-preflight inspect <path-or-hf-url>",),
            LLAMA_QUANTIZE_DOCS,
        ),
    },
    "slow_prefill": {
        "ollama": FixTemplate(
            "Check GPU offload",
            "Slow prompt processing usually means layers are running on the CPU.",
            ("ollama ps",),
            OLLAMA_CONTEXT_DOCS,
        ),
        "llama_server": FixTemplate(
            "Offload all layers to the GPU",
            "Slow prompt processing usually means layers are running on the CPU.",
            ("llama-server -m <model.gguf> -ngl 99",),
            LLAMA_SERVER_DOCS,
        ),
        ANY: FixTemplate(
            "Use a smaller model or context",
            "The stack is too slow at agent-sized prompts.",
        ),
    },
    "slow_generation": {
        "llama_server": FixTemplate(
            "Offload all layers to the GPU",
            "Low generation speed usually means layers are running on the CPU.",
            ("llama-server -m <model.gguf> -ngl 99",),
            LLAMA_SERVER_DOCS,
        ),
        "ollama": FixTemplate(
            "Check GPU offload",
            "Low generation speed usually means layers are running on the CPU.",
            ("ollama ps",),
            OLLAMA_CONTEXT_DOCS,
        ),
        ANY: FixTemplate(
            "Use a smaller model or quant",
            "Generation is too slow for interactive agent use.",
        ),
    },
}

RULES["model_context_limit"] = {
    ANY: FixTemplate(
        "Use a model with a larger context window",
        "The context setting is already at this model's limit. Long agent sessions will "
        "compact more often, or need a model that supports more context.",
    ),
}

_OOM_WHY = (
    "The server ran out of GPU memory during the run, so the results are incomplete. "
    "Close other GPU apps or load the model with a smaller context, then run again."
)
RULES["runtime_out_of_memory"] = {
    "ollama": FixTemplate(
        "Free GPU memory or lower Ollama's context",
        _OOM_WHY,
        ("nvidia-smi", "ollama ps", *env_commands("OLLAMA_CONTEXT_LENGTH", "16000", "ollama serve")),
        OLLAMA_CONTEXT_DOCS,
    ),
    "llama_server": FixTemplate(
        "Free GPU memory or lower llama-server's context",
        _OOM_WHY,
        ("nvidia-smi", "llama-server -m <model.gguf> -c 16000 --jinja"),
        LLAMA_SERVER_DOCS,
    ),
    "lmstudio": FixTemplate(
        "Free GPU memory or reload with a smaller context",
        _OOM_WHY,
        ("nvidia-smi", "lms unload --all", "lms load <model> --context-length 16000"),
        LMSTUDIO_LOAD_DOCS,
    ),
    ANY: FixTemplate("Free GPU memory or lower the context", _OOM_WHY, ("nvidia-smi",)),
}

_LLAMA_TOOL_FIX = RULES["tool_call_plaintext_leak"]
for _check in (
    "tool_call_empty_response",
    "tool_args_invalid_json",
    "tool_stream_malformed",
    "tool_call_http_error",
):
    RULES[_check] = _LLAMA_TOOL_FIX

# Check ids that may alias to another rule.
ALIASES = {
    "smoke_code_collapse": "quant_collapse",
    "smoke_tools_collapse": "quant_collapse",
    "smoke_context_collapse": "quant_collapse",
    "canary_recall_failed": "quant_collapse",
    "tool_call_wrong_shape": "quant_collapse",
}

_CLAUDE_CODE_CACHE_FIXES = (
    RecommendedFix(
        check="prefix_cache_invalidated",
        adapter="claude-code",
        title="Stop Claude Code's per-turn token reminder",
        why="The changing reminder can invalidate a local server's prompt cache every turn.",
        commands=env_commands("CLAUDE_CODE_TOTAL_TOKENS_REMINDER", "off"),
    ),
    RecommendedFix(
        check="prefix_cache_invalidated",
        adapter="claude-code",
        title="Turn off Claude Code's attribution header",
        why="Its changing value at the start of the prompt forces local servers to reprocess.",
        commands=env_commands("CLAUDE_CODE_ATTRIBUTION_HEADER", "0"),
        source=ATTRIBUTION_HEADER_NOTE,
    ),
)


def _template_for(check: str, adapter: str) -> FixTemplate | None:
    rule = RULES.get(ALIASES.get(check, check))
    if rule is None:
        return None
    return rule.get(adapter) or rule.get(ANY)


def recommend(
    findings: Iterable[Finding], adapter: str, agent: str = "generic"
) -> list[RecommendedFix]:
    """Return de-duplicated fixes, most severe first."""
    rank = {Severity.FAIL: 0, Severity.WARN: 1, Severity.INFO: 2}
    ordered = sorted(findings, key=lambda finding: rank[finding.severity])
    fixes: list[RecommendedFix] = []
    seen: set[str] = set()
    for finding in ordered:
        if finding.severity is Severity.INFO:
            continue
        if finding.check == "prefix_cache_invalidated" and agent == "claude-code":
            for fix in _CLAUDE_CODE_CACHE_FIXES:
                if fix.title not in seen:
                    seen.add(fix.title)
                    fixes.append(fix.model_copy(update={"failure_code": finding.code}))
        template = _template_for(finding.check, adapter)
        if template is None or template.title in seen:
            continue
        seen.add(template.title)
        fixes.append(
            RecommendedFix(
                check=finding.check,
                failure_code=finding.code,
                adapter=adapter,
                title=template.title,
                why=template.why,
                commands=list(template.commands),
                source=template.source,
            )
        )
    return fixes
