"""Lightweight hardware detection with an optional `llmfit --json` integration."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

from gguf_preflight.models import GpuInfo, HardwareInfo

_COMMAND_TIMEOUT_S = 5


def _run(command: list[str]) -> str | None:
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_COMMAND_TIMEOUT_S,
            check=False,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _windows_ram_gb() -> float | None:
    if sys.platform != "win32":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("available_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return round(int(status.total_physical) / 1024**3, 1)
    return None


def _ram_gb() -> float | None:
    if sys.platform == "win32":
        return _windows_ram_gb()
    if sys.platform == "darwin":
        output = _run(["sysctl", "-n", "hw.memsize"])
        return round(int(output) / 1024**3, 1) if output and output.strip().isdigit() else None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (AttributeError, OSError, ValueError):
        return None


def _cpu_name() -> str | None:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            with key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except OSError:
            return platform.processor() or None
    if sys.platform == "darwin":
        output = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        return output.strip() if output else None
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = re.match(r"model name\s*:\s*(.+)", line)
                if match:
                    return match[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def parse_nvidia_smi(output: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        gpus.append(
            GpuInfo(
                name=parts[0],
                vram_total_mb=int(parts[1]) if parts[1].isdigit() else None,
                vram_used_mb=int(parts[2]) if parts[2].isdigit() else None,
            )
        )
    return gpus


def _nvidia_gpus() -> list[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_nvidia_smi(output) if output else []


def _llmfit() -> dict[str, Any] | None:
    if not shutil.which("llmfit"):
        return None
    output = _run(["llmfit", "--json"])
    if not output:
        return None
    try:
        data = json.loads(output)
    except ValueError:
        return None
    return data if isinstance(data, dict) else {"result": data}


def detect_hardware() -> HardwareInfo:
    llmfit = _llmfit()
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        cpu=_cpu_name(),
        logical_cpus=os.cpu_count(),
        ram_total_gb=_ram_gb(),
        gpus=_nvidia_gpus(),
        source="llmfit+builtin" if llmfit else "builtin",
        llmfit=llmfit,
    )
