"""Keep DS4 vLLM source-root launches isolated in child Python processes."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _resolve_path(raw: str) -> Path | None:
    try:
        return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    except OSError:
        return None


def _looks_like_vllm_source_root(path: Path) -> bool:
    return (path / "vllm" / "__init__.py").is_file()


def _sanitize_sys_path(source_root: Path) -> None:
    source_text = str(source_root)
    sanitized: list[str] = []
    seen = set()
    for entry in sys.path:
        if not entry:
            continue
        resolved = _resolve_path(entry)
        if resolved is None:
            continue
        resolved_text = str(resolved)
        if resolved == source_root:
            continue
        if _looks_like_vllm_source_root(resolved):
            continue
        if resolved_text in seen:
            continue
        sanitized.append(resolved_text)
        seen.add(resolved_text)
    sys.path[:] = [source_text, *sanitized]


def _sanitize_meta_path() -> None:
    kept = []
    for finder in sys.meta_path:
        finder_type = type(finder)
        finder_module = getattr(finder, "__module__", finder_type.__module__)
        finder_name = getattr(finder, "__name__", finder_type.__name__)
        marker = f"{finder_module}.{finder_name}"
        if "__editable___vllm" in marker or "__editable__.vllm" in marker:
            continue
        kept.append(finder)
    sys.meta_path[:] = kept


def _main() -> None:
    source_root_raw = os.getenv("DS4_VLLM_SOURCE_ROOT", "")
    if not source_root_raw:
        return
    source_root = _resolve_path(source_root_raw)
    if source_root is None or not _looks_like_vllm_source_root(source_root):
        return
    _sanitize_meta_path()
    _sanitize_sys_path(source_root)


_main()
