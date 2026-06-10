#!/usr/bin/env python3
"""Run a vLLM module from one exact source checkout, fail-closed on drift.

This is intentionally used by DS4 Spark launch scripts instead of plain
``python -m vllm...``.  With ``python -m``, the current working directory and
editable-install .pth files can put an older checkout ahead of the requested
source tree.  That makes A/B tests invalid.  This wrapper makes the requested
source root sys.path[0], removes competing source roots, imports vLLM once, and
aborts if the resolved package is not the requested checkout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Iterable


DEFAULT_MODULE = "vllm.entrypoints.cli.main"


def _resolve_path(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def _looks_like_vllm_source_root(path_text: str) -> bool:
    if not path_text:
        return False
    try:
        path = _resolve_path(path_text)
    except OSError:
        return False
    return (path / "vllm" / "__init__.py").is_file()


def _sanitize_sys_path(source_root: Path, original: Iterable[str]) -> list[str]:
    sanitized: list[str] = [str(source_root)]
    seen = {str(source_root)}
    for entry in original:
        if not entry:
            # Empty entry means current working directory.  The wrapper chdirs to
            # source_root and already inserted it explicitly.
            continue
        try:
            resolved = _resolve_path(entry)
        except OSError:
            continue
        resolved_text = str(resolved)
        if resolved == source_root:
            continue
        if _looks_like_vllm_source_root(resolved_text):
            # Another checkout would make the launch ambiguous.  Drop it.
            continue
        if resolved_text in seen:
            continue
        sanitized.append(resolved_text)
        seen.add(resolved_text)
    return sanitized


def _sanitize_meta_path() -> list[str]:
    removed: list[str] = []
    kept = []
    for finder in sys.meta_path:
        finder_type = type(finder)
        finder_module = getattr(finder, "__module__", finder_type.__module__)
        finder_name = getattr(finder, "__name__", finder_type.__name__)
        marker = f"{finder_module}.{finder_name}"
        if "__editable___vllm" in marker or "__editable__.vllm" in marker:
            removed.append(marker)
            continue
        kept.append(finder)
    sys.meta_path[:] = kept
    return removed


def _package_root(module_file: str) -> Path:
    init_path = Path(module_file).resolve()
    # .../<source>/vllm/__init__.py -> .../<source>
    return init_path.parent.parent


def _constrain_vllm_package_path(vllm_module: object, source_root: Path) -> tuple[list[str], list[str]]:
    expected = str((source_root / "vllm").resolve())
    package_path = getattr(vllm_module, "__path__", None)
    original = [str(Path(p).resolve()) for p in package_path] if package_path is not None else []
    if package_path is not None:
        setattr(vllm_module, "__path__", [expected])
    module_spec = getattr(vllm_module, "__spec__", None)
    if module_spec is not None and getattr(module_spec, "submodule_search_locations", None) is not None:
        module_spec.submodule_search_locations = [expected]
    current_path = getattr(vllm_module, "__path__", [])
    current = [str(Path(p).resolve()) for p in current_path]
    return original, current


def _proof(source_root: Path, module: str) -> dict[str, object]:
    vllm_module = importlib.import_module("vllm")
    original_package_path, constrained_package_path = _constrain_vllm_package_path(
        vllm_module, source_root)
    actual_root = _package_root(str(vllm_module.__file__))
    return {
        "source_root": str(source_root),
        "module": module,
        "vllm_file": str(Path(str(vllm_module.__file__)).resolve()),
        "vllm_root": str(actual_root),
        "vllm_package_path_original": original_package_path,
        "vllm_package_path": constrained_package_path,
        "cwd": os.getcwd(),
        "sys_path_first": sys.path[:8],
        "python": sys.executable,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=os.getenv("DS4_VLLM_SOURCE_ROOT", ""),
        help="vLLM source checkout that must provide the imported vllm package",
    )
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only verify import resolution, then exit",
    )
    parser.add_argument(
        "--proof-json",
        default=os.getenv("DS4_VLLM_IMPORT_PROOF_JSON", ""),
        help="optional path to write import-resolution proof JSON",
    )
    parser.add_argument("module_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if not args.source_root:
        print("DS4 source-root guard requires --source-root or DS4_VLLM_SOURCE_ROOT", file=sys.stderr)
        return 64

    source_root = _resolve_path(args.source_root)
    if not (source_root / "vllm" / "__init__.py").is_file():
        print(f"DS4 source-root guard: not a vLLM source root: {source_root}", file=sys.stderr)
        return 64

    os.chdir(source_root)
    sys.path[:] = _sanitize_sys_path(source_root, sys.path)
    removed_meta_path = _sanitize_meta_path()
    os.environ["PYTHONPATH"] = str(source_root)

    proof = _proof(source_root, args.module)
    proof["removed_meta_path"] = removed_meta_path
    if Path(str(proof["vllm_root"])) != source_root:
        print("DS4 source-root guard: vLLM import drift detected", file=sys.stderr)
        print(json.dumps(proof, indent=2, sort_keys=True), file=sys.stderr)
        return 65

    if args.proof_json:
        proof_path = _resolve_path(args.proof_json)
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")

    print(
        "DS4 source-root guard: "
        f"imported vllm from {proof['vllm_file']} using source_root={source_root}",
        flush=True,
    )

    if args.check_only:
        return 0

    module_args = list(args.module_args)
    if module_args and module_args[0] == "--":
        module_args = module_args[1:]
    sys.argv = [args.module, *module_args]
    runpy.run_module(args.module, run_name="__main__", alter_sys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
