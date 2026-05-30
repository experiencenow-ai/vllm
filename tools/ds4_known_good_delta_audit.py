#!/usr/bin/env python3
"""Compare this vLLM tree against a known-good GB10/SM12x DSV4 vLLM tree.

This is intentionally a read-only audit tool.  It does not try to patch the
checkout.  It fetches a known-good reference tree, compares only the subsystems
that decide the DSV4/Qwen native GB10 fast path, and writes a JSON + Markdown
report that can be used to port exact deltas without symptom-driven guessing.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

DEFAULT_KNOWN_GOOD_REPO = "https://github.com/jasl/vllm.git"
DEFAULT_KNOWN_GOOD_REFS = [
    "dda4668b59567416f86956cfe7bbc1eab371a61e",
    "27fd665bdc3ba58afc5c34cbb9034c9fc1a95029",
]

FOCUSED_PATHS = [
    "vllm/platforms/cuda.py",
    "vllm/envs.py",
    "vllm/utils/deep_gemm.py",
    "vllm/_custom_ops.py",
    "vllm/config/kernel.py",
    "vllm/model_executor/kernels/linear/__init__.py",
    "vllm/model_executor/kernels/linear/scaled_mm/cutlass.py",
    "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py",
    "vllm/model_executor/layers/quantization/modelopt.py",
    "vllm/compilation/passes/utility/fix_functionalization.py",
    "vllm/models/deepseek_v4",
    "vllm/v1/attention/ops/deepseek_v4_ops",
    "csrc",
    "cmake",
    "CMakeLists.txt",
    "setup.py",
    "pyproject.toml",
]

REQUIRED_GOOD_PATTERNS = {
    "sm12x_fp8_einsum_dispatch": [
        "deepseek_v4_sm12x_fp8_einsum",
        "_use_deepseek_v4_sm12x_triton_fp8_einsum",
        "direct_register_custom_op",
    ],
    "sm12x_mqa_logits_dispatch": [
        "fp8_fp4_paged_mqa_logits",
        "is_device_capability_family(120)",
    ],
    "sm12x_mhc_dispatch": [
        "tf32_hc_prenorm_gemm",
        "is_device_capability_family(120)",
    ],
    "gb10_blackwell_family_gate": [
        "is_device_capability_family(120)",
    ],
    "native_no_marlin_guard": [
        "MARLIN",
        "Mxfp4MoeBackend",
    ],
    "e8m0_scale_handling": [
        "float8_e8m0fnu",
    ],
}

KNOWN_HIGH_VALUE_FILES = [
    "vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py",
    "vllm/models/deepseek_v4/nvidia/ops/cutedsl_utils.py",
    "vllm/v1/attention/ops/deepseek_v4_ops/sm12x_deep_gemm_fallbacks.py",
    "vllm/models/deepseek_v4/fp8_einsum.py",
    "vllm/models/deepseek_v4/cutedsl_utils.py",
]

FORBIDDEN_FASTPATH_PATTERNS = [
    "Using 'MARLIN' Mxfp4 MoE backend",
    "EmulationNvFp4LinearKernel",
    "MarlinNvFp4LinearKernel",
    "MarlinMxFp4LinearKernel",
]


@dataclasses.dataclass(slots=True)
class PathStatus:
    path: str
    ours_exists: bool
    good_exists: bool
    ours_sha256: str | None
    good_sha256: str | None
    same: bool | None


@dataclasses.dataclass(slots=True)
class PatternStatus:
    name: str
    path: str
    ours: bool
    good: bool
    needles: list[str]


def run_checked(args: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        command = " ".join(args)
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
        )
    return completed.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def collect_files(root: Path, relative: str) -> list[Path]:
    path = root / relative
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file())


def relative_file_set(root: Path, relative: str) -> set[str]:
    return {str(p.relative_to(root)) for p in collect_files(root, relative)}


def git_clone_reference(repo: str, ref: str, workspace: Path) -> Path:
    checkout = workspace / "known_good"
    try:
        run_checked([
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repo,
            str(checkout),
        ])
    except RuntimeError as exc:
        if "unknown option `filter=blob:none'" not in str(exc):
            raise
        run_checked(["git", "clone", "--no-checkout", repo, str(checkout)])
    try:
        run_checked(["git", "fetch", "--depth", "1", "origin", ref], cwd=checkout)
    except RuntimeError:
        run_checked(
            [
                "git",
                "fetch",
                "--depth",
                "1",
                "origin",
                "+refs/heads/*:refs/remotes/origin/*",
            ],
            cwd=checkout,
        )
    try:
        run_checked(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=checkout)
        run_checked(["git", "checkout", "--detach", ref], cwd=checkout)
    except RuntimeError:
        run_checked(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout)
    return checkout


def list_changed_paths(ours: Path, good: Path) -> list[PathStatus]:
    paths: set[str] = set()
    for focus in FOCUSED_PATHS:
        paths.update(relative_file_set(ours, focus))
        paths.update(relative_file_set(good, focus))
    paths.update(KNOWN_HIGH_VALUE_FILES)

    statuses: list[PathStatus] = []
    for relative in sorted(paths):
        ours_path = ours / relative
        good_path = good / relative
        ours_exists = ours_path.is_file()
        good_exists = good_path.is_file()
        ours_sha = sha256_file(ours_path) if ours_exists else None
        good_sha = sha256_file(good_path) if good_exists else None
        same = None if not (ours_exists and good_exists) else ours_sha == good_sha
        statuses.append(
            PathStatus(
                path=relative,
                ours_exists=ours_exists,
                good_exists=good_exists,
                ours_sha256=ours_sha,
                good_sha256=good_sha,
                same=same,
            )
        )
    return statuses


def grep_tree(root: Path, needle: str, focuses: Iterable[str] = FOCUSED_PATHS) -> list[str]:
    hits: list[str] = []
    for focus in focuses:
        for path in collect_files(root, focus):
            text = read_text(path)
            if needle in text:
                hits.append(str(path.relative_to(root)))
    return sorted(set(hits))


def pattern_statuses(ours: Path, good: Path) -> list[PatternStatus]:
    statuses: list[PatternStatus] = []
    for name, needles in REQUIRED_GOOD_PATTERNS.items():
        for needle in needles:
            ours_hits = grep_tree(ours, needle)
            good_hits = grep_tree(good, needle)
            statuses.append(
                PatternStatus(
                    name=f"{name}:{needle}",
                    path=";".join(sorted(set(ours_hits + good_hits))) or "<none>",
                    ours=bool(ours_hits),
                    good=bool(good_hits),
                    needles=[needle],
                )
            )
    return statuses


def no_index_diff_stat(ours: Path, good: Path, relative: str) -> str:
    ours_path = ours / relative
    good_path = good / relative
    if not ours_path.exists() and not good_path.exists():
        return ""
    completed = subprocess.run(
        ["git", "diff", "--no-index", "--stat", str(ours_path), str(good_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.stdout.strip()


def write_report(
    output_dir: Path,
    *,
    ours: Path,
    good: Path,
    repo: str,
    ref: str,
    paths: list[PathStatus],
    patterns: list[PatternStatus],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_good_files = [p for p in paths if p.good_exists and not p.ours_exists]
    extra_ours_files = [p for p in paths if p.ours_exists and not p.good_exists]
    different_files = [p for p in paths if p.same is False]

    payload = {
        "known_good_repo": repo,
        "known_good_ref": ref,
        "ours_root": str(ours),
        "known_good_root": str(good),
        "summary": {
            "focused_files_total": len(paths),
            "missing_good_files": len(missing_good_files),
            "extra_ours_files": len(extra_ours_files),
            "different_files": len(different_files),
        },
        "path_status": [dataclasses.asdict(p) for p in paths],
        "pattern_status": [dataclasses.asdict(p) for p in patterns],
        "known_high_value_files": KNOWN_HIGH_VALUE_FILES,
    }
    (output_dir / "ds4_known_good_delta.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines: list[str] = []
    lines.append("# DS4 known-good vLLM delta audit")
    lines.append("")
    lines.append(f"Known-good repo: `{repo}`")
    lines.append(f"Known-good ref: `{ref}`")
    lines.append(f"Current tree: `{ours}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Focused files checked: {len(paths)}")
    lines.append(f"- Files present in known-good but missing here: {len(missing_good_files)}")
    lines.append(f"- Files present here but not in known-good: {len(extra_ours_files)}")
    lines.append(f"- Files present in both but different: {len(different_files)}")
    lines.append("")

    lines.append("## High-value missing files")
    lines.append("")
    high_value_missing = [p for p in missing_good_files if p.path in KNOWN_HIGH_VALUE_FILES]
    if high_value_missing:
        for status in high_value_missing:
            lines.append(f"- `{status.path}`")
    else:
        lines.append("- None of the configured high-value files are missing.")
    lines.append("")

    lines.append("## Pattern mismatches")
    lines.append("")
    pattern_mismatches = [p for p in patterns if p.good and not p.ours]
    if pattern_mismatches:
        for status in pattern_mismatches:
            lines.append(f"- Missing here, present in known-good: `{status.name}` in `{status.path}`")
    else:
        lines.append("- No configured pattern is present in known-good and absent here.")
    lines.append("")

    lines.append("## Focused diff stats")
    lines.append("")
    for focus in FOCUSED_PATHS:
        stat = no_index_diff_stat(ours, good, focus)
        if stat:
            lines.append(f"### `{focus}`")
            lines.append("")
            lines.append("```text")
            lines.append(stat)
            lines.append("```")
            lines.append("")

    lines.append("## Porting rule")
    lines.append("")
    lines.append(
        "Treat this report as a source-delta map, not as a request to wholesale "
        "replace the tree.  Port known-good GB10/SM12x native fast-path code "
        "below our external-KV/API modifications, then re-run this audit until "
        "only intentional DS4 local deltas remain."
    )
    lines.append("")

    (output_dir / "ds4_known_good_delta.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_KNOWN_GOOD_REPO)
    parser.add_argument(
        "--ref",
        default=DEFAULT_KNOWN_GOOD_REFS[0],
        help="Known-good vLLM git ref to compare against.",
    )
    parser.add_argument(
        "--ours",
        default=str(Path(__file__).resolve().parents[1]),
        help="Current vLLM checkout root. Defaults to this script's repo.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Temporary work dir. Defaults to a new temporary directory.",
    )
    parser.add_argument(
        "--out",
        default="ds4_known_good_delta_report",
        help="Output directory for JSON/Markdown reports.",
    )
    args = parser.parse_args()

    ours = Path(args.ours).resolve()
    if not (ours / "vllm").is_dir():
        raise SystemExit(f"not a vLLM checkout: {ours}")

    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="ds4_known_good_delta_")
        work_dir = Path(temp_context.name)

    try:
        good = git_clone_reference(args.repo, args.ref, work_dir)
        paths = list_changed_paths(ours, good)
        patterns = pattern_statuses(ours, good)
        write_report(
            Path(args.out).resolve(),
            ours=ours,
            good=good,
            repo=args.repo,
            ref=args.ref,
            paths=paths,
            patterns=patterns,
        )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    print(f"wrote {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
