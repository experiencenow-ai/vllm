#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch DeepGEMM's SM12x scale-factor layout dispatch for DS4/GB10."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def replace_regex_required(
    text: str,
    pattern: str,
    patched_pattern: str,
    replacement: str,
    label: str,
) -> tuple[str, bool]:
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count == 0:
        if re.search(patched_pattern, text):
            return text, False
        raise RuntimeError(f"missing DeepGEMM patch target: {label}")
    return new_text, True


def replace_required(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise RuntimeError(f"missing DeepGEMM patch target: {label}")
    return text.replace(old, new), True


def patch_file(path: Path) -> bool:
    original = path.read_text()
    text = original

    if path.match("*/csrc/utils/layout.hpp"):
        replacements = (
            (
                "else if (arch_major == 10)",
                "else if (arch_major == 10 or arch_major == 12)",
                "default recipe SM100/SM12x branch",
            ),
            (
                "ab.scalar_type() == kPackedFP4 and arch_major == 10",
                "ab.scalar_type() == kPackedFP4 and (arch_major == 10 or arch_major == 12)",
                "FP4 AB layout check accepts SM12x",
            ),
        )
        for old, new, label in replacements:
            text, _ = replace_required(text, old, new, label)

    elif path.match("*/csrc/apis/layout.hpp"):
        replacements = (
            (
                "and (gran_k == 32 or gran_k == 128) and arch_major == 10) {",
                "and (gran_k == 32 or gran_k == 128) and (arch_major == 10 or arch_major == 12)) {",
                "FP32 SM100/SM12x SF transform",
            ),
            (
                "sf.scalar_type() == torch::kFloat and arch_major == 10)",
                "sf.scalar_type() == torch::kFloat and (arch_major == 10 or arch_major == 12))",
                "k-grouped FP32 SM100/SM12x SF transform",
            ),
            (
                "sf.scalar_type() == torch::kInt and arch_major == 10)",
                "sf.scalar_type() == torch::kInt and (arch_major == 10 or arch_major == 12))",
                "k-grouped INT SM100/SM12x SF transform",
            ),
        )
        for old, new, label in replacements:
            text, _ = replace_required(text, old, new, label)
        text, _ = replace_regex_required(
            text,
            r"and \(gran_k == 32 or gran_k == 128\) and arch_major == 10\)(\s+return)",
            r"and \(gran_k == 32 or gran_k == 128\) and \(arch_major == 10 or arch_major == 12\)\)(\s+return)",
            r"and (gran_k == 32 or gran_k == 128) and (arch_major == 10 or arch_major == 12))\1",
            "INT SM100/SM12x SF transform",
        )

    else:
        raise RuntimeError(f"unsupported path: {path}")

    if text == original:
        return False
    path.write_text(text)
    return True


def check_file(path: Path) -> None:
    text = path.read_text()
    if path.match("*/csrc/utils/layout.hpp"):
        required = (
            "else if (arch_major == 10 or arch_major == 12)",
            "ab.scalar_type() == kPackedFP4 and (arch_major == 10 or arch_major == 12)",
        )
        forbidden_patterns = (
            r"else if \(arch_major == 10\)",
            r"kPackedFP4 and arch_major == 10",
        )
    elif path.match("*/csrc/apis/layout.hpp"):
        required = (
            "and (gran_k == 32 or gran_k == 128) and (arch_major == 10 or arch_major == 12)) {",
            "sf.scalar_type() == torch::kFloat and (arch_major == 10 or arch_major == 12))",
            "sf.scalar_type() == torch::kInt and (arch_major == 10 or arch_major == 12))",
        )
        required_patterns = (
            r"and \(gran_k == 32 or gran_k == 128\) and \(arch_major == 10 or arch_major == 12\)\)\s+return",
        )
        forbidden_patterns = (
            r"gran_k == 32 or gran_k == 128\) and arch_major == 10",
            r"sf\.scalar_type\(\) == torch::kFloat and arch_major == 10",
            r"sf\.scalar_type\(\) == torch::kInt and arch_major == 10",
        )
    else:
        raise RuntimeError(f"unsupported path: {path}")

    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            f"DeepGEMM SM12x layout patch incomplete for {path}: missing {missing}"
        )

    for pattern in locals().get("required_patterns", ()):
        if not re.search(pattern, text):
            raise RuntimeError(
                f"DeepGEMM SM12x layout patch incomplete for {path}: missing pattern {pattern}"
            )

    for pattern in forbidden_patterns:
        if re.search(pattern, text):
            raise RuntimeError(
                f"DeepGEMM SM12x layout patch left stale SM100-only branch in {path}: {pattern}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("deepgemm_source", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify the patch has already been applied without modifying files.",
    )
    args = parser.parse_args()

    root = args.deepgemm_source.resolve()
    api_layout = root / "csrc" / "apis" / "layout.hpp"
    utils_layout = root / "csrc" / "utils" / "layout.hpp"

    for path in (api_layout, utils_layout):
        if not path.exists():
            raise RuntimeError(f"DeepGEMM source file not found: {path}")

    if not args.check_only:
        changed_paths = [path for path in (api_layout, utils_layout) if patch_file(path)]
        if changed_paths:
            for path in changed_paths:
                print(f"patched {path}")
        else:
            print("DeepGEMM SM12x layout patch already applied")

    for path in (api_layout, utils_layout):
        check_file(path)

    print("DeepGEMM SM12x layout patch verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
