#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRITON_KERNELS_GIT_REPO="${TRITON_KERNELS_GIT_REPO:-https://github.com/triton-lang/triton.git}"
TRITON_KERNELS_GIT_REF="${TRITON_KERNELS_GIT_REF:-v3.5.1}"

"$PYTHON_BIN" -m pip install --no-deps --force-reinstall \
  "git+${TRITON_KERNELS_GIT_REPO}@${TRITON_KERNELS_GIT_REF}#subdirectory=python/triton_kernels"

"$PYTHON_BIN" "$SCRIPT_DIR/ds4_triton_jit_preflight.py" --skip-active-jit-probe --skip-libcuda-link-probe
