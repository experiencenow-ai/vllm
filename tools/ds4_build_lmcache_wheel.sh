#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:?usage: tools/ds4_build_lmcache_wheel.sh /path/to/python /path/to/wheel-dir}"
WHEEL_DIR="${2:?usage: tools/ds4_build_lmcache_wheel.sh /path/to/python /path/to/wheel-dir}"

: "${CUDA_HOME:=/usr/local/cuda}"
: "${CPATH:=/tmp/ds4_python312_dev/root/usr/include:/tmp/ds4_python312_dev/root/usr/include/python3.12:/tmp/ds4_python312_dev/root/usr/include/aarch64-linux-gnu/python3.12}"
: "${TORCH_CUDA_ARCH_LIST:=12.1a}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "python binary is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi

if [[ ! -d "${CUDA_HOME}" ]]; then
    echo "CUDA_HOME does not exist: ${CUDA_HOME}" >&2
    exit 3
fi

mkdir -p "${WHEEL_DIR}"
export CUDA_HOME
export CPATH
export TORCH_CUDA_ARCH_LIST

exec "${PYTHON_BIN}" -m pip wheel --no-build-isolation --no-deps --wheel-dir "${WHEEL_DIR}" "lmcache>=0.4.5"
