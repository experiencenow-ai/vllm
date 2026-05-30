#!/usr/bin/env bash
# Build and install the SM12x-capable DeepGEMM fork required by
# DeepSeek-V4-Flash on DGX Spark / GB10. The upstream DeepGEMM ref used by
# vanilla vLLM can reject SM120/SM121 in tf32_hc_prenorm_gemm and paged-MQA
# kernels; this script pins the fork known to contain those DS4 paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${DEEPGEMM_GIT_REPO:=https://github.com/jasl/DeepGEMM.git}"
: "${DEEPGEMM_GIT_REF:=7a7a41a1}"
: "${CUDA_VERSION:=13.0}"
: "${TORCH_CUDA_ARCH_LIST:=12.1a}"

export DEEPGEMM_GIT_REPO
export DEEPGEMM_GIT_REF
export TORCH_CUDA_ARCH_LIST
export VLLM_DS4_PATCH_DEEPGEMM_SM12X=1
export DEEPGEMM_FORCE_REINSTALL=1

ds4_add_include_dir()
{
    local include_dir="$1"
    export CFLAGS="-I${include_dir} ${CFLAGS:-}"
    export CXXFLAGS="-I${include_dir} ${CXXFLAGS:-}"
    export CPATH="${include_dir}${CPATH:+:${CPATH}}"
}

if [[ -n "${DS4_PYTHON_INCLUDE_DIR:-}" ]]; then
    if [[ ! -f "${DS4_PYTHON_INCLUDE_DIR}/Python.h" ]]; then
        echo "DS4_PYTHON_INCLUDE_DIR=${DS4_PYTHON_INCLUDE_DIR} does not contain Python.h" >&2
        exit 64
    fi
    ds4_add_include_dir "${DS4_PYTHON_INCLUDE_DIR}"
    ds4_add_include_dir "$(dirname "${DS4_PYTHON_INCLUDE_DIR}")"
    for extra_include_dir in "$(dirname "${DS4_PYTHON_INCLUDE_DIR}")"/*/"$(basename "${DS4_PYTHON_INCLUDE_DIR}")"; do
        if [[ -f "${extra_include_dir}/pyconfig.h" ]]; then
            ds4_add_include_dir "${extra_include_dir}"
        fi
    done
fi

echo "DS4 GB10 DeepGEMM native build"
echo "  repo: ${DEEPGEMM_GIT_REPO}"
echo "  ref:  ${DEEPGEMM_GIT_REF}"
echo "  CUDA: ${CUDA_VERSION}"
echo "  arch: ${TORCH_CUDA_ARCH_LIST}"
if [[ -n "${DS4_PYTHON_INCLUDE_DIR:-}" ]]; then
    echo "  python include: ${DS4_PYTHON_INCLUDE_DIR}"
fi

exec "${SCRIPT_DIR}/install_deepgemm.sh" \
    --repo "${DEEPGEMM_GIT_REPO}" \
    --ref "${DEEPGEMM_GIT_REF}" \
    --cuda-version "${CUDA_VERSION}" \
    "$@"
