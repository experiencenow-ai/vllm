#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DS4_FLASHINFER_AUTOTUNE_TUNING_JOB=1
export DS4_ENABLE_FLASHINFER_AUTOTUNE=1
export DS4_FLASHINFER_JIT_MAX_JOBS="${DS4_FLASHINFER_JIT_MAX_JOBS:-1}"

echo "DS4 FlashInfer autotune tuning job: this is not a production service launcher" >&2
exec "$SCRIPT_DIR/ds4_launch_dsv4_flash_tp2_native_benchmark.sh" "$@"
