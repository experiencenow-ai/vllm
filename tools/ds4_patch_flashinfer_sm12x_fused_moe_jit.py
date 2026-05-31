#!/usr/bin/env python3
"""Patch FlashInfer SM12x fused-MoE JIT to avoid irrelevant arch kernels.

FlashInfer 0.6.11's `fused_moe_120` JIT module can generate a build graph
containing SM120 plus older SM80/SM90 generated kernels.  On GB10/SM121 DS4
deployments that makes normal startup spend hours compiling kernels that cannot
be selected by the native SM120 path.  Keep the package behavior unchanged for
other architectures, but filter the SM120 module to its SM120 generated kernels.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import py_compile
import shutil
import sys


OLD = '''            # Add all generated kernels
            *(output_dir / kernel for kernel in output_dir.rglob("*.generated.cu")),
'''

NEW = '''            # Add only generated kernels for the active SM120 module.
            *(
                output_dir / kernel
                for kernel in output_dir.rglob("*.generated.cu")
                if device_arch != "120" or "_sm120_" in kernel.name
            ),
'''

BAD_NEW = '''            # Add only generated kernels for the active SM120 module.
            *(
                output_dir / kernel
                for kernel in output_dir.rglob("*.generated.cu")
                if device_arch != "120" or kernel.name.startswith("120_")
            ),
'''


def main() -> int:
    spec = importlib.util.find_spec("flashinfer.jit.fused_moe")
    if spec is None or spec.origin is None:
        print("ds4_flashinfer_sm12x_patch: flashinfer.jit.fused_moe not found", file=sys.stderr)
        return 2
    path = Path(spec.origin)
    text = path.read_text()
    if NEW in text:
        print(f"ds4_flashinfer_sm12x_patch: already patched {path}")
        return 0
    if OLD in text:
        old = OLD
    elif BAD_NEW in text:
        old = BAD_NEW
    else:
        print(f"ds4_flashinfer_sm12x_patch: expected source block not found in {path}", file=sys.stderr)
        return 3
    backup = path.with_suffix(path.suffix + ".ds4-sm12x-backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(old, NEW, 1))
    py_compile.compile(str(path), doraise=True)
    print(f"ds4_flashinfer_sm12x_patch: patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
