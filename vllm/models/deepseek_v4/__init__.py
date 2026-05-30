# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under ``nvidia/`` and ``amd/``. Keep this
package initializer light: low-level SM12x fallback modules import through this
package during native preflight, before the full model/MoE stack should load.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .quant_config import DeepseekV4FP8Config
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP


def __getattr__(name: str):
    if name == "DeepseekV4FP8Config":
        from .quant_config import DeepseekV4FP8Config

        return DeepseekV4FP8Config
    if name in {"DeepseekV4ForCausalLM", "DeepSeekV4MTP"}:
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            from .amd.model import DeepseekV4ForCausalLM
            from .amd.mtp import DeepSeekV4MTP
        else:
            from .nvidia.model import DeepseekV4ForCausalLM
            from .nvidia.mtp import DeepSeekV4MTP
        return {
            "DeepseekV4ForCausalLM": DeepseekV4ForCausalLM,
            "DeepSeekV4MTP": DeepSeekV4MTP,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
