#!/usr/bin/env python3
"""Static checks for DS4 DSV4 sparse-MLA correctness diagnostics."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def check(name: str, ok: bool) -> int:
    print(("PASS" if ok else "FAIL") + f": {name}")
    return 0 if ok else 1


def main() -> int:
    envs = read("vllm/envs.py")
    flashmla = read("vllm/models/deepseek_v4/nvidia/flashmla.py")
    indexer = read("vllm/model_executor/layers/sparse_attn_indexer.py")
    indexer_op = indexer.split("def sparse_attn_indexer(", 1)[1].split(
        "def sparse_attn_indexer_fake(", 1
    )[0]
    launcher = read("tools/ds4_launch_dsv4_flash_pp8.sh")
    failures = 0
    for name in (
        "VLLM_DS4_DSV4_SPARSE_MLA_VALIDATE",
        "VLLM_DS4_DSV4_SPARSE_MLA_TRACE",
        "VLLM_DS4_DSV4_SPARSE_MLA_REF_CHECK",
        "VLLM_DS4_DSV4_SPARSE_MLA_REF_MAX_TOKENS",
        "VLLM_DS4_DSV4_SPARSE_MLA_REF_ATOL",
        "VLLM_DS4_DSV4_SPARSE_MLA_SELECTED_ABSMAX",
        "VLLM_DS4_DSV4_SPARSE_MLA_PREFILL_BACKEND",
    ):
        failures += check(
            f"{name} registered in envs",
            f"{name}:" in envs and f'"{name}": lambda:' in envs,
        )
        failures += check(
            f"{name} exported by DSV4 launcher",
            f"export {name}=" in launcher,
        )
    failures += check(
        "launcher logs sparse MLA mode",
        "triton_sparse_mla=$VLLM_TRITON_MLA_SPARSE" in launcher
        and "sparse_mla_validate=$VLLM_DS4_DSV4_SPARSE_MLA_VALIDATE" in launcher
        and "sparse_mla_ref_check=$VLLM_DS4_DSV4_SPARSE_MLA_REF_CHECK"
        in launcher,
    )
    failures += check(
        "prefill combined indices are validated",
        "_ds4_validate_indexed_sparse_mla_inputs(" in flashmla
        and 'stage="prefill"' in flashmla
        and "max_index=kv_flat.shape[0]" in flashmla,
    )
    failures += check(
        "decode topk and SWA indices are validated",
        'stage="compressed_decode_topk"' in flashmla
        and 'stage="compressed_decode_swa"' in flashmla
        and 'stage="swa_decode"' in flashmla,
    )
    failures += check(
        "singleton decode index head axis is normalized",
        "indices.dim() == 3 and indices.shape[1] == 1" in flashmla
        and "indices = indices[:, 0, :]" in flashmla,
    )
    failures += check(
        "sparse MLA output finite checks are present",
        "_ds4_check_sparse_mla_output(" in flashmla
        and "torch.isfinite(active).all()" in flashmla,
    )
    failures += check(
        "prefill selected row ranges are validated",
        "_ds4_validate_sparse_mla_prefill_selection(" in flashmla
        and "bad_compressed_rows" in flashmla
        and "bad_swa_rows" in flashmla
        and "selected invalid KV values" in flashmla,
    )
    failures += check(
        "prefill reference check is diagnostic-only",
        "_ds4_reference_check_sparse_mla_prefill(" in flashmla
        and "if not envs.VLLM_DS4_DSV4_SPARSE_MLA_REF_CHECK:" in flashmla
        and "torch.softmax(scores_with_sink" in flashmla,
    )
    failures += check(
        "prefill backend selector fails closed",
        "_ds4_sparse_mla_prefill_backend(" in flashmla
        and 'backend in ("", "unset", "auto")' in flashmla
        and "must be set " in flashmla
        and "explicitly for DSV4 sparse MLA prefill" in flashmla
        and "matmul-debug is diagnostic-only" in flashmla
        and "expected gathered, indexed-unsafe, or matmul-debug" in flashmla
        and 'backend in ("gathered", "gathered-sparse")' in flashmla
        and '"indexed-unsafe"' in flashmla
        and "VLLM_DS4_DSV4_SPARSE_MLA_PREFILL_BACKEND" in flashmla,
    )
    failures += check(
        "sparse prefill backend defaults fail closed",
        'VLLM_DS4_DSV4_SPARSE_MLA_PREFILL_BACKEND: str = "unset"' in envs
        and '"VLLM_DS4_DSV4_SPARSE_MLA_PREFILL_BACKEND", "unset"' in envs
        and "VLLM_DS4_DSV4_SPARSE_MLA_PREFILL_BACKEND must be explicit" in launcher
        and 'VLLM_DS4_DSV4_SPARSE_MLA_PREFILL_BACKEND="${VLLM_DS4_DSV4_SPARSE_MLA_PREFILL_BACKEND}"' in launcher,
    )
    failures += check(
        "gathered sparse prefill materializes selected KV before accumulation",
        "gather_indexed_sparse_mla_kv(" in flashmla
        and "accumulate_gathered_sparse_mla_attention_chunk(" in flashmla
        and "selected_kv_buffer" in flashmla,
    )
    failures += check(
        "materialized prefill diagnostic uses normal sparse MLA matmul path",
        "_forward_sparse_mla_prefill_matmul_debug(" in flashmla
        and "matmul_sparse_mla_attention_with_sink(" in flashmla
        and "valid_tokens = offsets < combined_lens.view(-1, 1)" in flashmla
        and "kv_flat.index_select" in flashmla,
    )
    matmul_debug_body = flashmla.split(
        "def _forward_sparse_mla_prefill_matmul_debug(", 1
    )[1].split("\n    @classmethod", 1)[0]
    failures += check(
        "matmul-debug prefill path also runs reference check",
        "_ds4_reference_check_sparse_mla_prefill(" in matmul_debug_body,
    )
    sparse_prefill_body = flashmla.split(
        "def _forward_sparse_mla_prefill_triton(", 1
    )[1].split("\n    @classmethod", 1)[0]
    outer_prefill_body = flashmla.split(
        "def _forward_prefill(", 1
    )[1].split("\n    @classmethod", 1)[0]
    failures += check(
        "indexed sparse prefill accepts caller-owned scratch buffers",
        "workspace_buffers:" in sparse_prefill_body
        and "len(workspace_buffers) == 4" in sparse_prefill_body
        and "selected_kv_buffer" in sparse_prefill_body,
    )
    failures += check(
        "prefill gather KV and sparse scratch share one workspace reservation",
        (
            "kv,\n                max_score_buffer,\n                denom_buffer,\n                output_buffer,"
            in outer_prefill_body
            or "workspace_specs = [" in outer_prefill_body
        )
        and "current_workspace_manager().get_simultaneous(" in outer_prefill_body
        and (
            "workspace_buffers=(" in outer_prefill_body
            or "workspace_buffers=tuple(" in outer_prefill_body
        ),
    )
    failures += check(
        "gathered selected-KV scratch shares the outer workspace reservation",
        "workspace_specs.append(" in outer_prefill_body
        and "selected_kv_buffer = workspace[4]" in outer_prefill_body
        and "current_workspace_manager().get_simultaneous(*workspace_specs)"
        in outer_prefill_body,
    )
    failures += check(
        "indexed sparse prefill no longer nests workspace allocation on hot path",
        "current_workspace_manager().get_simultaneous(" not in sparse_prefill_body,
    )
    failures += check(
        "prefill indexer can rebase packed absolute topk rows",
        "_localize_prefill_topk_indices_kernel" in indexer
        and "idx - start" in indexer
        and "idx >= start" in indexer
        and "idx < end" in indexer,
    )
    failures += check(
        "direct native topk path is localized before attention",
        "used_direct_topk = fp8_fp4_mqa_topk_indices(" in indexer
        and "if not used_direct_topk:" in indexer
        and "if current_platform.is_cuda() and used_direct_topk:" in indexer
        and "_localize_prefill_topk_indices(" in indexer,
    )
    failures += check(
        "materialized CUDA top_k_per_row_prefill is not localized twice",
        "ops.top_k_per_row_prefill(" in indexer
        and "if current_platform.is_cuda() and used_direct_topk:" in indexer
        and "if current_platform.is_cuda():\n                _localize_prefill_topk_indices("
        not in indexer,
    )
    failures += check(
        "prefill topk local-index contract is validated",
        "_validate_prefill_topk_indices_are_local(" in indexer
        and "DS4 sparse indexer produced non-local prefill top-k indices"
        in indexer
        and "view >= local_lens.view(-1, 1)" in indexer
        and "direct-localized" in indexer
        and "materialized" in indexer,
    )
    failures += check(
        "prefill topk contract validation skips CUDA graph capture",
        "_validate_prefill_topk_indices_are_local(" in indexer
        and "torch.cuda.is_current_stream_capturing()" in indexer
        and indexer.find("torch.cuda.is_current_stream_capturing()")
        < indexer.find("bad.any()"),
    )
    failures += check(
        "direct native topk path cannot bypass validation",
        "used_direct_topk = fp8_fp4_mqa_topk_indices(" in indexer
        and "_validate_prefill_topk_indices_are_local(" in indexer
        and "continue\n            if current_platform.is_xpu()" not in indexer,
    )
    failures += check(
        "registered sparse indexer op does not reference class self",
        "self." not in indexer_op,
    )
    failures += check(
        "launcher logs prefill backend",
        "sparse_mla_prefill_backend=$VLLM_DS4_DSV4_SPARSE_MLA_PREFILL_BACKEND"
        in launcher,
    )
    failures += check(
        "diagnostic helpers are outside torch compiler",
        "@torch.compiler.disable" in flashmla
        and "_ds4_sparse_mla_tensor_stats" in flashmla,
    )
    failures += check(
        "diagnostics avoid CUDA value sync during graph capture",
        "_ds4_cuda_graph_capture_active" in flashmla
        and "torch.cuda.is_current_stream_capturing()" in flashmla
        and "if _ds4_cuda_graph_capture_active():" in flashmla,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
