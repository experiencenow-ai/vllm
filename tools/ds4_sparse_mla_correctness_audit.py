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
        and "expected indexed or matmul-debug" in flashmla
        and "VLLM_DS4_DSV4_SPARSE_MLA_PREFILL_BACKEND" in flashmla,
    )
    failures += check(
        "materialized prefill diagnostic uses normal sparse MLA matmul path",
        "_forward_sparse_mla_prefill_matmul_debug(" in flashmla
        and "matmul_sparse_mla_attention_with_sink(" in flashmla
        and "valid_tokens = offsets < combined_lens.view(-1, 1)" in flashmla
        and "kv_flat.index_select" in flashmla,
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
