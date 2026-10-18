# Blackwell HC operator provenance

Pinned from sgl-project/sglang 5c154c214df569403cb42c5dbb874e4f865aa379:
`kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py` and
`kernels/ops/elementwise/hc_mix.py`.
The GEMM implementation is originally NVIDIA/FlashInfer, as its header records.
Apache-2.0 license and original attribution are retained.
The dependency namespace in hc_mix.py is changed. No SGLang runtime is imported.
These operators retain the actual SM100 split-K and fused epilogue arithmetic.

## Retained entry points and pruning boundary

Production uses `permute_pad_up_weight` during loading and `hc_mix` during small
batch execution. `hc_mix` calls `run_splitk_dense_silu` and
`run_splitk_dense_gate`; its tactic selection uses `SplitKTactic`,
`default_tactic`, and `validate_tactic` (not `autotune_tactics`).

A repository-wide production/test call-site audit removed the unused
`hc_mix_shape_ok`, `autotune_tactics`, and general `run_splitk_dense` entry points.
The last removal also eliminates its exclusive bias launcher `_bmm_bias` and
bias tensor plumbing in the host validation/compilation helpers. The remaining
host paths always instantiate the original device template with `has_bias=False`.

The device template, including its compile-time-disabled bias branch, is kept
unchanged to avoid an unrelated rewrite of the pinned kernel. Tactic validation,
scratch buffers, weight packing, layout handling, split-K reduction, SiLU/gate
epilogues, and CUDA Graph support remain. `test_qwen4_sglang.py` checks HC output
and graph replay against the independent SGLang implementation, including the
32-row scratch-buffer boundary.
