# Blackwell HC operator provenance

Pinned from sgl-project/sglang 5c154c214df569403cb42c5dbb874e4f865aa379:
`kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py` and
`kernels/ops/elementwise/hc_mix.py`.
The GEMM implementation is originally NVIDIA/FlashInfer, as its header records.
Apache-2.0 license and original attribution are retained.
Only the dependency namespace in hc_mix.py is changed. No SGLang runtime is imported.
These operators retain the actual SM100 split-K and fused epilogue arithmetic.
