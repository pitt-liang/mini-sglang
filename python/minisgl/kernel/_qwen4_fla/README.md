# Qwen4 GDN operator provenance

Pinned from sgl-project/sglang commit 5c154c214df569403cb42c5dbb874e4f865aa379,
python/sglang/kernels/ops/attention/fla. Original copyright and attribution
headers are retained; SGLang's Apache-2.0 license is included in LICENSE.
These operators derive from flash-linear-attention and, for packed decode,
vLLM as indicated by their original source headers.

Local changes: package-relative dependency namespace; local torch version
inspection; removal of the unrelated Intel runtime override from chunk.py;
only GDN packed decode retained from fused_recurrent.py. No SGLang scheduler,
model, runtime context, or request/cache manager is imported.

## Retained entry points and pruning boundary

This is a private inference dependency, not a general FLA API. The model imports
`causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`,
`fused_gdn_gating`, and `fused_recurrent_gated_delta_rule_packed_decode`.
The chunk entry point retains its transitive L2 normalization, cumsum, chunk
index/offset, intra-chunk solve, W/U reconstruction, state update and output
dependencies. Checkpoint tests also call the chunk entry point directly.

A repository-wide production/test call-site audit removed:

- `gdn_prefill_qkv_prepare_fwd` and its exclusive materialization kernel: no caller;
  the model uses chunk prefill and `l2norm_fwd` instead of this FlashInfer adapter.
- `L2NormFunction`, `l2norm`/`l2_norm`, and `L2Norm`: an unreferenced wrapper chain;
  production calls `l2norm_fwd` directly.
- The unused `fwd_recompute_w_u` alias; `recompute_w_u_fwd` remains in use.
- Unused environment-check/error-reporting, training checkpoint/version-check,
  backward-autocast and multiprocessor-query helpers, their exclusive flags,
  and unused log/gather/tensor-descriptor exports.
- Tuning tables referenced only by commented-out autotune decorators. Active
  autotune configurations and their capability checks are unchanged.

Shape-conditional implementations are not dead entries: both L2 kernels and
scalar/vector cumsum remain reachable from their retained wrappers. Their
arithmetic is unchanged and is covered by `test_qwen4_fla_primitives.py`.
Do not remove state tracking, slot masks, ragged inputs or graph support based
only on the default model shape or a single observed execution trace.

`causal_conv.py` comes from `kernels/ops/mamba/causal_conv1d_triton.py`
at the same revision. Its runtime PDL capability query is replaced with a
local constant `False`; this changes launch scheduling, not arithmetic.

The chunk64 prefill algorithm and packed FP32-normalized decode are retained
to match the actual SGLang FP32-state baseline, not replaced with a different
FlashInfer chunk algorithm. Independent parity tests must accompany updates.

The sibling JIT source `../csrc/jit/qwen4_topk.cu` adapts
`kernels/jit/csrc/elementwise/fast_topk.cuh` from the same SGLang revision
under the included Apache-2.0 license. Its radix algorithm is unchanged;
the host wrapper uses mini's launch utilities and disables PDL scheduling.

The sibling `../qwen4_ops.py` and `../csrc/jit/qwen4_hc.cu` reproduce the
pinned SGLang HC, MoE, GDN output, QSA index normalization and PLE numerical
contracts. In particular, PLE gate-value arithmetic follows
`kernels/ops/qwen4_ple.py`; the decode state mover is adapted to mini's explicit
valid mask (slot zero is a real mini request slot), while Conv1D/SiLU remain
native Torch as in the independent SGLang model.
