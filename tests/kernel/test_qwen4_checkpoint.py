"""FP32 internal checkpoint must not change the prefill's outputs or final state."""

import pytest
import torch
from minisgl.kernel._qwen4_fla.chunk import chunk_gated_delta_rule


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length,boundary", [(65, 64), (133, 128), (2053, 2048)])
@pytest.mark.parametrize("key_heads,value_heads", [(2, 2), (8, 24)])
def test_fp32_internal_checkpoint(length, boundary, key_heads, value_heads):
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(1, length, heads, 128, device="cuda", dtype=torch.bfloat16)
        for heads in (key_heads, key_heads, value_heads)
    ]
    g = -torch.rand(1, length, value_heads, device="cuda")
    beta = torch.rand(1, length, value_heads, device="cuda")
    seed = torch.randn(3, value_heads, 128, 128, device="cuda") * 0.1
    slots = torch.tensor([2], device="cuda", dtype=torch.int32)

    def run(state, end, start=0, **kwargs):
        return chunk_gated_delta_rule(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            initial_state=state,
            initial_state_indices=slots,
            cu_seqlens=torch.tensor([0, end - start], device="cuda", dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
            **kwargs,
        )[0]

    baseline_state, tracked_state, prefix_state = [seed.clone() for _ in range(3)]
    baseline = run(baseline_state, length)
    snapshot = torch.empty(1, value_heads, 128, 128, device="cuda", dtype=torch.float32)
    tracked = run(
        tracked_state,
        length,
        track_state=snapshot,
        track_chunk_idx=torch.tensor([boundary // 64], device="cuda", dtype=torch.int32),
    )
    run(prefix_state, boundary)
    torch.testing.assert_close(tracked, baseline, atol=0, rtol=0)
    torch.testing.assert_close(tracked_state, baseline_state, atol=0, rtol=0)
    torch.testing.assert_close(snapshot[0], prefix_state[2], atol=0, rtol=0)
    restored_state = seed.clone()
    restored_state[2].copy_(snapshot[0])
    resumed = run(restored_state, length, start=boundary)
    torch.testing.assert_close(resumed, baseline[:, boundary:], atol=0, rtol=0)
    torch.testing.assert_close(restored_state, baseline_state, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hc_projection_is_invariant_to_prefix_removal():
    from minisgl.kernel.qwen4_ops import gr_project_mix_stable, linear_stable

    torch.manual_seed(17)
    x = torch.randn(133, 10240, device="cuda", dtype=torch.bfloat16)
    down = torch.randn(320, 10240, device="cuda", dtype=torch.bfloat16) * 0.01
    up = torch.randn(10240, 320, device="cuda", dtype=torch.bfloat16) * 0.01
    baseline = gr_project_mix_stable(x, down, up, 4, 2560)
    for suffix in (1, 5, 17, 65):
        actual = gr_project_mix_stable(x[-suffix:], down, up, 4, 2560)
        torch.testing.assert_close(actual, baseline[-suffix:], atol=0, rtol=0)
    # Check the fixed-reduction GEMM against a higher precision reference,
    # independently of the batch-invariance assertion above.
    projected = linear_stable(x[-5:], down)
    reference = torch.nn.functional.linear(x[-5:].double(), down.double()).to(x.dtype)
    torch.testing.assert_close(projected, reference, atol=1e-3, rtol=1 / 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qsa_prefill_is_invariant_to_prefix_removal():
    from minisgl.kernel.qwen4_qsa import sparse_gqa

    torch.manual_seed(37)
    q = torch.randn(17, 16, 256, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(256, 2, 256, device="cuda", dtype=q.dtype) for _ in range(2)]
    table = torch.randperm(256, device="cuda", dtype=torch.int32)[None]
    indices = torch.arange(129, device="cuda", dtype=torch.int32).repeat(17, 1)
    slots = torch.zeros(17, device="cuda", dtype=torch.int32)
    # A long prefill's last sparse tile and a hit's short suffix contain the
    # same queries/KV, but previously selected different online-softmax tiles.
    baseline = sparse_gqa(
        q, k, v, indices, table, slots, sglang_prefill_rows=2053, stable_prefill=True
    )
    actual = sparse_gqa(
        q[-5:],
        k,
        v,
        indices[-5:],
        table,
        slots[-5:],
        sglang_prefill_rows=5,
        stable_prefill=True,
    )
    torch.testing.assert_close(actual, baseline[-5:], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gdn_output_is_invariant_to_prefix_removal():
    from minisgl.kernel.qwen4_ops import gdn_output

    # This seed reaches a BF16 rounding boundary where the R=1/R=4 layouts
    # disagree despite identical inputs and per-head reduction dimensions.
    torch.manual_seed(31)
    x = torch.randn(5, 24, 128, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x) * 4
    weight = torch.randn(128, device="cuda", dtype=x.dtype)
    baseline = gdn_output(x.repeat(27, 1, 1), z.repeat(27, 1, 1), weight, stable=True)
    actual = gdn_output(x, z, weight, stable=True)
    torch.testing.assert_close(actual, baseline[-5:], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_expert_merge_is_invariant_to_prefix_removal():
    from minisgl.kernel.qwen4_ops import moe_combine

    torch.manual_seed(2)
    x = torch.randn(17, 2560, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1, 2560, device="cuda", dtype=x.dtype) * 0.03
    shared, routed = torch.randn_like(x), torch.randn_like(x)
    baseline = moe_combine(
        x.repeat(65, 1), weight, shared.repeat(65, 1), routed.repeat(65, 1), stable=True
    )
    actual = moe_combine(x, weight, shared, routed, stable=True)
    torch.testing.assert_close(actual, baseline[-17:], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [512, 513, 4097, 65536])
def test_qsa_stable_selector_handles_ties_and_large_candidate_bins(width):
    from minisgl.kernel.qwen4_ops import index_topk

    torch.manual_seed(19)
    scores = torch.randn(5, width, device="cuda")
    scores[0].zero_()
    scores[1].round_()
    lengths = torch.tensor([width, width, width - 1, 123, 0], device="cuda", dtype=torch.int32)
    expected = torch.full((5, 512), -1, device="cuda", dtype=torch.int32)
    for row, length in enumerate(lengths.tolist()):
        selected = torch.argsort(scores[row, :length], descending=True, stable=True)[:512]
        expected[row, : len(selected)] = selected.sort().values.int()
    for _ in range(3):
        actual = index_topk(scores, lengths, stable=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Row stride / batch size cannot affect the selected set or its order.
    actual = index_topk(scores[1:2], lengths[1:2], stable=True)
    torch.testing.assert_close(actual, expected[1:2], atol=0, rtol=0)
