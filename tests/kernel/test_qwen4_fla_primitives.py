"""Guard reachable shape variants while pruning unrelated vendored entry points."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dim", [128, 1024])
@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_l2norm_retains_both_dimension_dispatches(dim, output_dtype):
    from minisgl.kernel._qwen4_fla.l2norm import l2norm_fwd

    torch.manual_seed(17)
    x = torch.randn(1, 17, 2, dim, device="cuda", dtype=torch.bfloat16)
    actual = l2norm_fwd(x, output_dtype=output_dtype)
    expected = x.float() / (x.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
    expected = expected.to(output_dtype or x.dtype)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1 / 128)
    assert actual.shape == x.shape


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_chunk_cumsum_retains_scalar_and_vector_dispatch(vector, reverse):
    from minisgl.kernel._qwen4_fla.cumsum import chunk_local_cumsum

    torch.manual_seed(23)
    lengths = [5, 65, 133]
    shape = (1, sum(lengths), 2) + ((16,) if vector else ())
    x = torch.rand(shape, device="cuda")
    cu = torch.tensor([0, 5, 70, 203], device="cuda", dtype=torch.int32)
    actual = chunk_local_cumsum(x, 64, reverse=reverse, scale=0.5, cu_seqlens=cu)
    expected = torch.empty_like(x)
    start = 0
    for length in lengths:
        for offset in range(0, length, 64):
            a, b = start + offset, start + min(offset + 64, length)
            chunk = x[:, a:b]
            if reverse:
                result = chunk.flip(1).cumsum(1).flip(1)
            else:
                result = chunk.cumsum(1)
            expected[:, a:b] = result * 0.5
        start += length
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
