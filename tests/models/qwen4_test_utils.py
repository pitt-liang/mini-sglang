# SPDX-License-Identifier: Apache-2.0
"""Controlled numerical-test utilities; never imported by the model runtime."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _stable_index_topk(S, L, O, WIDTH: tl.constexpr, STRIDE: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, N)
    length = tl.load(L + row)
    score = tl.load(S + row * STRIDE + col, col < WIDTH, 0)
    # QSA scores are nonnegative. Lowest index wins an exact score tie.
    key = (score.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | (N - 1 - col).to(tl.uint64)
    key = tl.where((col < WIDTH) & (col < length), key, 0)
    ordered = tl.sort(key, descending=True)
    index = (N - 1 - (ordered & 0xFFFFFFFF)).to(tl.int32)
    # Canonicalize the selected set before BF16 attention reductions.
    index = tl.where((col < 512) & (col < length), index, 0x7FFFFFFF)
    index = tl.sort(index, descending=False)
    tl.store(O + row * 512 + col, tl.where(col < length, index, -1), col < 512)


def index_topk_deterministic(scores, lengths):
    """Controlled accuracy-test policy, independent of the stock radix selector."""
    n = max(512, tr.next_power_of_2(scores.shape[1]))
    if n > 8192:
        raise ValueError("Deterministic QSA comparison supports at most 8192 compressed keys")
    out = torch.empty((scores.shape[0], 512), device=scores.device, dtype=torch.int32)
    _stable_index_topk[(scores.shape[0],)](
        scores, lengths, out, scores.shape[1], scores.stride(0), n, num_warps=8
    )
    return out


@tr.jit
def _l2(X, Y, T, D: tl.constexpr):
    p = tl.make_block_ptr(X, (T, D), (D, 1), (tl.program_id(0) * 16, 0), (16, D), (1, 0))
    x = tl.load(p, boundary_check=(0, 1)).to(tl.float32)
    y = x / tl.sqrt(tl.sum(x * x, 1) + 1e-6)[:, None]
    out = tl.make_block_ptr(Y, (T, D), (D, 1), (tl.program_id(0) * 16, 0), (16, D), (1, 0))
    tl.store(out, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


def l2(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    d = x.shape[-1]
    rows = x.numel() // d
    _l2[(tr.cdiv(rows, 16),)](x, out, rows, d, num_warps=8, num_stages=3)
    return out
