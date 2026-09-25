# SPDX-License-Identifier: Apache-2.0
# SGLang-derived numerical contracts; provenance in _qwen4_fla/README.md.
"""Qwen4 numerical primitives matching SGLang's FP32 fusion contract.

These kernels do not import SGLang. Independent operator comparisons live in
tests/models/test_qwen4_sglang.py; the old Torch/BF16-step reference is a different contract.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl

from .utils import load_jit


@lru_cache(None)
def rope_cache(device, rotary_dim, base, max_position):
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    angles = torch.arange(max_position, device=device).float()[:, None] * inv[None, :]
    return torch.cat((angles.cos(), angles.sin()), -1)


@tr.jit
def _norm_rope(
    X,
    W,
    C,
    P,
    O,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    STRIDE: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, D)
    x = tl.load(X + row * STRIDE + j).to(tl.float32)
    w = tl.load(W + j).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    y = (x * scale * (w + 1.0)).to(O.dtype.element_ty)
    tl.store(O + row * D + j, y, j >= RD)
    half = tl.arange(0, RD // 2)
    x1 = tl.load(X + row * STRIDE + half).to(tl.float32)
    x2 = tl.load(X + row * STRIDE + half + RD // 2).to(tl.float32)
    w1 = tl.load(W + half).to(tl.float32)
    w2 = tl.load(W + half + RD // 2).to(tl.float32)
    y1 = (x1 * scale * (w1 + 1.0)).to(O.dtype.element_ty).to(tl.float32)
    y2 = (x2 * scale * (w2 + 1.0)).to(O.dtype.element_ty).to(tl.float32)
    pos = tl.load(P + row // H).to(tl.int64)
    co = tl.load(C + pos * RD + half)
    si = tl.load(C + pos * RD + half + RD // 2)
    tl.store(O + row * D + half, y1 * co - y2 * si)
    tl.store(O + row * D + half + RD // 2, y2 * co + y1 * si)


def norm_rope(x, weight, positions, cache, eps=1e-6):
    x = x.contiguous()
    out = torch.empty_like(x)
    _, heads, dim = x.shape
    _norm_rope[(x.shape[0] * heads,)](
        x, weight, cache, positions, out, heads, dim, cache.shape[-1], dim, eps
    )
    return out


@lru_cache(None)
def _index_module(dim):
    return load_jit(
        "qwen4_index_norm",
        str(dim),
        cuda_files=["qwen4_hc.cu"],
        cuda_wrappers=[("run", f"Qwen4Index<{dim}>::run")],
    )


@lru_cache(None)
def _topk_module():
    return load_jit(
        "qwen4_topk", cuda_files=["qwen4_topk.cu"], cuda_wrappers=[("run", "Qwen4FastTopK::run")]
    )


def index_topk(scores, lengths):
    if scores.dtype != torch.float32 or scores.stride(-1) != 1:
        raise ValueError("QSA top-k requires contiguous FP32 score rows")
    starts = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.int32)
    out = torch.empty((scores.shape[0], 512), device=scores.device, dtype=torch.int32)
    _topk_module().run(scores, starts, out, lengths.to(torch.int32).contiguous())
    return out


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


def index_norm_rope(x, weight, positions, cache, eps=1e-6):
    if x.dtype != torch.bfloat16 or x.shape[-1] not in (64, 128, 256):
        raise ValueError("Qwen4 indexer requires BF16 and head dim 64/128/256")
    x = x.contiguous()
    out = torch.empty_like(x)
    _index_module(x.shape[-1]).run(x, weight, cache, positions.long().contiguous(), out, eps)
    return out


@tr.jit
def _sigmoid_mul(X, G, O, SIZE: tl.constexpr, N: tl.constexpr):
    j = tl.program_id(0) * N + tl.arange(0, N)
    x = tl.load(X + j, j < SIZE, 0).to(tl.float32)
    g = tl.load(G + j, j < SIZE, 0).to(tl.float32)
    tl.store(O + j, x * tl.sigmoid(g), j < SIZE)


def sigmoid_mul(x, gate):
    x, gate = x.contiguous(), gate.contiguous()
    out = torch.empty_like(x)
    _sigmoid_mul[(tr.cdiv(x.numel(), 256),)](x, gate, out, x.numel(), 256)
    return out


@tr.jit
def _moe_combine(X, W, SHARED, ROUTED, OUT, H: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, N)
    x = tl.load(X + row * H + col, col < H, 0).to(tl.float32)
    w = tl.load(W + col, col < H, 0).to(tl.float32)
    shared = tl.load(SHARED + row * H + col, col < H, 0).to(tl.float32)
    routed = tl.load(ROUTED + row * H + col, col < H, 0).to(tl.float32)
    gate = tl.sigmoid(tl.sum(x * w, 0))
    tl.store(OUT + row * H + col, gate * shared + routed, col < H)


def moe_combine(x, gate_weight, shared, routed):
    """FP32 gate/merge before the single TP reduction, as in SGLang."""
    rows, h = x.shape
    warps = max(min(tr.next_power_of_2(tr.cdiv(h, 256)), 32), 4)
    if rows >= 1024:
        warps = min(warps, 8)
    out = torch.empty_like(routed)
    _moe_combine[(rows,)](
        x, gate_weight, shared, routed, out, h, tr.next_power_of_2(h), num_warps=warps
    )
    return out


@tr.jit
def _router(LOGITS, WEIGHTS, IDS, E: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.arange(0, E)[None, :]
    logits = tl.load(LOGITS + row * E + expert).to(tl.float32)
    ex = tl.exp(logits - tl.max(logits, 1)[:, None])
    probabilities = ex / tl.sum(ex, 1)[:, None]
    current = tl.where(logits == logits, logits, -1e30)
    slot = tl.arange(0, BK)[None, :]
    values = tl.full((1, BK), 0.0, tl.float32)
    ids = tl.full((1, BK), 0, tl.int32)
    remaining = tl.full((1, E), True, tl.int1)
    for k in tl.static_range(K):
        best = tl.max(current, 1)[:, None]
        winner = tl.min(tl.where(remaining & (current == best), expert, E + 1), 1)[:, None]
        value = tl.sum(tl.where(expert == winner, probabilities, 0.0), 1)[:, None]
        values = tl.where(slot == k, value, values)
        ids = tl.where(slot == k, winner, ids)
        remaining = remaining & (expert != winner)
        current = tl.where(remaining, current, -float("inf"))
    total = tl.sum(values, 1)[:, None]
    values /= tl.where(total > 0.0, total, 1.0)
    tl.store(WEIGHTS + row * K + slot, values, slot < K)
    tl.store(IDS + row * K + slot, ids, slot < K)


def router(logits, top_k):
    if logits.shape[1] != 512 or not logits.is_contiguous():
        raise ValueError("Qwen4 aligned router requires contiguous 512-expert logits")
    weights = torch.empty((logits.shape[0], top_k), device=logits.device, dtype=torch.float32)
    ids = torch.empty_like(weights, dtype=torch.int32)
    _router[(logits.shape[0],)](
        logits, weights, ids, 512, top_k, tr.next_power_of_2(top_k), num_warps=1
    )
    return weights, ids


@lru_cache(None)
def _hc_module(group_size, hc):
    return load_jit(
        "qwen4_hc",
        str(group_size),
        str(hc),
        cuda_files=["qwen4_hc.cu"],
        cuda_wrappers=[
            ("norm", f"Qwen4HC<{group_size}, {hc}>::norm"),
            ("combine", f"Qwen4HC<{group_size}, {hc}>::combine"),
            ("combine_split", f"Qwen4HC<{group_size}, {hc}>::combine_split"),
        ],
    )


def grouped_norm(x, weight, group_size, eps=1e-6):
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype:
        raise ValueError("Qwen4 HC kernels require BF16")
    if group_size % 512 or x.shape[-1] != weight.numel() or x.shape[-1] % group_size:
        raise ValueError("Invalid grouped HC norm shape")
    x = x.contiguous()
    out = torch.empty_like(x)
    _hc_module(group_size, x.shape[-1] // group_size).norm(x, weight.contiguous(), out, eps)
    return out


def gr_combine(residual, output, normed, weight):
    h, hc = output.shape[-1], weight.shape[0]
    if h % 512 or residual.shape != normed.shape or residual.shape != (output.shape[0], hc * h):
        raise ValueError("Invalid Qwen4 GR combine shape")
    if any(
        x.dtype != torch.bfloat16 or not x.is_contiguous()
        for x in (residual, output, normed, weight)
    ):
        raise ValueError("Qwen4 GR combine requires contiguous BF16 tensors")
    out = torch.empty_like(residual)
    if output.shape[0] <= 32:
        partials = torch.empty((output.shape[0], 8, hc), device=output.device, dtype=torch.float32)
        _hc_module(h, hc).combine_split(residual, output, normed, weight, out, partials)
    else:
        _hc_module(h, hc).combine(residual, output, normed, weight, out)
    return out


@tr.jit
def _gdn_output(X, Z, W, O, M, D: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, D)
    offsets = rows[:, None] * D + cols[None, :]
    x = tl.load(X + offsets, rows[:, None] < M, 0).to(tl.float32)
    z = tl.load(Z + offsets, rows[:, None] < M, 0).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    var = tl.sum(x * x, 1) / D
    y = (x * tl.rsqrt(var + 1e-6)[:, None]) * w[None, :]
    y *= tl.sigmoid(z)
    tl.store(O + offsets, y, rows[:, None] < M)


def gdn_output(x, z, weight, eps=1e-6):
    if eps != 1e-6:
        raise ValueError("Qwen4 checkpoint expects rms_norm_eps=1e-6")
    x, z = x.contiguous(), z.contiguous()
    rows, d = x.numel() // x.shape[-1], x.shape[-1]
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    block_rows = min(tr.next_power_of_2(tr.cdiv(rows, 2 * sm)), 4)
    out = torch.empty_like(x)
    _gdn_output[(tr.cdiv(rows, block_rows),)](x, z, weight, out, rows, d, block_rows, num_warps=1)
    return out


@tr.jit
def _ple_gate(G, V, O, H: tl.constexpr, HC: tl.constexpr, N: tl.constexpr):
    group = tl.program_id(0)
    d = tl.arange(0, N)
    g = tl.load(G + group).to(tl.float32)
    root = tl.sqrt(tl.maximum(tl.abs(g), 1e-6)).to(tl.bfloat16).to(tl.float32)
    sign = tl.where(g > 0, 1.0, tl.where(g < 0, -1.0, 0.0))
    transformed = (root * sign).to(tl.bfloat16).to(tl.float32)
    activated = tl.sigmoid(transformed).to(tl.bfloat16).to(tl.float32)
    value = tl.load(V + (group // HC) * H + d, d < H, 0).to(tl.float32)
    tl.store(O + group * H + d, activated * value, d < H)


def ple_gate_value(gate, value):
    rows, hc = gate.shape[:2]
    h = value.shape[-1]
    out = torch.empty((rows, hc * h), device=value.device, dtype=value.dtype)
    _ple_gate[(rows * hc,)](gate, value, out, h, hc, tr.next_power_of_2(h))
    return out


@tr.jit
def _ple_conv_state(X, S, SLOTS, VALID, O, C: tl.constexpr, L: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * 64 + tl.arange(0, 64)
    j = tl.arange(0, N)
    slot, live = tl.load(SLOTS + row), tl.load(VALID + row)
    old = tl.load(
        S + (slot * C + c[:, None]) * L + j[None, :], live & (c[:, None] < C) & (j[None, :] < L), 0
    )
    x = tl.load(X + row * C + c, live & (c < C), 0)
    full = tl.where(j[None, :] == L, x[:, None], old)
    tl.store(
        O + (row * C + c[:, None]) * (L + 1) + j[None, :],
        full,
        (c[:, None] < C) & (j[None, :] <= L),
    )
    shifted = tl.gather(full, tl.broadcast_to(tl.minimum(j + 1, N - 1)[None, :], (64, N)), 1)
    tl.store(
        S + (slot * C + c[:, None]) * L + j[None, :],
        shifted,
        live & (c[:, None] < C) & (j[None, :] < L),
    )


def ple_conv_decode(x, weight, state, slots, valid, dilation):
    # Match SGLang's native depthwise Conv1D + SiLU. In particular, the
    # legacy Triton decode's different sum/exp order can flip a BF16 ulp.
    rows, channels = x.shape
    length = state.shape[-1]
    conv_input = torch.empty((rows, channels, length + 1), device=x.device, dtype=x.dtype)
    _ple_conv_state[(rows, tr.cdiv(channels, 64))](
        x,
        state,
        slots,
        valid,
        conv_input,
        channels,
        length,
        tr.next_power_of_2(length + 1),
    )
    out = F.silu(F.conv1d(conv_input, weight, groups=channels, dilation=dilation).squeeze(-1))
    return torch.where(valid[:, None], out, 0)


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


@tr.jit
def _gr_down_finish(P, T, M: tl.constexpr, R: tl.constexpr, HC: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    r = tl.arange(0, N)
    # SGLang's cluster split-K owner adds peers in rank order, not a tree.
    acc = tl.load(P + row * R + r, r < R, 0)
    for peer in tl.static_range(1, 16):
        acc += tl.load(P + (peer * M + row) * R + r, r < R, 0)
    acc *= 1.0 / HC
    tl.store(T + row * R + r, acc * tl.sigmoid(acc), r < R)


@tr.jit
def _gr_up_mix(T, W, X, O, M: tl.constexpr, H: tl.constexpr, R: tl.constexpr, HC: tl.constexpr):
    j = tl.program_id(0) * 16 + tl.arange(0, 16)
    m = tl.arange(0, 16)
    rr = tl.arange(0, 64)
    total = tl.full((16, 16), 0.0, tl.float32)
    for branch in tl.static_range(HC):
        acc = tl.zeros((16, 16), tl.float32)
        for start in range(tr.cdiv(R, 64)):
            r = start * 64 + rr
            a = tl.load(T + m[:, None] * R + r[None, :], (m[:, None] < M) & (r[None, :] < R), 0)
            b = tl.load(
                W + (branch * H + j[None, :]) * R + r[:, None],
                (j[None, :] < H) & (r[:, None] < R),
                0,
            )
            acc = tl.dot(a, b, acc)
        x = tl.load(
            X + m[:, None] * HC * H + branch * H + j[None, :],
            (m[:, None] < M) & (j[None, :] < H),
            0,
        ).to(tl.float32)
        total += tl.sigmoid(acc) * x
    tl.store(
        O + m[:, None] * H + j[None, :], total * (1.0 / HC), (m[:, None] < M) & (j[None, :] < H)
    )


def gr_project_mix(x, down, up, hc):
    from .qwen4_fused import _gr_down

    m, k = x.shape
    if m > 16:
        return torch.cat([gr_project_mix(chunk, down, up, hc) for chunk in x.split(16)])
    r, h = down.shape[0], k // hc
    partial = torch.empty((16, m, r), device=x.device, dtype=torch.float32)
    middle = torch.empty((m, r), device=x.device, dtype=x.dtype)
    out = torch.empty((m, h), device=x.device, dtype=x.dtype)
    _gr_down[(tr.cdiv(r, 32), 16)](x, down, partial, m, k, r, 16, num_warps=4)
    _gr_down_finish[(m,)](partial, middle, m, r, hc, tr.next_power_of_2(r))
    _gr_up_mix[(tr.cdiv(h, 16),)](middle, up, x, out, m, h, r, hc, num_warps=4)
    return out


@torch.compile
def gr_project_mix_large(x, down, up, hc, hidden):
    # Keep the same GEMM boundaries and compiler-visible expression as the
    # SGLang >24-row path. Eager BF16 pointwise steps are not equivalent.
    gate = F.silu(F.linear(x, down) / hc)
    gate = torch.sigmoid(F.linear(gate, up)).unflatten(-1, (hc, hidden))
    return (gate * x.unflatten(-1, (hc, hidden))).mean(-2)
