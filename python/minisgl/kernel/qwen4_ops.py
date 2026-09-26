# SPDX-License-Identifier: Apache-2.0
"""Qwen4 production primitives and explicit legacy fallbacks.

Unprefixed primitives implement the aligned numerical contract or are shared by
both paths. legacy_* primitives retain the earlier BF16 reference contract.
SGLang-derived operator provenance is in _qwen4_fla/README.md.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .utils import load_jit


# Shared helpers and legacy BF16 primitives.
@tr.jit
def _legacy_gdn_gates(
    A, B, LOG, DT, G, BETA, H: tl.constexpr, STRIDE: tl.constexpr, N: tl.constexpr
):
    row = tl.program_id(0)
    j = tl.arange(0, N)
    a = tl.load(A + row * STRIDE + j, j < H, 0).to(tl.float32)
    b = tl.load(B + row * STRIDE + j, j < H, 0).to(tl.float32)
    log = tl.load(LOG + j, j < H, 0).to(tl.float32)
    dt = tl.load(DT + j, j < H, 0).to(tl.float32)
    x = a + dt
    softplus = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
    tl.store(G + row * H + j, -libdevice.exp(log) * softplus, j < H)
    tl.store(BETA + row * H + j, 1.0 / (1.0 + libdevice.exp(-b)), j < H)


def legacy_gdn_gates(a, b, log_a, dt):
    decay = torch.empty(a.shape, device=a.device, dtype=torch.float32)
    beta = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    _legacy_gdn_gates[(a.shape[0],)](
        a,
        b,
        log_a,
        dt,
        decay,
        beta,
        a.shape[1],
        a.stride(0),
        tr.next_power_of_2(a.shape[1]),
        enable_fp_fusion=False,
    )
    return decay, beta


@tr.jit
def _gr_down(X, W, P, M: tl.constexpr, K: tl.constexpr, R: tl.constexpr, SK: tl.constexpr):
    n, split = tl.program_id(0), tl.program_id(1)
    m = tl.arange(0, 16)
    r = n * 32 + tl.arange(0, 32)
    k = split * (K // SK) + tl.arange(0, 128)
    acc = tl.zeros((16, 32), tl.float32)
    for _ in range(K // SK // 128):
        x = tl.load(X + m[:, None] * K + k[None, :], m[:, None] < M, 0)
        w = tl.load(W + r[None, :] * K + k[:, None], r[None, :] < R, 0)
        acc = tl.dot(x, w, acc)
        k += 128
    tl.store(
        P + (split * M + m[:, None]) * R + r[None, :], acc, (m[:, None] < M) & (r[None, :] < R)
    )


@tr.jit
def _legacy_gr_down_finish(
    P, T, M: tl.constexpr, R: tl.constexpr, SK: tl.constexpr, HC: tl.constexpr, N: tl.constexpr
):
    row = tl.program_id(0)
    r = tl.arange(0, N)
    s = tl.arange(0, SK)
    partial = tl.load(P + (s[:, None] * M + row) * R + r[None, :], r[None, :] < R, 0)
    raw = tl.sum(partial, 0).to(T.dtype.element_ty).to(tl.float32)
    raw = (raw / HC).to(T.dtype.element_ty).to(tl.float32)
    tl.store(T + row * R + r, raw * tl.sigmoid(raw), r < R)


@tr.jit
def _legacy_gr_up_mix(
    T, W, X, O, M: tl.constexpr, H: tl.constexpr, R: tl.constexpr, HC: tl.constexpr
):
    j = tl.program_id(0) * 16 + tl.arange(0, 16)
    m = tl.arange(0, 16)
    rr = tl.arange(0, 64)
    branch = tl.arange(0, HC)
    col = tl.reshape(branch[:, None] * H + j[None, :], (HC * 16,))
    mask_col = tl.reshape(tl.broadcast_to(j[None, :] < H, (HC, 16)), (HC * 16,))
    acc = tl.zeros((16, HC * 16), tl.float32)
    for start in range(tr.cdiv(R, 64)):
        r = start * 64 + rr
        t = tl.load(T + m[:, None] * R + r[None, :], (m[:, None] < M) & (r[None, :] < R), 0)
        w = tl.load(W + col[None, :] * R + r[:, None], mask_col[None, :] & (r[:, None] < R), 0)
        acc = tl.dot(t, w, acc)
    acc = acc.to(X.dtype.element_ty).to(tl.float32)
    gate = tl.sigmoid(acc).to(X.dtype.element_ty).to(tl.float32)
    x = tl.load(X + m[:, None] * HC * H + col[None, :], (m[:, None] < M) & mask_col[None, :], 0).to(
        tl.float32
    )
    product = (gate * x).to(X.dtype.element_ty).to(tl.float32)
    total = tl.sum(tl.reshape(product, (16, HC, 16)), 1)
    tl.store(O + m[:, None] * H + j[None, :], total / HC, (m[:, None] < M) & (j[None, :] < H))


def legacy_gr_project_mix(x, down, up, hc):
    m, k = x.shape
    r, h, sk = down.shape[0], k // hc, 16
    partial = torch.empty((sk, m, r), device=x.device, dtype=torch.float32)
    middle = torch.empty((m, r), device=x.device, dtype=x.dtype)
    out = torch.empty((m, h), device=x.device, dtype=x.dtype)
    _gr_down[(tr.cdiv(r, 32), sk)](x, down, partial, m, k, r, sk, num_warps=4)
    _legacy_gr_down_finish[(m,)](
        partial, middle, m, r, sk, hc, tr.next_power_of_2(r), enable_fp_fusion=False
    )
    _legacy_gr_up_mix[(tr.cdiv(h, 16),)](
        middle, up, x, out, m, h, r, hc, num_warps=4, enable_fp_fusion=False
    )
    return out


@tr.jit
def _legacy_shared_act(X, GU, W, ACT, GATE, H: tl.constexpr, I: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, N)
    x = tl.load(X + row * H + j, j < H, 0).to(tl.float32)
    w = tl.load(W + j, j < H, 0).to(tl.float32)
    dot = tl.sum(x * w, 0).to(X.dtype.element_ty).to(tl.float32)
    tl.store(GATE + row, tl.sigmoid(dot))
    g = tl.load(GU + row * 2 * I + j, j < I, 0).to(tl.float32)
    u = tl.load(GU + row * 2 * I + I + j, j < I, 0).to(tl.float32)
    a = (g * tl.sigmoid(g)).to(X.dtype.element_ty).to(tl.float32)
    tl.store(ACT + row * I + j, a * u, j < I)


def legacy_shared_activation(x, gate_up, gate_weight):
    i = gate_up.shape[-1] // 2
    act = torch.empty((x.shape[0], i), device=x.device, dtype=x.dtype)
    gate = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
    _legacy_shared_act[(x.shape[0],)](
        x,
        gate_up,
        gate_weight,
        act,
        gate,
        x.shape[-1],
        i,
        tr.next_power_of_2(max(x.shape[-1], i)),
        enable_fp_fusion=False,
    )
    return act, gate


@tr.jit
def _legacy_moe_combine(P, G, O, H: tl.constexpr, SIZE: tl.constexpr, N: tl.constexpr):
    j = tl.program_id(0) * N + tl.arange(0, N)
    row, col = j // H, j % H
    routed = tl.load(P + row * 2 * H + col, j < SIZE, 0).to(tl.float32)
    shared = tl.load(P + row * 2 * H + H + col, j < SIZE, 0).to(tl.float32)
    gate = tl.load(G + row, j < SIZE, 0).to(tl.float32)
    y = (shared * gate).to(P.dtype.element_ty).to(tl.float32)
    tl.store(O + j, routed + y, j < SIZE)


def legacy_moe_combine(packed, gate):
    out = torch.empty(
        (packed.shape[0], packed.shape[1] // 2), device=packed.device, dtype=packed.dtype
    )
    _legacy_moe_combine[(tr.cdiv(out.numel(), 256),)](
        packed, gate, out, out.shape[1], out.numel(), 256, enable_fp_fusion=False
    )
    return out


@tr.jit
def _legacy_norm(
    X,
    W,
    Y,
    POS,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    RD: tl.constexpr,
    BASE: tl.constexpr,
    EPS: tl.constexpr,
    GROUPS: tl.constexpr,
    ZERO: tl.constexpr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, N)
    x = tl.load(X + row * STRIDE + j, j < D, 0).to(tl.float32)
    w = tl.load(W + (row % GROUPS) * D + j, j < D, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    y = (x * scale * (w + (1.0 if ZERO else 0.0))).to(Y.dtype.element_ty)
    if RD > 0:
        peer = tl.where(j < RD // 2, j + RD // 2, j - RD // 2)
        px = tl.load(X + row * STRIDE + peer, j < RD, 0).to(tl.float32)
        pw = tl.load(W + (row % GROUPS) * D + peer, j < RD, 0).to(tl.float32)
        py = (px * scale * (pw + (1.0 if ZERO else 0.0))).to(Y.dtype.element_ty)
        pos = tl.load(POS + row // HEADS).to(tl.float32)
        angle = pos * tl.exp(-tl.log(BASE) * (2.0 * (j % (RD // 2)) / RD))
        co = tl.cos(angle).to(Y.dtype.element_ty).to(tl.float32)
        si = tl.sin(angle).to(Y.dtype.element_ty).to(tl.float32)
        a = (y.to(tl.float32) * co).to(Y.dtype.element_ty).to(tl.float32)
        b = (
            (py.to(tl.float32) * tl.where(j < RD // 2, -si, si))
            .to(Y.dtype.element_ty)
            .to(tl.float32)
        )
        y = tl.where(j < RD, a + b, y.to(tl.float32)).to(Y.dtype.element_ty)
    tl.store(Y + row * D + j, y, j < D)


def legacy_norm(x, weight, eps=1e-6, group_size=None, positions=None, rotary_dim=0, base=10000.0):
    d = group_size or x.shape[-1]
    flat = x.reshape(-1, d)
    out = torch.empty(flat.shape, device=x.device, dtype=x.dtype)
    groups = weight.numel() // d
    heads = x.shape[-2] if x.ndim == 3 else 1
    _legacy_norm[(flat.shape[0],)](
        flat,
        weight,
        out,
        positions,
        d,
        flat.stride(0),
        heads,
        rotary_dim,
        float(base),
        eps,
        groups,
        True,
        tr.next_power_of_2(d),
        enable_fp_fusion=False,
    )
    return out.view(x.shape)


@tr.jit
def _legacy_silu_div(X, Y, SIZE: tl.constexpr, DIV: tl.constexpr, N: tl.constexpr):
    j = tl.program_id(0) * N + tl.arange(0, N)
    x = (tl.load(X + j, j < SIZE, 0).to(tl.float32) / DIV).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + j, x * tl.sigmoid(x), j < SIZE)


def legacy_silu_div(x, divisor):
    out = torch.empty_like(x)
    _legacy_silu_div[(tr.cdiv(x.numel(), 256),)](x, out, x.numel(), divisor, 256)
    return out


@tr.jit
def _legacy_mix(G, X, O, H: tl.constexpr, HC: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, N)
    acc = tl.full((N,), 0.0, tl.float32)
    for h in range(HC):
        off = (row * HC + h) * H + j
        g = tl.sigmoid(tl.load(G + off, j < H, 0).to(tl.float32)).to(X.dtype.element_ty)
        x = tl.load(X + off, j < H, 0)
        acc += (g.to(tl.float32) * x.to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    tl.store(O + row * H + j, acc / HC, j < H)


def legacy_gr_mix(gates, normed, hc):
    h = normed.shape[-1] // hc
    out = torch.empty((normed.shape[0], h), device=normed.device, dtype=normed.dtype)
    _legacy_mix[(out.shape[0],)](gates, normed, out, h, hc, tr.next_power_of_2(h))
    return out


@tr.jit
def _legacy_combine(
    R, X, NORM, W, O, H: tl.constexpr, HC: tl.constexpr, NW: tl.constexpr, NH: tl.constexpr
):
    row, branch = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, NW)
    v = tl.load(NORM + row * H * HC + j, j < H * HC, 0).to(tl.float32)
    w = tl.load(W + branch * H * HC + j, j < H * HC, 0).to(tl.float32)
    dot = tl.sum(v * w, 0).to(R.dtype.element_ty).to(tl.float32)
    dot = (dot / HC).to(R.dtype.element_ty).to(tl.float32)
    gate = (2.0 * tl.sigmoid(dot).to(R.dtype.element_ty).to(tl.float32)).to(R.dtype.element_ty)
    k = tl.arange(0, NH)
    off = (row * HC + branch) * H + k
    r = tl.load(R + off, k < H, 0).to(tl.float32)
    x = tl.load(X + row * H + k, k < H, 0).to(tl.float32)
    inject = (gate.to(tl.float32) * x).to(R.dtype.element_ty).to(tl.float32)
    tl.store(O + off, r + inject, k < H)


def legacy_gr_combine(residual, output, normed, weight):
    hc, h = weight.shape[0], output.shape[-1]
    out = torch.empty_like(residual)
    _legacy_combine[(out.shape[0], hc)](
        residual,
        output,
        normed,
        weight,
        out,
        h,
        hc,
        tr.next_power_of_2(h * hc),
        tr.next_power_of_2(h),
        num_warps=8,
        enable_fp_fusion=False,
    )
    return out


@tr.jit
def _legacy_conv(
    X,
    W,
    S,
    SLOTS,
    VALID,
    O,
    C: tl.constexpr,
    K: tl.constexpr,
    DIL: tl.constexpr,
    NH: tl.constexpr,
    NC: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(VALID + row):
        slot = tl.load(SLOTS + row)
        c = tl.program_id(1) * NC + tl.arange(0, NC)
        hs = tl.arange(0, NH)
        hist = tl.load(
            S + (slot * C + c[:, None]) * ((K - 1) * DIL) + hs[None, :],
            (c[:, None] < C) & (hs[None, :] < (K - 1) * DIL),
            0,
        )
        x = tl.load(X + row * C + c, c < C, 0).to(tl.float32)
        acc = x * tl.load(W + c * K + K - 1, c < C, 0).to(tl.float32)
        for i in range(K - 1):
            old = tl.sum(tl.where(hs[None, :] == i * DIL, hist.to(tl.float32), 0.0), 1)
            acc += old * tl.load(W + c * K + i, c < C, 0).to(tl.float32)
        acc = acc.to(X.dtype.element_ty).to(tl.float32)
        tl.store(O + row * C + c, acc * tl.sigmoid(acc), c < C)
        shifted = tl.load(
            S + (slot * C + c[:, None]) * ((K - 1) * DIL) + hs[None, :] + 1,
            (c[:, None] < C) & (hs[None, :] < (K - 1) * DIL - 1),
            0,
        )
        shifted = tl.where(hs[None, :] == (K - 1) * DIL - 1, x[:, None], shifted.to(tl.float32))
        tl.store(
            S + (slot * C + c[:, None]) * ((K - 1) * DIL) + hs[None, :],
            shifted,
            (c[:, None] < C) & (hs[None, :] < (K - 1) * DIL),
        )
    else:
        c = tl.program_id(1) * NC + tl.arange(0, NC)
        tl.store(O + row * C + c, 0.0, c < C)


def legacy_conv_decode(x, weight, state, slots, valid, dilation=1):
    out = torch.empty_like(x)
    k = weight.shape[-1]
    _legacy_conv[(x.shape[0], tr.cdiv(x.shape[1], 64))](
        x,
        weight,
        state,
        slots,
        valid,
        out,
        x.shape[1],
        k,
        dilation,
        tr.next_power_of_2((k - 1) * dilation),
        64,
        enable_fp_fusion=False,
    )
    return out


@tr.jit
def _legacy_l2(X, O, D: tl.constexpr, SX: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, N)
    x = tl.load(X + row * SX + j, j < D, 0).to(tl.float32)
    xx = (x * x).to(X.dtype.element_ty).to(tl.float32)
    den = tl.sum(xx, 0).to(X.dtype.element_ty).to(tl.float32)
    den = (den + 1e-6).to(X.dtype.element_ty).to(tl.float32)
    inv = tl.rsqrt(den).to(X.dtype.element_ty).to(tl.float32)
    tl.store(O + row * D + j, x * inv, j < D)


def legacy_l2(x):
    flat = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(flat)
    _legacy_l2[(flat.shape[0],)](
        flat,
        out,
        flat.shape[-1],
        flat.stride(0),
        tr.next_power_of_2(flat.shape[-1]),
        enable_fp_fusion=False,
    )
    return out.view(x.shape)


@tr.jit
def _legacy_gdn_out(X, Z, W, O, D: tl.constexpr, EPS: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, N)
    x = tl.load(X + row * D + j, j < D, 0).to(tl.float32)
    z = tl.load(Z + row * D + j, j < D, 0).to(tl.float32)
    w = tl.load(W + j, j < D, 0).to(tl.float32)
    y = (x * tl.rsqrt(tl.sum(x * x, 0) / D + EPS)).to(X.dtype.element_ty).to(tl.float32)
    y = (y * w).to(X.dtype.element_ty).to(tl.float32)
    tl.store(O + row * D + j, y * tl.sigmoid(z), j < D)


def legacy_gdn_output(x, z, weight, eps):
    out = torch.empty_like(x)
    _legacy_gdn_out[(x.numel() // x.shape[-1],)](
        x, z, weight, out, x.shape[-1], eps, tr.next_power_of_2(x.shape[-1]), enable_fp_fusion=False
    )
    return out


@tr.jit
def _hash(
    IDS,
    HISTORY,
    MUL,
    SIZE,
    OFFSET,
    SLOTS,
    VALID,
    O,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    EOS: tl.constexpr,
    NH: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, NH)
    if tl.load(VALID + row):
        slot = tl.load(SLOTS + row)
        token = tl.load(IDS + row).to(tl.int64)
        mix = token * tl.load(MUL)
        alive = True
        # Read old history before any writer changes it.
        for shift in tl.static_range(1, N):
            old = tl.load(HISTORY + slot * (N - 1) + (N - 1 - shift))
            shifted = tl.where(alive, old, EOS)
            alive = alive & (old != EOS)
            mix = mix ^ (shifted * tl.load(MUL + shift))
            index = (shift - 1) * HEADS + h
            size = tl.load(SIZE + index, h < HEADS, 1)
            rem = mix % size
            rem = tl.where(rem < 0, rem + size, rem)
            off = tl.load(OFFSET + index, h < HEADS, 0)
            tl.store(O + row * (N - 1) * HEADS + index, rem + off, h < HEADS)
        for j in tl.static_range(N - 2):
            old = tl.load(HISTORY + slot * (N - 1) + j + 1)
            tl.store(HISTORY + slot * (N - 1) + j, old)
        tl.store(HISTORY + slot * (N - 1) + N - 2, token)
    else:
        for shift in tl.static_range(N - 1):
            tl.store(O + row * (N - 1) * HEADS + shift * HEADS + h, 0, h < HEADS)


def hash_decode(ids, history, multipliers, sizes, offsets, heads, eos, slots, valid):
    n = multipliers.numel()
    out = torch.empty((ids.numel(), (n - 1) * heads), device=ids.device, dtype=torch.int64)
    _hash[(ids.numel(),)](
        ids,
        history,
        multipliers,
        sizes,
        offsets,
        slots,
        valid,
        out,
        n,
        heads,
        eos,
        tr.next_power_of_2(heads),
    )
    return out


# Aligned production primitives.
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
