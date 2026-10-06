# SPDX-License-Identifier: Apache-2.0
"""Qwen4 production primitives.

The numerical contract follows the selected SGLang production operators.
SGLang-derived operator provenance is in _qwen4_fla/README.md.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl

from .utils import load_jit

# Shared production helpers (Torch where a fused replacement is not yet available).


def causal_conv(x, weight, history, dilation=1):
    """x [T,C], weight [C,1,K], history [C,(K-1)*dilation]."""
    full = torch.cat((history, x.T), dim=-1)
    out = F.conv1d(full[None], weight, groups=x.shape[-1], dilation=dilation)
    if history.shape[-1]:
        history.copy_(full[:, -history.shape[-1] :])
    return F.silu(out[0].T)


# PLE host/device lookup.
def validate_ple_host_device(device):
    """Fail before allocating a large pinned table on an unsupported device."""
    from cuda.bindings import runtime

    index = device.index if device.index is not None else torch.cuda.current_device()
    for attr in ("cudaDevAttrCanMapHostMemory", "cudaDevAttrUnifiedAddressing"):
        error, supported = runtime.cudaDeviceGetAttribute(
            getattr(runtime.cudaDeviceAttr, attr), index
        )
        if error != runtime.cudaError_t.cudaSuccess or not supported:
            raise RuntimeError(f"Pinned PLE requires CUDA {attr} on {device}: {error}")


def ple_host_pointer(weight, device):
    from cuda.bindings import runtime

    if weight.device.type != "cpu" or not weight.is_pinned():
        raise ValueError("PLE host table must be allocated in pinned CPU memory")
    validate_ple_host_device(device)
    error, pointer = runtime.cudaHostGetDevicePointer(weight.data_ptr(), 0)
    if error != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Cannot map pinned PLE table into CUDA: {error}")
    return pointer


@tr.jit
def _ple_lookup(
    WEIGHT,
    IDS,
    OUT,
    START: tl.constexpr,
    ROWS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    local = tl.load(IDS + row).to(tl.int64) - START
    col = tl.arange(0, BLOCK)
    weight = WEIGHT.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    value = tl.load(
        weight + local * DIM + col, (local >= 0) & (local < ROWS) & (col < DIM), other=0
    )
    tl.store(OUT + row * DIM + col, value, col < DIM)


def ple_lookup(pointer, ids, out, start, rows):
    """Local BF16 rows only; the caller owns both the mapped table and output."""
    if ids.dtype != torch.int64 or not ids.is_cuda or not ids.is_contiguous():
        raise ValueError("PLE lookup IDs must be contiguous CUDA int64")
    if (
        out.dtype != torch.bfloat16
        or out.device != ids.device
        or not out.is_contiguous()
        or out.ndim != 2
        or out.shape[0] != ids.numel()
    ):
        raise ValueError("PLE lookup output must be contiguous CUDA BF16 [ids, width]")
    if ids.numel():
        _ple_lookup[(ids.numel(),)](
            pointer,
            ids,
            out,
            start,
            rows,
            out.shape[1],
            tr.next_power_of_2(out.shape[1]),
            num_warps=4,
        )
    return out


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


def _hash_decode(ids, history, multipliers, sizes, offsets, heads, eos, slots, valid):
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


@tr.jit
def _hash_prefill(
    IDS,
    HISTORY,
    MUL,
    SIZE,
    OFFSET,
    SLOTS,
    VALID,
    CU,
    O,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    EOS: tl.constexpr,
    BT: tl.constexpr,
    NH: tl.constexpr,
):
    row = tl.program_id(0)
    start, end = tl.load(CU + row), tl.load(CU + row + 1)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    h = tl.arange(0, NH)
    mask = start + t < end
    if tl.load(VALID + row):
        slot = tl.load(SLOTS + row)
        token = tl.load(IDS + start + t, mask, 0).to(tl.int64)
        mix = token * tl.load(MUL)
        alive = tl.full((BT,), True, tl.int1)
        for shift in tl.static_range(1, N):
            p = t - shift
            local = tl.load(IDS + start + p, mask & (p >= 0), 0).to(tl.int64)
            old = tl.load(HISTORY + slot * (N - 1) + N - 1 + p, mask & (p < 0), 0)
            previous = tl.where(p >= 0, local, old)
            shifted = tl.where(alive, previous, EOS)
            alive = alive & (previous != EOS)
            mix = mix ^ (shifted * tl.load(MUL + shift))
            index = (shift - 1) * HEADS + h
            size = tl.load(SIZE + index, h < HEADS, 1)
            rem = mix[:, None] % size[None, :]
            rem = tl.where(rem < 0, rem + size[None, :], rem)
            off = tl.load(OFFSET + index, h < HEADS, 0)
            tl.store(
                O + (start + t[:, None]) * (N - 1) * HEADS + index[None, :],
                rem + off[None, :],
                mask[:, None] & (h[None, :] < HEADS),
            )
    else:
        for shift in tl.static_range(N - 1):
            tl.store(
                O + (start + t[:, None]) * (N - 1) * HEADS + shift * HEADS + h[None, :],
                0,
                mask[:, None] & (h[None, :] < HEADS),
            )


@tr.jit
def _hash_update_history(IDS, HISTORY, SLOTS, VALID, CU, W: tl.constexpr, BW: tl.constexpr):
    row = tl.program_id(0)
    start, end = tl.load(CU + row), tl.load(CU + row + 1)
    length = end - start
    if tl.load(VALID + row) & (length > 0):
        slot = tl.load(SLOTS + row)
        j = tl.arange(0, BW)
        p = length - W + j
        token = tl.load(IDS + start + p, (j < W) & (p >= 0), 0).to(tl.int64)
        old = tl.load(HISTORY + slot * W + W + p, (j < W) & (p < 0), 0)
        # One writer per request, after *all* hash readers on the same stream.
        tl.store(HISTORY + slot * W + j, tl.where(p >= 0, token, old), j < W)


def ngram_ids(
    ids: torch.Tensor,
    history: torch.Tensor,
    multipliers: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
    heads: int,
    eos: int,
    slots: torch.Tensor,
    valid: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    max_extend_len: int | None = None,
) -> torch.Tensor:
    """Packed PLE hashes; update each request's raw history exactly once.

    Decode retains the one-token fused fast path. Prefill computes against a
    read-only initial history, then commits tails in a separate ordered launch.
    Valid requests must own distinct history slots; padding may share slots.
    """
    if cu_seqlens is None:
        return _hash_decode(ids, history, multipliers, sizes, offsets, heads, eos, slots, valid)
    if max_extend_len is None or max_extend_len < 0:
        raise ValueError("Prefill hash requires nonnegative max_extend_len")
    n = multipliers.numel()
    out = torch.empty((ids.numel(), (n - 1) * heads), device=ids.device, dtype=torch.int64)
    rows = cu_seqlens.numel() - 1
    if rows and max_extend_len:
        _hash_prefill[(rows, tr.cdiv(max_extend_len, 32))](
            ids,
            history,
            multipliers,
            sizes,
            offsets,
            slots,
            valid,
            cu_seqlens,
            out,
            n,
            heads,
            eos,
            32,
            tr.next_power_of_2(heads),
            num_warps=4,
        )
        _hash_update_history[(rows,)](
            ids,
            history,
            slots,
            valid,
            cu_seqlens,
            n - 1,
            tr.next_power_of_2(n - 1),
            num_warps=1,
        )
    return out


# Production primitives.
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
        "qwen4_topk",
        cuda_files=["qwen4_topk.cu"],
        cuda_wrappers=[("run", "Qwen4FastTopK::run"), ("stable", "Qwen4FastTopK::stable")],
    )


def index_topk(scores, lengths, *, stable=False):
    if scores.dtype != torch.float32 or scores.stride(-1) != 1:
        raise ValueError("QSA top-k requires contiguous FP32 score rows")
    starts = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.int32)
    out = torch.empty((scores.shape[0], 512), device=scores.device, dtype=torch.int32)
    select = _topk_module().stable if stable else _topk_module().run
    select(scores, starts, out, lengths.to(torch.int32).contiguous())
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


def moe_combine(x, gate_weight, shared, routed, *, stable=False):
    """FP32 gate/merge before the single TP reduction, as in SGLang."""
    rows, h = x.shape
    warps = max(min(tr.next_power_of_2(tr.cdiv(h, 256)), 32), 4)
    if stable or rows >= 1024:
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
        raise ValueError("Qwen4 router requires contiguous 512-expert logits")
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


def gdn_output(x, z, weight, eps=1e-6, *, stable=False):
    if eps != 1e-6:
        raise ValueError("Qwen4 checkpoint expects rms_norm_eps=1e-6")
    x, z = x.contiguous(), z.contiguous()
    rows, d = x.numel() // x.shape[-1], x.shape[-1]
    sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    # Different row layouts can change compiler contraction/rounding even
    # without changing the mathematical per-head reduction.
    block_rows = 4 if stable else min(tr.next_power_of_2(tr.cdiv(rows, 2 * sm)), 4)
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
    # Preserve SGLang's native depthwise Conv1D + SiLU rounding points.
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


@tr.jit
def _linear_stable(
    X, W, Y, M, N: tl.constexpr, K: tl.constexpr, SX: tl.constexpr, BM: tl.constexpr
):
    # Fixed MMA/reduction geometry: a suffix must not select a different
    # reduction algorithm merely because its prefill has fewer token rows.
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 64)
    acc = tl.zeros((BM, 64), tl.float32)
    for start in range(tl.cdiv(K, 64)):
        kk = start * 64 + k
        a = tl.load(X + m[:, None] * SX + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0)
        b = tl.load(W + n[None, :] * K + kk[:, None], (n[None, :] < N) & (kk[:, None] < K), 0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def linear_stable(x, weight):
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype:
        raise ValueError("Stable Qwen4 projection requires BF16")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("Invalid Qwen4 projection shape")
    x, weight = x.contiguous(), weight.contiguous()
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    bm = 64 if n >= 512 else 16
    _linear_stable[(tr.cdiv(m, bm), tr.cdiv(n, 64))](
        x, weight, out, m, n, k, x.stride(0), bm, num_warps=4, num_stages=3
    )
    return out


@tr.jit
def _gr_stable_activation(X, Y, SIZE, HC: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    x = tl.load(X + i, i < SIZE, 0).to(tl.float32) / HC
    tl.store(Y + i, x * tl.sigmoid(x), i < SIZE)


@tr.jit
def _gr_stable_finish(X, G, Y, SIZE, H: tl.constexpr, HC: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    row, col = i // H, i % H
    result = tl.full((256,), 0, tl.float32)
    for branch in tl.static_range(HC):
        offset = row * HC * H + branch * H + col
        gate = tl.load(G + offset, i < SIZE, 0).to(tl.float32)
        x = tl.load(X + offset, i < SIZE, 0).to(tl.float32)
        result += tl.sigmoid(gate) * x
    tl.store(Y + i, result / HC, i < SIZE)


def gr_project_mix_stable(x, down, up, hc, hidden):
    projected = linear_stable(x, down)
    gate = torch.empty_like(projected)
    _gr_stable_activation[(tr.cdiv(gate.numel(), 256),)](
        projected, gate, gate.numel(), hc, enable_fp_fusion=False
    )
    gate = linear_stable(gate, up)
    out = torch.empty((x.shape[0], hidden), dtype=x.dtype, device=x.device)
    _gr_stable_finish[(tr.cdiv(out.numel(), 256),)](
        x, gate, out, out.numel(), hidden, hc, enable_fp_fusion=False
    )
    return out
