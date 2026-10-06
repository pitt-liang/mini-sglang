"""Paged sparse GQA and stateful indexer decode kernels."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _expand_compact(
    CHOSEN,
    COUNTS,
    LENGTHS,
    VALID,
    OUT,
    K: tl.constexpr,
    RATIO: tl.constexpr,
    NI: tl.constexpr,
    N: tl.constexpr,
    HAS_VALID: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, N)
    count, length = tl.load(COUNTS + row), tl.load(LENGTHS + row)
    selected = tl.minimum(count, K) * RATIO
    block = tl.load(CHOSEN + row * K + j // RATIO, j < selected, -1)
    token = tl.where(j < selected, block * RATIO + j % RATIO, count * RATIO + j - selected)
    live = (j < selected + length % RATIO) & ((j >= selected) | (block >= 0))
    if HAS_VALID:
        live = live & tl.load(VALID + row)
    tl.store(OUT + row * NI + j, tl.where(live, token, -1), j < NI)


def expand_compact(chosen, counts, lengths, valid, ratio):
    """Expand radix top-k's valid-prefix layout without sorting token indices."""
    rows, k = chosen.shape
    ni = k * ratio + ratio - 1
    out = torch.empty((rows, ni), device=chosen.device, dtype=torch.int32)
    _expand_compact[(rows,)](
        chosen, counts, lengths, valid, out, k, ratio, ni, tr.next_power_of_2(ni), valid is not None
    )
    return out


@tr.jit
def _gather_sparse_kv(
    K,
    V,
    IND,
    TABLE,
    SLOTS,
    PK,
    PV,
    NI: tl.constexpr,
    TS: tl.constexpr,
    STRIDE: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    col = tile * 16 + tl.arange(0, 16)
    dim = tl.arange(0, D)
    token = tl.load(IND + row * NI + col, col < NI, -1)
    slot = tl.load(SLOTS + row)
    valid = (token >= 0) & (token < TS)
    loc = tl.load(TABLE + slot * TS + token, valid, 0).to(tl.int64)
    source = (loc[:, None] * HKV + head) * D + dim[None, :]
    dest = ((row * STRIDE + col[:, None]) * HKV + head) * D + dim[None, :]
    tl.store(PK + dest, tl.load(K + source, valid[:, None], 0), col[:, None] < STRIDE)
    tl.store(PV + dest, tl.load(V + source, valid[:, None], 0), col[:, None] < STRIDE)


class SparseDecode:
    """FlashInfer post-gather decode matching SGLang's SM100 QSA path."""

    def __init__(self):
        self.buffers = {}
        self.workspace = None

    def __call__(self, q, k, v, indices, table, slots):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache

        batch, topk = indices.shape
        hkv, d = k.shape[-2:]
        pages = tr.cdiv(topk, 64)
        stride = pages * 64
        key = (batch, pages, hkv, d, q.dtype, q.device)
        if key not in self.buffers:
            shape = (batch * stride, hkv, d)
            self.buffers[key] = (
                torch.empty(shape, device=q.device, dtype=q.dtype),
                torch.empty(shape, device=q.device, dtype=q.dtype),
                torch.arange(batch * pages, device=q.device, dtype=torch.int32).view(batch, pages),
            )
        pk, pv, blocks = self.buffers[key]
        if self.workspace is None:
            self.workspace = torch.zeros(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
        _gather_sparse_kv[(batch, hkv, tr.cdiv(stride, 16))](
            k,
            v,
            indices,
            table,
            slots,
            pk,
            pv,
            topk,
            table.shape[1],
            stride,
            hkv,
            d,
            num_warps=8,
        )
        counts = (indices >= 0).sum(-1).to(torch.int32)
        kc = pk.view(-1, 64, hkv, d).permute(0, 2, 1, 3)
        vc = pv.view(-1, 64, hkv, d).permute(0, 2, 1, 3)
        out = trtllm_batch_decode_with_kv_cache(
            query=q.contiguous(),
            kv_cache=(kc, vc),
            workspace_buffer=self.workspace,
            block_tables=blocks,
            seq_lens=counts,
            max_seq_len=stride,
            bmm1_scale=d**-0.5,
            bmm2_scale=1.0,
        ).reshape_as(q)
        # The external kernel leaves zero-length output rows unwritten.
        # Padding must not propagate allocator contents/NaNs into later layers.
        return torch.where(counts[:, None, None] > 0, out, 0.0)


@tr.jit
def _index_scores(
    Q,
    K,
    TABLE,
    SLOTS,
    LENGTHS,
    VALID,
    O,
    TS: tl.constexpr,
    NB: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    RATIO: tl.constexpr,
    SKIP_DENSE: tl.constexpr = 0,
    SCALE: tl.constexpr = 1.0,
    BF16_DOT: tl.constexpr = False,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    block = tile * 32 + tl.arange(0, 32)
    count = tl.load(LENGTHS + row) // RATIO
    live = tl.load(VALID + row)
    scores = tl.full((32,), -float("inf"), tl.float32)
    if live & (count > SKIP_DENSE) & (tile * 32 < count):
        slot = tl.load(SLOTS + row)
        loc = tl.load(TABLE + slot * TS + block * RATIO, (block < count) & (block < NB), 0) // RATIO
        d, h = tl.arange(0, D), tl.arange(0, 16)
        q = tl.load(Q + (row * HQ + h[:, None]) * D + d[None, :], h[:, None] < HQ, 0)
        k = tl.load(
            K + loc[None, :] * D + d[:, None], (block[None, :] < count) & (block[None, :] < NB), 0
        )
        if BF16_DOT:
            product = tl.dot(q, k)
        else:
            product = tl.dot(q.to(tl.float32), k.to(tl.float32), input_precision="tf32x3")
        scores = tl.sum(tl.maximum(product, 0.0), 0) * SCALE
        scores = tl.where(block < count, scores, -float("inf"))
    tl.store(O + row * NB + block, scores, block < NB)


def index_scores_decode(iq, keys, table, slots, lengths, valid, ratio, budget):
    """Paged BF16 indexer GEMM; no gather or GEMM is needed below the budget."""
    nb = table.shape[1] // ratio
    out = torch.empty((iq.shape[0], nb), device=iq.device, dtype=torch.float32)
    _index_scores[(iq.shape[0], tr.cdiv(nb, 32))](
        iq,
        keys,
        table,
        slots,
        lengths,
        valid,
        out,
        table.shape[1],
        nb,
        iq.shape[1],
        iq.shape[2],
        ratio,
        budget // ratio,
        iq.shape[2] ** -0.5,
        True,
    )
    return out


@tr.jit
def _store(K, V, KC, VC, LOC, VALID, D: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(VALID + row):
        loc = tl.load(LOC + row)
        j = tl.arange(0, N)
        tl.store(KC + loc * D + j, tl.load(K + row * D + j, j < D, 0), j < D)
        tl.store(VC + loc * D + j, tl.load(V + row * D + j, j < D, 0), j < D)


def store_decode(k, v, kc, vc, locations, valid):
    k, v = k.contiguous(), v.contiguous()
    d = k.shape[1] * k.shape[2]
    _store[(k.shape[0],)](k, v, kc, vc, locations, valid, d, tr.next_power_of_2(d))


@tr.jit
def _pool_decode(
    RAW,
    PENDING,
    SLOTS,
    POSITIONS,
    VALID,
    OUT,
    D: tl.constexpr,
    RS: tl.constexpr,
    RATIO: tl.constexpr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, N)
    live = tl.load(VALID + row)
    slot = tl.load(SLOTS + row)
    past = tl.load(POSITIONS + row) % RATIO
    x = tl.load(RAW + row * RS + d, live & (d < D), 0).to(tl.float32)
    pooled = tl.full((N,), 0, tl.float32)
    if live:
        if past == RATIO - 1:
            for t in range(RATIO - 1):
                pooled += tl.load(PENDING + (slot * (RATIO - 1) + t) * D + d, d < D, 0).to(
                    tl.float32
                )
            pooled = (pooled + x) / RATIO
        else:
            tl.store(PENDING + (slot * (RATIO - 1) + past) * D + d, x, d < D)
    tl.store(OUT + row * D + d, pooled, d < D)


@tr.jit
def _store_compressed(
    X,
    CACHE,
    TABLE,
    SLOTS,
    POSITIONS,
    VALID,
    D: tl.constexpr,
    TS: tl.constexpr,
    RATIO: tl.constexpr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(POSITIONS + row)
    if tl.load(VALID + row) & (pos % RATIO == RATIO - 1):
        slot = tl.load(SLOTS + row)
        loc = tl.load(TABLE + slot * TS + pos - RATIO + 1) // RATIO
        d = tl.arange(0, N)
        tl.store(CACHE + loc * D + d, tl.load(X + row * D + d, d < D, 0), d < D)


def compress_decode(
    raw,
    weight,
    pending,
    cache,
    table,
    slots,
    positions,
    valid,
    ratio,
    rd,
    base,
    eps,
    *,
    rope_cache,
):
    from minisgl.kernel.qwen4_ops import index_norm_rope

    # Compression must use the indexer's CUDA warp reduction, not the
    # general-purpose reduction tree.
    pooled = torch.empty_like(raw, memory_format=torch.contiguous_format)
    _pool_decode[(raw.shape[0],)](
        raw,
        pending,
        slots,
        positions,
        valid,
        pooled,
        raw.shape[-1],
        raw.stride(0),
        ratio,
        tr.next_power_of_2(raw.shape[-1]),
        enable_fp_fusion=False,
    )
    starts = (positions - positions.remainder(ratio)).clamp_min(0)
    normalized = index_norm_rope(pooled[:, None], weight, starts, rope_cache, eps)
    _store_compressed[(raw.shape[0],)](
        normalized,
        cache,
        table,
        slots,
        positions,
        valid,
        raw.shape[-1],
        table.shape[1],
        ratio,
        tr.next_power_of_2(raw.shape[-1]),
    )


@tr.jit
def _sparse(
    Q,
    K,
    V,
    IND,
    TABLE,
    SLOTS,
    O,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    NI: tl.constexpr,
    TS: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
):
    row, group = tl.program_id(0), tl.program_id(1)
    h = tl.arange(0, BH)
    d = tl.arange(0, D)
    gh = HQ // HKV
    q = tl.load(Q + row * Q0 + (group * gh + h[:, None]) * Q1 + d[None, :], h[:, None] < gh, 0)
    q = (q.to(tl.float32) * (D**-0.5) * 1.4426950408).to(q.dtype)
    slot = tl.load(SLOTS + row)
    m = tl.full((BH,), -float("inf"), tl.float32)
    l = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, D), tl.float32)
    for start in range(0, tr.cdiv(NI, BN) * BN, BN):
        n = start + tl.arange(0, BN)
        tok = tl.load(IND + row * NI + n, n < NI, -1)
        loc = tl.load(TABLE + slot * TS + tok, (tok >= 0) & (tok < TS), 0)
        k = tl.load(
            K + (loc[None, :] * HKV + group) * D + d[:, None],
            (tok[None, :] >= 0) & (tok[None, :] < TS),
            0,
        )
        v = tl.load(
            V + (loc[:, None] * HKV + group) * D + d[None, :],
            (tok[:, None] >= 0) & (tok[:, None] < TS),
            0,
        )
        score = tl.dot(q, k)
        score = tl.where((tok[None, :] >= 0) & (tok[None, :] < TS), score, -float("inf"))
        new_m = tl.maximum(m, tl.max(score, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp2(m - safe_m)
        p = tl.exp2(score - safe_m[:, None])
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        l = l * alpha + tl.sum(p, 1)
        m = new_m
    result = acc / tl.where(l > 0, l, 1.0)[:, None]
    tl.store(O + (row * HQ + group * gh + h[:, None]) * D + d[None, :], result, h[:, None] < gh)


def sparse_gqa(q, k, v, indices, table, slots, *, prefill_rows, stable_prefill=False):
    """Paged sparse prefill; decode uses SparseDecode instead."""
    rows, hq, d = q.shape
    hkv = k.shape[-2]
    if stable_prefill:
        # Keep long-prefill reduction geometry for short cache-hit suffixes.
        bn, warps = 16, 1
    elif prefill_rows <= 32:
        bn, warps = 32, 8
    elif prefill_rows <= 64:
        bn, warps = 64, 8
    elif prefill_rows <= 128:
        bn, warps = 64, 4
    elif prefill_rows <= 512:
        bn, warps = 32, 4
    else:
        bn, warps = 16, 1
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _sparse[(rows, hkv)](
        q,
        k,
        v,
        indices,
        table,
        slots,
        out,
        hq,
        hkv,
        d,
        indices.shape[1],
        table.shape[1],
        max(16, tr.next_power_of_2(hq // hkv)),
        bn,
        q.stride(0),
        q.stride(1),
        num_warps=warps,
        num_stages=2,
    )
    return out
