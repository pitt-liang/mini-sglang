"""Qwen4 recurrent kernel. State is FP32 [value_head, value_dim, key_dim].

This deliberately small eager baseline supports arbitrary prefill chunk lengths.
It does not require FlashInfer's CUDA-13-only Blackwell prefill implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _delta(
    Q,
    K,
    V,
    G,
    B,
    S,
    O,
    T: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    h = tl.program_id(0)
    vi = tl.program_id(1) * BV + tl.arange(0, BV)
    ki = tl.arange(0, BK)
    kh = h // (HV // HK)
    offsets = (h * DV + vi[:, None]) * DK + ki[None, :]
    mask = (vi[:, None] < DV) & (ki[None, :] < DK)
    state = tl.load(S + offsets, mask, 0)
    for t in range(T):
        q = tl.load(Q + (t * HK + kh) * DK + ki, ki < DK, 0).to(tl.float32)
        k = tl.load(K + (t * HK + kh) * DK + ki, ki < DK, 0).to(tl.float32)
        v = tl.load(V + (t * HV + h) * DV + vi, vi < DV, 0).to(tl.float32)
        alpha = tl.exp(tl.load(G + t * HV + h).to(tl.float32))
        beta = tl.load(B + t * HV + h).to(tl.float32)
        state *= alpha
        delta = (v - tl.sum(state * k[None, :], 1)) * beta
        state += delta[:, None] * k[None, :]
        out = tl.sum(state * q[None, :], 1) * (DK**-0.5)
        tl.store(O + (t * HV + h) * DV + vi, out, vi < DV)
    tl.store(S + offsets, state, mask)


def gated_delta_rule(q, k, v, log_decay, beta, state):
    """Q/K are already L2-normalized. Updates state in place; returns [T,HV,DV]."""
    t, hk, dk = q.shape
    hv, dv = v.shape[1:]
    assert q.is_cuda and state.dtype == torch.float32
    assert hv % hk == 0 and state.shape == (hv, dv, dk)
    q, k, v, log_decay, beta = [x.contiguous() for x in (q, k, v, log_decay, beta)]
    out = torch.empty_like(v)
    _delta[(hv, triton.cdiv(dv, 16))](
        q,
        k,
        v,
        log_decay,
        beta,
        state,
        out,
        t,
        hk,
        hv,
        dk,
        dv,
        triton.next_power_of_2(dk),
        16,
        num_warps=4,
    )
    return out


@triton.jit
def _delta_batch(
    Q,
    K,
    V,
    G,
    B,
    S,
    O,
    SLOTS,
    VALID,
    HK: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    row, h = tl.program_id(0), tl.program_id(1)
    vi = tl.program_id(2) * BV + tl.arange(0, BV)
    ki = tl.arange(0, BK)
    if tl.load(VALID + row):
        slot = tl.load(SLOTS + row)
        kh = h // (HV // HK)
        offsets = ((slot * HV + h) * DV + vi[:, None]) * DK + ki[None, :]
        mask = (vi[:, None] < DV) & (ki[None, :] < DK)
        state = tl.load(S + offsets, mask, 0)
        q = tl.load(Q + (row * HK + kh) * DK + ki, ki < DK, 0).to(tl.float32)
        k = tl.load(K + (row * HK + kh) * DK + ki, ki < DK, 0).to(tl.float32)
        v = tl.load(V + (row * HV + h) * DV + vi, vi < DV, 0).to(tl.float32)
        state *= tl.exp(tl.load(G + row * HV + h).to(tl.float32))
        beta = tl.load(B + row * HV + h).to(tl.float32)
        delta = (v - tl.sum(state * k[None, :], 1)) * beta
        state += delta[:, None] * k[None, :]
        out = tl.sum(state * q[None, :], 1) * (DK**-0.5)
        tl.store(S + offsets, state, mask)
    else:
        out = tl.full((BV,), 0.0, tl.float32)
    tl.store(O + (row * HV + h) * DV + vi, out, vi < DV)


def gated_delta_decode(q, k, v, log_decay, beta, state_pool, slots, valid):
    rows, hk, dk = q.shape
    hv, dv = v.shape[1:]
    out = torch.empty_like(v)
    q, k, v = [x.contiguous() for x in (q, k, v)]
    _delta_batch[(rows, hv, triton.cdiv(dv, 16))](
        q,
        k,
        v,
        log_decay,
        beta,
        state_pool,
        out,
        slots,
        valid,
        hk,
        hv,
        dk,
        dv,
        triton.next_power_of_2(dk),
        16,
        num_warps=4,
    )
    return out
