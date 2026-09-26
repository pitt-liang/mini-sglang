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


def can_chunk_prefill(q):
    return (
        q.shape[0] >= 128
        and torch.version.cuda is not None
        and int(torch.version.cuda.split(".")[0]) >= 13
        and torch.cuda.get_device_capability(q.device)[0] == 10
    )


def prefill_delta_pool(q, k, v, log_decay, beta, state, slots, cu_seqlens):
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    output, _ = chunk_gated_delta_rule(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        log_decay.exp().contiguous(),
        beta.float().contiguous(),
        initial_state=state,
        output_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        state_indices=slots,
        cu_seqlens=cu_seqlens,
    )
    return output


def prefill_delta(q, k, v, log_decay, beta, state):
    # Blackwell's chunked kernel needs CUDA 13; the original cu128 deployment
    # and short chunks keep the portable native recurrence. State stays FP32.
    if can_chunk_prefill(q):
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        output, _ = chunk_gated_delta_rule(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            log_decay.exp().contiguous(),
            beta.float().contiguous(),
            initial_state=state[None],
            output_state=state[None],
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device),
        )
        return output
    return gated_delta_rule(q, k, v, log_decay, beta, state)
