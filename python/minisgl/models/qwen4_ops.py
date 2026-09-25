"""Plain PyTorch building blocks; also usable in CPU correctness tests."""

import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps=1e-6, group_size=None):
    shape = x.shape
    y = x.float().reshape(*shape[:-1], -1, group_size or shape[-1])
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y.reshape(shape) * (1 + weight.float())).to(x.dtype)


def l2_norm(x):
    # Match the native Transformers BF16 fallback's rounding points. An FP32
    # fused normalization is a separate numerical mode, not bit-equivalent.
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def rotary(x, positions, rotary_dim, base):
    freq = base ** (-torch.arange(0, rotary_dim, 2, device=x.device).float() / rotary_dim)
    angle = positions.float()[:, None] * freq[None, :]
    angle = torch.cat((angle, angle), dim=-1)[:, None, :]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    a = x[..., :rotary_dim]
    half = rotary_dim // 2
    rotated = a * cos + torch.cat((-a[..., half:], a[..., :half]), -1) * sin
    return torch.cat((rotated, x[..., rotary_dim:]), -1)


def causal_conv(x, weight, history, dilation=1):
    """x [T,C], weight [C,1,K], history [C,(K-1)*dilation]."""
    full = torch.cat((history, x.T), dim=-1)
    out = F.conv1d(full[None], weight, groups=x.shape[-1], dilation=dilation)
    if history.shape[-1]:
        history.copy_(full[:, -history.shape[-1] :])
    return F.silu(out[0].T)


def ngram_ids(ids, history, multipliers, sizes, offsets, heads_per_ngram, eos):
    """Int64 overflow and signed remainder match the checkpoint's hash contract."""
    tokens = torch.cat((history, ids.long()))
    positions = torch.arange(tokens.numel(), device=tokens.device)
    previous = torch.cummax(torch.where(tokens == eos, positions, -1), 0).values
    previous = F.pad(previous[:-1], (1, 0), value=-1)
    shifted = [tokens]
    for shift in range(1, multipliers.numel()):
        source = positions - shift
        valid = (source >= 0) & (source > previous)
        shifted.append(torch.where(valid, tokens[source.clamp_min(0)], eos))
    mixed = shifted[0] * multipliers[0]
    result = []
    for n in range(1, multipliers.numel()):
        mixed = torch.bitwise_xor(mixed, shifted[n] * multipliers[n])
        sl = slice((n - 1) * heads_per_ngram, n * heads_per_ngram)
        result.append(mixed[:, None].remainder(sizes[sl]) + offsets[sl])
    history.copy_(tokens[-history.numel() :])
    return torch.cat(result, -1)[-ids.numel() :]
