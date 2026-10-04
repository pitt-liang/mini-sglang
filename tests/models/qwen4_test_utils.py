# SPDX-License-Identifier: Apache-2.0
"""Controlled numerical-test utilities; never imported by the model runtime."""

import torch
import triton as tr
import triton.language as tl
from minisgl.models import Qwen4Config, RotaryConfig


def make_qwen4_config(**overrides) -> Qwen4Config:
    """Small, complete typed config; tests opt into environment parsing explicitly."""
    values = {
        "num_layers": 4,
        "num_qo_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 8,
        "hidden_size": 32,
        "vocab_size": 64,
        "intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "rotary_config": RotaryConfig(8, 4, 128, 10000.0, None),
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "norm_topk_prob": True,
        "model_type": "qwen4_exp_text",
        "architectures": ["Qwen4ExpForCausalLM"],
        "layer_types": ("linear_attention",) * 3 + ("full_attention",),
        "hc_count": 4,
        "hc_lowrank": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "output_gate_type": "sigmoid",
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 8,
        "indexer_compress_ratio": 4,
        "indexer_budget": 8,
        "shared_expert_intermediate_size": 64,
        "ple_layer_ids": (2,),
        "ple_embed_dim": 32,
        "ple_conv_kernel_size": 4,
        "heads_per_ngram": 2,
        "ngram_size": 3,
        "ngram_vocab_size_base": 97,
        "make_ngram_vocab_size_divisible_by": 1,
        "split_ngram_parts": 2,
        "eos_token_id": 17,
    }
    values.update(overrides)
    return Qwen4Config(**values)


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


def index_topk_deterministic(scores, lengths, **_kwargs):
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
