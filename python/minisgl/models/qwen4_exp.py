"""Native BF16 text path for Qwen3.8-Flash-Next (Qwen4Exp).

Eager/reference and native graph decode, GPU-resident PLE, TP sharding.
Visual inputs, MTP, quantization and prefix snapshots are not enabled.
"""

import math

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import BaseOP, LinearColParallelMerged, LinearOProj, LinearReplicated, MoELayer
from minisgl.layers.base import OPList
from minisgl.layers.embedding import ParallelLMHead, VocabParallelEmbedding

from .base import BaseLLMModel
from .qwen4_ops import causal_conv, l2_norm, ngram_ids, rms_norm, rotary


class Weight(BaseOP):
    def __init__(self, *shape):
        self.weight = torch.empty(*shape)


class NarrowLinear(LinearReplicated):
    """Avoid BF16 narrow GEMV's batch-size-dependent reduction on Blackwell.

    GR's 4-output injection and the scalar shared-expert gate are tiny; FP32
    accumulation here avoids amplified chunk/decode drift for negligible storage.
    """

    def forward(self, x):
        return F.linear(x.float(), self.weight.float()).to(x.dtype)


class GatedResidual(BaseOP):
    def __init__(self, config, combine=True):
        self.h, self.n = config.hidden_size, config.hybrid["hc_count"]
        self.eps = config.rms_norm_eps
        width = self.h * self.n
        self.hc_norm = Weight(width)
        self.input_mix_weight_down = LinearReplicated(width, config.hybrid["hc_lowrank"], False)
        self.input_mix_weight_up = LinearReplicated(config.hybrid["hc_lowrank"], width, False)
        self.block_inject_weight = NarrowLinear(width, self.n, False) if combine else None

    def forward(self, residual):
        norm = rms_norm(residual, self.hc_norm.weight, self.eps, self.h)
        gate = F.silu(self.input_mix_weight_down.forward(norm) / self.n)
        gate = self.input_mix_weight_up.forward(gate).sigmoid()
        mixed = (gate * norm).view(-1, self.n, self.h).mean(1)
        injection = (
            None
            if self.block_inject_weight is None
            else (2 * (self.block_inject_weight.forward(norm) / self.n).sigmoid())
        )
        return mixed, injection

    def combine(self, residual, output, injection):
        return (residual.view(-1, self.n, self.h) + injection[..., None] * output[:, None]).flatten(
            1
        )


class GatedDeltaNet(BaseOP):
    def __init__(self, config, layer_id):
        c, tp = config.hybrid, get_tp_info().size
        h = config.hidden_size
        self.hk, self.hv = c["linear_num_key_heads"] // tp, c["linear_num_value_heads"] // tp
        self.dk, self.dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        self.layer_id, self.eps = layer_id, config.rms_norm_eps
        kd, vd = c["linear_num_key_heads"] * self.dk, c["linear_num_value_heads"] * self.dv
        self.in_proj_qkv = LinearColParallelMerged(h, [kd, kd, vd], False)
        self.in_proj_z = LinearColParallelMerged(h, [vd], False)
        self.in_proj_a = LinearColParallelMerged(h, [c["linear_num_value_heads"]], False)
        self.in_proj_b = LinearColParallelMerged(h, [c["linear_num_value_heads"]], False)
        self.conv1d = Weight((2 * kd + vd) // tp, 1, c["linear_conv_kernel_dim"])
        self.A_log = torch.empty(self.hv)
        self.dt_bias = torch.empty(self.hv)
        self.norm = Weight(self.dv)
        self.out_proj = LinearOProj(vd, h, False)

    def forward(self, x):
        from minisgl.kernel.qwen4 import gated_delta_rule

        ctx = get_global_ctx()
        qkv = self.in_proj_qkv.forward(x)
        z = self.in_proj_z.forward(x).view(-1, self.hv, self.dv)
        # A_log and dt_bias are converted to FP32 by the streaming loader.
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a.forward(x).float() + self.dt_bias.float()
        )
        beta = self.in_proj_b.forward(x).sigmoid()
        output = torch.empty_like(z)
        kd, vd = self.hk * self.dk, self.hv * self.dv
        for slot, _, _, a, b in ctx.batch.attn_metadata.spans:
            mixed = causal_conv(
                qkv[a:b], self.conv1d.weight, ctx.kv_cache.conv[self.layer_id][slot]
            )
            q, k, v = mixed.split((kd, kd, vd), -1)
            q, k = [l2_norm(t.reshape(-1, self.hk, self.dk)) for t in (q, k)]
            output[a:b] = gated_delta_rule(
                q,
                k,
                v.reshape(-1, self.hv, self.dv),
                decay[a:b],
                beta[a:b],
                ctx.kv_cache.recurrent[self.layer_id][slot],
            )
        y = output.float()
        y = (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        y = y * self.norm.weight  # direct scale, NOT zero-centered Gemma RMSNorm
        y = (y.float() * z.float().sigmoid()).to(x.dtype)
        return self.out_proj.forward(y.flatten(1))


class QSAIndexer(BaseOP):
    def __init__(self, config):
        c = config.hybrid
        self.heads, self.dim = c["indexer_n_heads"], c["indexer_head_dim"]
        self.index_qk_proj = LinearReplicated(
            config.hidden_size, (self.heads + 1) * self.dim, False
        )
        self.q_layernorm, self.k_layernorm = Weight(self.dim), Weight(self.dim)


class QSAAttention(BaseOP):
    def __init__(self, config, layer_id):
        tp = get_tp_info().size
        self.config, self.layer_id = config, layer_id
        self.hq, self.hkv, self.dim = (
            config.num_qo_heads // tp,
            config.num_kv_heads // tp,
            config.head_dim,
        )
        h, d = config.hidden_size, config.head_dim
        self.q_proj = LinearColParallelMerged(h, [2 * config.num_qo_heads * d], False)
        self.k_proj = LinearColParallelMerged(h, [config.num_kv_heads * d], False)
        self.v_proj = LinearColParallelMerged(h, [config.num_kv_heads * d], False)
        self.q_norm, self.k_norm = Weight(d), Weight(d)
        self.o_proj = LinearOProj(config.num_qo_heads * d, h, False)
        self.indexer = QSAIndexer(config)

    def forward(self, x):
        ctx, rc = get_global_ctx(), self.config.rotary_config
        positions = ctx.batch.positions
        q, gate = self.q_proj.forward(x).view(-1, self.hq, 2 * self.dim).chunk(2, -1)
        k = self.k_proj.forward(x).view(-1, self.hkv, self.dim)
        v = self.v_proj.forward(x).view(-1, self.hkv, self.dim)
        q = rotary(
            rms_norm(q, self.q_norm.weight, self.config.rms_norm_eps),
            positions,
            rc.rotary_dim,
            rc.base,
        )
        k = rotary(
            rms_norm(k, self.k_norm.weight, self.config.rms_norm_eps),
            positions,
            rc.rotary_dim,
            rc.base,
        )
        indexer = self.indexer
        iq, ik = self.indexer.index_qk_proj.forward(x).split(
            (indexer.heads * indexer.dim, indexer.dim), -1
        )
        iq = rms_norm(
            iq.view(-1, indexer.heads, indexer.dim),
            indexer.q_layernorm.weight,
            self.config.rms_norm_eps,
        )
        iq = rotary(iq, positions, rc.rotary_dim, rc.base)
        out = ctx.attn_backend.qsa(q, k, v, iq, ik, indexer.k_layernorm.weight, self.layer_id)
        return self.o_proj.forward((out * gate.sigmoid()).flatten(1))


class SharedMLP(BaseOP):
    def __init__(self, hidden, intermediate):
        self.gate_proj = LinearColParallelMerged(hidden, [intermediate], False)
        self.up_proj = LinearColParallelMerged(hidden, [intermediate], False)
        self.down_proj = LinearOProj(intermediate, hidden, False)

    def forward(self, x):
        return self.down_proj.forward(F.silu(self.gate_proj.forward(x)) * self.up_proj.forward(x))


class SparseMoE(BaseOP):
    def __init__(self, config):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, False)
        self.experts = MoELayer(
            config.num_experts,
            config.num_experts_per_tok,
            config.hidden_size,
            config.moe_intermediate_size,
            config.norm_topk_prob,
        )
        self.shared_expert = SharedMLP(
            config.hidden_size, config.hybrid["shared_expert_intermediate_size"]
        )
        self.shared_expert_gate = NarrowLinear(config.hidden_size, 1, False)

    def forward(self, x):
        # The existing fused MoE backend reuses x as its output buffer.
        shared = self.shared_expert.forward(x) * self.shared_expert_gate.forward(x).sigmoid()
        routed = self.experts.forward(x, self.gate.forward(x))
        return routed + shared


class NGramTable(VocabParallelEmbedding):
    def forward(self, ids):
        # PLE's 160 BF16 values/row are not supported by the generic warp-copy
        # embedding kernel (320 bytes is not a multiple of 128 bytes).
        start, count = self.vocab_range
        local = ids.long() - start
        valid = (local >= 0) & (local < count)
        out = F.embedding(local.clamp(0, count - 1), self.weight)
        out = out * valid[:, None]
        return self._comm.all_reduce(out) if self.tp_size > 1 else out


class NGramEmbedding(BaseOP):
    def __init__(self, config, ple_index):
        from sympy import nextprime

        c = config.hybrid
        self.heads, self.n = c["heads_per_ngram"], c["ngram_size"]
        total_heads = self.heads * (self.n - 1)
        prime, total = c["ngram_vocab_size_base"] - 1, 0
        for i in range((ple_index + 1) * total_heads):
            prime = int(nextprime(prime))
            if i >= ple_index * total_heads:
                total += prime
        divisor = c["make_ngram_vocab_size_divisible_by"]
        padded = math.ceil(total / divisor) * divisor
        self.layer_multipliers = torch.empty(self.n, dtype=torch.int64)
        self.ngram_heads_offsets = torch.empty(total_heads, dtype=torch.int64)
        self.ngram_heads_vocab_sizes = torch.empty(total_heads, dtype=torch.int64)
        self.ngram_embedding = NGramTable(padded, c["ple_embed_dim"] // total_heads)


class PLE(BaseOP):
    def __init__(self, config, layer_id):
        c = config.hybrid
        self.layer_id, self.config = layer_id, config
        self.ple_embedding = NGramEmbedding(config, c["ple_layer_ids"].index(layer_id + 1))
        width = c["hc_count"] * config.hidden_size
        self.key_proj = LinearReplicated(c["ple_embed_dim"], width, False)
        self.value_proj = LinearReplicated(c["ple_embed_dim"], config.hidden_size, False)
        self.norm_key, self.norm_query, self.norm_conv = Weight(width), Weight(width), Weight(width)
        self.conv1d = Weight(width, 1, c["ple_conv_kernel_size"])

    def forward(self, residual):
        ctx, cfg = get_global_ctx(), self.config
        c, pe = cfg.hybrid, self.ple_embedding
        ids = []
        for slot, _, _, a, b in ctx.batch.attn_metadata.spans:
            ids.append(
                ngram_ids(
                    ctx.batch.input_ids[a:b],
                    ctx.kv_cache.ple_history[self.layer_id][slot],
                    pe.layer_multipliers,
                    pe.ngram_heads_vocab_sizes,
                    pe.ngram_heads_offsets,
                    pe.heads,
                    c["eos_token_id"],
                )
            )
        hashed = torch.cat(ids)
        embeddings = pe.ngram_embedding.forward(hashed.flatten()).view(residual.shape[0], -1)
        h, n, eps = cfg.hidden_size, c["hc_count"], cfg.rms_norm_eps
        from minisgl.models import qwen4_fast

        norm = rms_norm
        if qwen4_fast.SGLANG_NUMERICS and qwen4_fast.enabled(residual):
            from minisgl.kernel.qwen4_sglang import grouped_norm

            def norm(x, w, eps, h):
                return grouped_norm(x, w, h, eps)

        key = norm(self.key_proj.forward(embeddings), self.norm_key.weight, eps, h).view(-1, n, h)
        query = norm(residual, self.norm_query.weight, eps, h).view(-1, n, h)
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(h)
        gate = gate.sign() * gate.abs().clamp_min(1e-6).sqrt()
        value = (gate.sigmoid() * self.value_proj.forward(embeddings)[:, None]).flatten(1)
        normed = norm(value, self.norm_conv.weight, eps, h)
        out = torch.empty_like(value)
        for slot, _, _, a, b in ctx.batch.attn_metadata.spans:
            out[a:b] = causal_conv(
                normed[a:b],
                self.conv1d.weight,
                ctx.kv_cache.ple_conv[self.layer_id][slot],
                c["ngram_size"],
            )
        return value + out


class DecoderLayer(BaseOP):
    def __init__(self, config, layer_id):
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.linear_attn = (
            GatedDeltaNet(config, layer_id)
            if config.hybrid["layer_types"][layer_id] == "linear_attention"
            else None
        )
        self.self_attn = QSAAttention(config, layer_id) if self.linear_attn is None else None
        self.mlp = SparseMoE(config)
        self.ple = PLE(config, layer_id) if layer_id + 1 in config.hybrid["ple_layer_ids"] else None

    def forward(self, residual):
        if self.ple is not None:
            residual = residual + self.ple.forward(residual)
        x, gate = self.attn_hyper_connection.forward(residual)
        x = (self.linear_attn or self.self_attn).forward(x)
        residual = self.attn_hyper_connection.combine(residual, x, gate)
        x, gate = self.mlp_hyper_connection.forward(residual)
        return self.mlp_hyper_connection.combine(residual, self.mlp.forward(x), gate)


class Qwen4Text(BaseOP):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([DecoderLayer(config, i) for i in range(config.num_layers)])
        self.hyper_connection_mixer = GatedResidual(config, combine=False)


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config):
        tp = get_tp_info().size
        c = config.hybrid
        for heads in (
            config.num_qo_heads,
            config.num_kv_heads,
            c["linear_num_key_heads"],
            c["linear_num_value_heads"],
        ):
            if heads % tp:
                raise ValueError(
                    "Qwen4 baseline requires TP to divide all head counts (checkpoint: TP=1/2)"
                )
        if c["indexer_kv_heads"] != 1 or c["output_gate_type"] != "sigmoid":
            raise ValueError("Unsupported Qwen4 indexer/gate configuration")
        self.config = config
        self.model = Qwen4Text(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        from .qwen4_optimized import install

        install(self)

    def load_weights(self, model_path, device):
        from .qwen4_weight import load_qwen4_weights

        load_qwen4_weights(self, model_path, device)
        from .qwen4_optimized import pack_weights

        pack_weights(self)

    def forward(self):
        ctx = get_global_ctx()
        if ctx.batch.is_prefill:
            for slot, cached, _, _, _ in ctx.batch.attn_metadata.spans:
                if cached == 0:
                    ctx.kv_cache.reset(slot)
        residual = self.model.embed_tokens.forward(ctx.batch.input_ids).repeat(
            1, self.config.hybrid["hc_count"]
        )
        for layer in self.model.layers.op_list:
            residual = layer.forward(residual)
        hidden, _ = self.model.hyper_connection_mixer.forward(residual)
        return self.lm_head.forward(hidden)
