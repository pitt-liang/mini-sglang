"""Native BF16 text path for Qwen3.8-Flash-Next (Qwen4Exp).

Explicit per-model policy, native graph decode, GPU/UVA PLE lookup overlap and TP.
Each layer owns its eager/legacy/aligned forwards; classes are never replaced.
Visual inputs, MTP and quantization are not enabled.
"""

import math

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.kernel import qwen4_ops as ops
from minisgl.kernel.qwen4 import (
    can_chunk_prefill,
    gated_delta_decode,
    prefill_delta,
    prefill_delta_pool,
)
from minisgl.layers import BaseOP, LinearColParallelMerged, LinearOProj, LinearReplicated, MoELayer
from minisgl.layers.base import OPList
from minisgl.layers.embedding import ParallelLMHead, VocabParallelEmbedding

from .base import BaseLLMModel
from .qwen4_ops import causal_conv, l2_norm, ngram_ids, rms_norm, rotary


def _prefill_linear(ctx):
    if getattr(ctx.kv_cache, "stable_numerics", False) and ctx.batch.is_prefill:
        return ops.linear_stable
    return F.linear


def _output_projection(layer, x, linear):
    if linear is F.linear:
        return layer.forward(x)
    output = linear(x, layer.weight)
    return layer._comm.all_reduce(output) if layer._tp_size > 1 else output


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
        self.runtime = config.qwen4_runtime
        self.h, self.n = config.hidden_size, config.hybrid["hc_count"]
        self.eps = config.rms_norm_eps
        width = self.h * self.n
        self.hc_norm = Weight(width)
        self.input_mix_weight_down = LinearReplicated(width, config.hybrid["hc_lowrank"], False)
        self.input_mix_weight_up = LinearReplicated(config.hybrid["hc_lowrank"], width, False)
        self.block_inject_weight = NarrowLinear(width, self.n, False) if combine else None

    def _forward_eager(self, residual):
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

    def _combine_eager(self, residual, output, injection):
        return (residual.view(-1, self.n, self.h) + injection[..., None] * output[:, None]).flatten(
            1
        )

    def forward(self, residual):
        if not self.runtime.enabled(residual):
            return self._forward_eager(residual)
        if self.runtime.aligned:
            normed = ops.grouped_norm(residual, self.hc_norm.weight, self.h, self.eps)
            # Cache hits change GEMM row counts. Both cuBLAS and the fused
            # small-row HC path can then choose different reduction orders.
            ctx = get_global_ctx()
            stable_prefill = (
                getattr(ctx.kv_cache, "stable_numerics", False) and ctx.batch.is_prefill
            )
            if stable_prefill:
                mixed = ops.gr_project_mix_stable(
                    normed,
                    self.input_mix_weight_down.weight,
                    self.input_mix_weight_up.weight,
                    self.n,
                    self.h,
                )
            elif normed.shape[0] <= 24:
                if hasattr(self, "_sg_up"):
                    from minisgl.kernel._qwen4_cute.hc_mix import hc_mix

                    mixed = hc_mix(
                        normed, self.input_mix_weight_down.weight, self._sg_up, self.n, self.h
                    )
                else:
                    mixed = ops.gr_project_mix(
                        normed,
                        self.input_mix_weight_down.weight,
                        self.input_mix_weight_up.weight,
                        self.n,
                    )
            else:
                mixed = ops.gr_project_mix_large(
                    normed,
                    self.input_mix_weight_down.weight,
                    self.input_mix_weight_up.weight,
                    self.n,
                    self.h,
                )
            return mixed, normed
        normed = ops.legacy_norm(residual, self.hc_norm.weight, self.eps, self.h)
        if normed.shape[0] <= 16 and normed.shape[1] % 2048 == 0:
            mixed = ops.legacy_gr_project_mix(
                normed, self.input_mix_weight_down.weight, self.input_mix_weight_up.weight, self.n
            )
        else:
            gate = ops.legacy_silu_div(self.input_mix_weight_down.forward(normed), self.n)
            gate = self.input_mix_weight_up.forward(gate)
            mixed = ops.legacy_gr_mix(gate, normed, self.n)
        return mixed, normed

    def combine(self, residual, output, injection):
        if not self.runtime.enabled(residual):
            return self._combine_eager(residual, output, injection)
        if self.runtime.aligned:
            return ops.gr_combine(residual, output, injection, self.block_inject_weight.weight)
        return ops.legacy_gr_combine(residual, output, injection, self.block_inject_weight.weight)


class GatedDeltaNet(BaseOP):
    def __init__(self, config, layer_id):
        self.runtime = config.qwen4_runtime
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

    def _forward_eager(self, x):
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

    def forward(self, x):
        if not self.runtime.enabled(x):
            return self._forward_eager(x)
        if self.runtime.aligned:
            return self.forward_aligned(x)
        ctx = get_global_ctx()
        qkv, z, a, b = F.linear(x, self._packed).split(self._widths, -1)
        qkv = qkv.contiguous()
        z = z.reshape(-1, self.hv, self.dv).contiguous()
        decay, beta = ops.legacy_gdn_gates(a, b, self.A_log, self.dt_bias)
        kd, vd = self.hk * self.dk, self.hv * self.dv
        m = ctx.batch.attn_metadata
        if ctx.batch.is_decode:
            mixed = ops.legacy_conv_decode(
                qkv, self.conv1d.weight, ctx.kv_cache.conv[self.layer_id], m.slots, m.valid
            )
            q, k, v = mixed.split((kd, kd, vd), -1)
            q, k = [ops.legacy_l2(t.reshape(-1, self.hk, self.dk)) for t in (q, k)]
            output = gated_delta_decode(
                q,
                k,
                v.reshape(-1, self.hv, self.dv),
                decay,
                beta,
                ctx.kv_cache.recurrent[self.layer_id],
                m.slots,
                m.valid,
            )
        elif can_chunk_prefill(qkv):
            mixed = torch.empty_like(qkv)
            for slot, _, _, a, b in m.spans:
                mixed[a:b] = causal_conv(
                    qkv[a:b], self.conv1d.weight, ctx.kv_cache.conv[self.layer_id][slot]
                )
            q, k, v = mixed.split((kd, kd, vd), -1)
            q, k = [ops.legacy_l2(t.reshape(-1, self.hk, self.dk)) for t in (q, k)]
            output = prefill_delta_pool(
                q,
                k,
                v.reshape(-1, self.hv, self.dv),
                decay,
                beta,
                ctx.kv_cache.recurrent[self.layer_id],
                m.slots,
                m.cu_seqlens,
            )
        else:
            output = torch.empty_like(z)
            for slot, _, _, a, b in m.spans:
                mixed = causal_conv(
                    qkv[a:b], self.conv1d.weight, ctx.kv_cache.conv[self.layer_id][slot]
                )
                q, k, v = mixed.split((kd, kd, vd), -1)
                q, k = [ops.legacy_l2(t.reshape(-1, self.hk, self.dk)) for t in (q, k)]
                output[a:b] = prefill_delta(
                    q,
                    k,
                    v.reshape(-1, self.hv, self.dv),
                    decay[a:b],
                    beta[a:b],
                    ctx.kv_cache.recurrent[self.layer_id][slot],
                )
        y = ops.legacy_gdn_output(output, z, self.norm.weight, self.eps)
        return self.out_proj.forward(y.flatten(1))

    def forward_aligned(self, x):
        from minisgl.kernel._qwen4_fla.causal_conv import causal_conv1d_fn, causal_conv1d_update
        from minisgl.kernel._qwen4_fla.chunk import chunk_gated_delta_rule
        from minisgl.kernel._qwen4_fla.fused_gdn_gating import fused_gdn_gating
        from minisgl.kernel._qwen4_fla.packed_decode import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        ctx = get_global_ctx()
        m, pool = ctx.batch.attn_metadata, ctx.kv_cache
        # Preserve the long-prefill QKV/Z and B/A GEMM boundaries even for a
        # short cache-hit suffix. Decode retains the single packed GEMM.
        stable_prefill = getattr(pool, "stable_numerics", False) and ctx.batch.is_prefill
        linear = _prefill_linear(ctx)
        if not stable_prefill and x.shape[0] <= 1024:
            projected = F.linear(x, self._packed)
        else:
            cut = self._widths[0] + self._widths[1]
            projected = torch.cat(
                (linear(x, self._packed[:cut]), linear(x, self._packed[cut:])), -1
            )
        qkv, z, b, a = projected.split(self._widths, -1)
        z = z.reshape(-1, self.hv, self.dv)
        state = pool.recurrent[self.layer_id]
        weight = self.conv1d.weight[:, 0]
        if ctx.batch.is_decode:
            slots = torch.where(m.valid, m.slots, -1)
            mixed = causal_conv1d_update(
                qkv.contiguous(),
                pool.conv[self.layer_id],
                weight,
                activation="silu",
                conv_state_indices=slots,
            )
            output = torch.empty((x.shape[0], 1, self.hv, self.dv), device=x.device, dtype=x.dtype)
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed,
                a,
                b,
                self.A_log,
                self.dt_bias,
                self.dk**-0.5,
                state,
                output,
                slots,
                use_qk_l2norm_in_kernel=True,
            )
            output = output[:, 0]
        else:
            initial = torch.tensor(
                [cached > 0 for _, cached, _, _, _ in m.spans], device=x.device, dtype=torch.bool
            )
            tracking = pool.track_states if m.track_offsets else None
            if tracking is not None:
                tracking.capture_window("conv", self.layer_id, qkv, m)
            mixed = causal_conv1d_fn(
                qkv.T,
                weight,
                None,
                conv_states=pool.conv[self.layer_id],
                has_initial_state=initial,
                cache_indices=m.slots,
                query_start_loc=m.cu_seqlens,
                seq_lens_cpu=[end - start for _, _, _, start, end in m.spans],
                activation="silu",
            ).T
            kd, vd = self.hk * self.dk, self.hv * self.dv
            q, k, v = mixed.split((kd, kd, vd), -1)
            g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            output, _, _ = chunk_gated_delta_rule(
                q.reshape(1, -1, self.hk, self.dk),
                k.reshape(1, -1, self.hk, self.dk),
                v.reshape(1, -1, self.hv, self.dv),
                g,
                beta,
                initial_state=state,
                initial_state_indices=m.slots,
                cu_seqlens=m.cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                track_state=tracking.recurrent[self.layer_id] if tracking is not None else None,
                track_chunk_idx=m.track_chunk_idx,
            )
            if tracking is not None:
                # The tracking hook records chunk *starts*. An exact forward
                # end is instead copied from the final FP32 working state.
                for row, (offset, span) in enumerate(zip(m.track_offsets, m.spans)):
                    slot, cached, end, _, _ = span
                    if offset and offset == end - cached:
                        tracking.recurrent[self.layer_id][row].copy_(state[slot])
            output = output[0]
        return _output_projection(
            self.out_proj,
            ops.gdn_output(output, z, self.norm.weight, self.eps, stable=stable_prefill).flatten(1),
            linear,
        )


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
        self.runtime = config.qwen4_runtime
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

    def _forward_eager(self, x):
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

    def forward(self, x):
        if not self.runtime.enabled(x):
            return self._forward_eager(x)
        ctx, rc = get_global_ctx(), self.config.rotary_config
        pos = ctx.batch.positions
        linear = _prefill_linear(ctx)
        if self.runtime.aligned:
            cut = sum(self._widths[:3])
            qg, k, v = linear(x, self._packed[:cut]).split(self._widths[:3], -1)
            iqk = linear(x, self._packed[cut:])
        else:
            qg, k, v, iqk = F.linear(x, self._packed).split(self._widths, -1)
        q, gate = qg.reshape(-1, self.hq, 2 * self.dim).chunk(2, -1)
        k = k.reshape(-1, self.hkv, self.dim)
        v = v.reshape(-1, self.hkv, self.dim).contiguous()
        kwargs = {
            "eps": self.config.rms_norm_eps,
            "positions": pos,
            "rotary_dim": rc.rotary_dim,
            "base": rc.base,
        }

        def norm(t, w):
            if self.runtime.aligned:
                return ops.norm_rope(t, w, pos, self._rope_cache, self.config.rms_norm_eps)
            return ops.legacy_norm(t, w, **kwargs)

        q = norm(q, self.q_norm.weight)
        k = norm(k, self.k_norm.weight)
        ix = self.indexer
        iq, ik = iqk.split((ix.heads * ix.dim, ix.dim), -1)
        iq = iq.view(-1, ix.heads, ix.dim)
        if self.runtime.aligned:
            iq = ops.index_norm_rope(
                iq, ix.q_layernorm.weight, pos, self._rope_cache, self.config.rms_norm_eps
            )
        else:
            iq = norm(iq, ix.q_layernorm.weight)
        out = ctx.attn_backend.qsa(q, k, v, iq, ik, ix.k_layernorm.weight, self.layer_id)
        if self.runtime.aligned:
            return _output_projection(self.o_proj, ops.sigmoid_mul(out, gate).flatten(1), linear)
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
        self.runtime = config.qwen4_runtime
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

    def _forward_eager(self, x):
        # The existing fused MoE backend reuses x as its output buffer.
        shared = self.shared_expert.forward(x) * self.shared_expert_gate.forward(x).sigmoid()
        routed = self.experts.forward(x, self.gate.forward(x))
        return routed + shared

    def forward(self, x):
        if not self.runtime.enabled(x):
            return self._forward_eager(x)
        if self.runtime.aligned:
            from minisgl.layers import silu_and_mul
            from minisgl.moe.fused import fused_experts_impl

            # The routed backend writes its result into its input buffer.
            linear = _prefill_linear(get_global_ctx())
            original = x.clone()
            shared = silu_and_mul(linear(x, self._packed))
            shared = linear(shared, self.shared_expert.down_proj.weight)
            weights, ids = ops.router(linear(x, self.gate.weight), self.experts.top_k)
            routed = fused_experts_impl(
                x, self.experts.gate_up_proj, self.experts.down_proj, weights, ids
            )
            output = ops.moe_combine(
                original,
                self.shared_expert_gate.weight,
                shared,
                routed,
                stable=linear is ops.linear_stable,
            )
            return self.experts._comm.all_reduce(output) if get_tp_info().size > 1 else output
        shared = self.shared_expert
        y, gate = ops.legacy_shared_activation(
            x, F.linear(x, self._packed), self.shared_expert_gate.weight
        )
        y = F.linear(y, shared.down_proj.weight)
        # The routed implementation may overwrite x. Shared/gate must run first.
        routed = self.experts.forward(x, self.gate.forward(x), reduce_results=False)
        if get_tp_info().size > 1:
            # Pack independent sums into one collective. Summing shared+routed
            # before reduction changes BF16 rounding and can amplify MoE routing
            # drift. Preserve the reference's two reduction results instead.
            packed = self.experts._comm.all_reduce(torch.cat((routed, y), dim=-1))
        else:
            packed = torch.cat((routed, y), dim=-1)
        return ops.legacy_moe_combine(packed, gate)


class NGramTable(VocabParallelEmbedding):
    def initialize_runtime(self, device):
        if device.type == "cuda" and (
            self.weight.dtype != torch.bfloat16 or not self.weight.is_contiguous()
        ):
            raise ValueError("Qwen4 PLE lookup requires a contiguous BF16 table")
        self._host_pointer = (
            ops.ple_host_pointer(self.weight, device)
            if self.weight.device.type == "cpu" and device.type == "cuda"
            else None
        )

    def lookup_local(self, ids, out=None):
        # PLE's 160 BF16 values/row are not supported by the generic warp-copy
        # embedding kernel (320 bytes is not a multiple of 128 bytes).
        start, count = self.vocab_range
        if ids.is_cuda:
            shape = (ids.numel(), self.weight.shape[1])
            if self.weight.is_cuda and self.weight.device != ids.device:
                raise ValueError("PLE table and IDs must be on the same CUDA device")
            if out is None:
                out = torch.empty(shape, device=ids.device, dtype=self.weight.dtype)
            elif out.shape != shape:
                raise ValueError(f"PLE lookup output shape must be {shape}")
            pointer = self.weight.data_ptr() if self.weight.is_cuda else self._host_pointer
            return ops.ple_lookup(pointer, ids, out, start, count)
        local = ids.long() - start
        valid = (local >= 0) & (local < count)
        result = F.embedding(local.clamp(0, count - 1), self.weight) * valid[:, None]
        if out is not None:
            out.copy_(result)
            return out
        return result

    def reduce_embeddings(self, out):
        return self._comm.all_reduce(out) if self.tp_size > 1 else out

    def forward(self, ids):
        return self.reduce_embeddings(self.lookup_local(ids))


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
        self.runtime = config.qwen4_runtime
        c = config.hybrid
        self.layer_id, self.config = layer_id, config
        self.ple_embedding = NGramEmbedding(config, c["ple_layer_ids"].index(layer_id + 1))
        width = c["hc_count"] * config.hidden_size
        self.key_proj = LinearReplicated(c["ple_embed_dim"], width, False)
        self.value_proj = LinearReplicated(c["ple_embed_dim"], config.hidden_size, False)
        self.norm_key, self.norm_query, self.norm_conv = Weight(width), Weight(width), Weight(width)
        self.conv1d = Weight(width, 1, c["ple_conv_kernel_size"])
        self._prefetch_stream = None
        self._prefetch_buffers = {}
        self._prefetch_state = None

    def initialize_runtime(self, device):
        self.ple_embedding.ngram_embedding.initialize_runtime(device)
        if self.runtime.ple_prefetch:
            self._prefetch_stream = torch.cuda.Stream(device=device)

    def prepare_ids(self):
        """Advance token history exactly once, whether lookup is prefetched or not."""
        ctx, cfg = get_global_ctx(), self.config
        c, pe = cfg.hybrid, self.ple_embedding
        if self.runtime.enabled(ctx.batch.input_ids) and not ctx.batch.is_prefill:
            m = ctx.batch.attn_metadata
            return ops.hash_decode(
                ctx.batch.input_ids,
                ctx.kv_cache.ple_history[self.layer_id],
                pe.layer_multipliers,
                pe.ngram_heads_vocab_sizes,
                pe.ngram_heads_offsets,
                pe.heads,
                c["eos_token_id"],
                m.slots,
                m.valid,
            ).flatten()
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
        return torch.cat(ids).flatten()

    def start_prefetch(self):
        if self._prefetch_stream is None:
            return
        if self._prefetch_state is not None:
            raise RuntimeError("PLE prefetch must be consumed before reuse")
        ids = self.prepare_ids()
        table = self.ple_embedding.ngram_embedding
        # Captured outputs need stable addresses. Eager prefill/decode share one
        # growable buffer, avoiding quadratic retention across batch sizes.
        key = ids.numel() if torch.cuda.is_current_stream_capturing() else "eager"
        buffer = self._prefetch_buffers.get(key)
        if buffer is None or buffer.shape[0] < ids.numel():
            buffer = torch.empty(
                (ids.numel(), table.weight.shape[1]), device=ids.device, dtype=table.weight.dtype
            )
            self._prefetch_buffers[key] = buffer
        output = buffer[: ids.numel()]
        stream = self._prefetch_stream
        stream.wait_stream(torch.cuda.current_stream())
        ids.record_stream(stream)
        with torch.cuda.stream(stream):
            table.lookup_local(ids, out=output)
        self._prefetch_state = output

    def consume_embeddings(self):
        table = self.ple_embedding.ngram_embedding
        if self._prefetch_state is None:
            return table.forward(self.prepare_ids())
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)
        output = self._prefetch_state
        self._prefetch_state = None
        # Collectives must stay ordered on the model stream: the communication
        # backend shares its workspace with attention and MoE.
        return table.reduce_embeddings(output)

    def _forward_eager(self, residual):
        ctx, cfg = get_global_ctx(), self.config
        c = cfg.hybrid
        embeddings = self.consume_embeddings().view(residual.shape[0], -1)
        h, n, eps = cfg.hidden_size, c["hc_count"], cfg.rms_norm_eps

        norm = rms_norm
        if self.runtime.aligned and self.runtime.enabled(residual):
            from minisgl.kernel.qwen4_ops import grouped_norm

            def norm(x, w, eps, h):
                return grouped_norm(x, w, h, eps)

        linear = _prefill_linear(ctx)
        key = norm(linear(embeddings, self.key_proj.weight), self.norm_key.weight, eps, h).view(
            -1, n, h
        )
        query = norm(residual, self.norm_query.weight, eps, h).view(-1, n, h)
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(h)
        gate = gate.sign() * gate.abs().clamp_min(1e-6).sqrt()
        value = (gate.sigmoid() * linear(embeddings, self.value_proj.weight)[:, None]).flatten(1)
        normed = norm(value, self.norm_conv.weight, eps, h)
        if ctx.batch.attn_metadata.track_offsets:
            ctx.kv_cache.track_states.capture_window(
                "ple_conv", self.layer_id, normed, ctx.batch.attn_metadata
            )
        out = torch.empty_like(value)
        for slot, _, _, a, b in ctx.batch.attn_metadata.spans:
            out[a:b] = causal_conv(
                normed[a:b],
                self.conv1d.weight,
                ctx.kv_cache.ple_conv[self.layer_id][slot],
                c["ngram_size"],
            )
        return value + out

    def forward(self, residual):
        ctx = get_global_ctx()
        if not self.runtime.enabled(residual) or ctx.batch.is_prefill:
            return self._forward_eager(residual)
        cfg = self.config
        c, m = cfg.hybrid, ctx.batch.attn_metadata
        emb = self.consume_embeddings().view(residual.shape[0], -1)
        h, n, eps = cfg.hidden_size, c["hc_count"], cfg.rms_norm_eps

        def norm(x, w):
            if self.runtime.aligned:
                return ops.grouped_norm(x, w, h, eps)
            return ops.legacy_norm(x, w, eps, h)

        key = norm(self.key_proj.forward(emb), self.norm_key.weight).view(-1, n, h)
        query = norm(residual, self.norm_query.weight).view(-1, n, h)
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(h)
        if self.runtime.aligned:
            value = ops.ple_gate_value(gate, self.value_proj.forward(emb))
        else:
            gate = gate.sign() * gate.abs().clamp_min(1e-6).sqrt()
            value = (gate.sigmoid() * self.value_proj.forward(emb)[:, None]).flatten(1)
        normed = norm(value, self.norm_conv.weight)
        conv = ops.ple_conv_decode if self.runtime.aligned else ops.legacy_conv_decode
        out = conv(
            normed,
            self.conv1d.weight,
            ctx.kv_cache.ple_conv[self.layer_id],
            m.slots,
            m.valid,
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
        self.runtime = config.qwen4_runtime
        if self.runtime.aligned is None:
            raise ValueError("Resolve Qwen4RuntimeConfig on the target device before construction")
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

    def load_weights(self, model_path, device):
        from .qwen4_weight import load_qwen4_weights, pack_qwen4_weights

        load_qwen4_weights(self, model_path, device)
        pack_qwen4_weights(self)
        for layer in self.model.layers.op_list:
            if layer.ple is not None:
                layer.ple.initialize_runtime(device)

    def forward(self):
        ctx = get_global_ctx()
        if ctx.batch.is_prefill:
            for slot, cached, _, _, _ in ctx.batch.attn_metadata.spans:
                if cached == 0:
                    ctx.kv_cache.reset(slot)
            if ctx.batch.attn_metadata.track_offsets:
                for row, req in enumerate(ctx.batch.reqs):
                    if not req.checkpoint_len:
                        continue
                    for history in ctx.kv_cache.track_states.ple_history.values():
                        length, width = req.checkpoint_len, history.shape[-1]
                        tokens = req.input_ids[max(0, length - width) : length]
                        history[row].fill_(ctx.kv_cache.eos)
                        history[row, width - len(tokens) :].copy_(tokens, non_blocking=True)
        residual = self.model.embed_tokens.forward(ctx.batch.input_ids).repeat(
            1, self.config.hybrid["hc_count"]
        )
        layers = self.model.layers.op_list
        if layers and layers[0].ple is not None:
            layers[0].ple.start_prefetch()
        for i, layer in enumerate(layers):
            if i + 1 < len(layers) and layers[i + 1].ple is not None:
                layers[i + 1].ple.start_prefetch()
            residual = layer.forward(residual)
        hidden, _ = self.model.hyper_connection_mixer.forward(residual)
        return self.lm_head.forward(hidden)
