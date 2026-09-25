"""Fast implementations on the exact same parameters as the reference model.

Class specialization preserves the checkpoint schema. Set qwen4_fast.ENABLED=False
for same-weight correctness comparisons (eager only).
"""

import math

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.kernel import qwen4_fused as fused
from minisgl.kernel import qwen4_sglang as aligned
from minisgl.kernel.qwen4 import gated_delta_decode, gated_delta_rule

from . import qwen4_fast
from .qwen4_exp import PLE, GatedDeltaNet, GatedResidual, QSAAttention, SparseMoE
from .qwen4_ops import causal_conv


class FastGR(GatedResidual):
    def forward(self, residual):
        if not qwen4_fast.enabled(residual):
            return super().forward(residual)
        if qwen4_fast.SGLANG_NUMERICS:
            normed = aligned.grouped_norm(residual, self.hc_norm.weight, self.h, self.eps)
            if normed.shape[0] <= 24:
                if hasattr(self, "_sg_up"):
                    from minisgl.kernel._qwen4_cute.hc_mix import hc_mix

                    mixed = hc_mix(
                        normed, self.input_mix_weight_down.weight, self._sg_up, self.n, self.h
                    )
                else:
                    mixed = aligned.gr_project_mix(
                        normed,
                        self.input_mix_weight_down.weight,
                        self.input_mix_weight_up.weight,
                        self.n,
                    )
            else:
                mixed = aligned.gr_project_mix_large(
                    normed,
                    self.input_mix_weight_down.weight,
                    self.input_mix_weight_up.weight,
                    self.n,
                    self.h,
                )
            return mixed, normed
        normed = fused.norm(residual, self.hc_norm.weight, self.eps, self.h)
        if normed.shape[0] <= 16 and normed.shape[1] % 2048 == 0:
            mixed = fused.gr_project_mix(
                normed, self.input_mix_weight_down.weight, self.input_mix_weight_up.weight, self.n
            )
        else:
            gate = fused.silu_div(self.input_mix_weight_down.forward(normed), self.n)
            gate = self.input_mix_weight_up.forward(gate)
            mixed = fused.gr_mix(gate, normed, self.n)
        return mixed, normed

    def combine(self, residual, output, injection):
        if not qwen4_fast.enabled(residual):
            return super().combine(residual, output, injection)
        if qwen4_fast.SGLANG_NUMERICS:
            return aligned.gr_combine(residual, output, injection, self.block_inject_weight.weight)
        return fused.gr_combine(residual, output, injection, self.block_inject_weight.weight)


class FastGDN(GatedDeltaNet):
    def forward(self, x):
        if not qwen4_fast.enabled(x):
            return super().forward(x)
        if qwen4_fast.SGLANG_NUMERICS:
            return self.forward_sglang(x)
        ctx = get_global_ctx()
        qkv, z, a, b = F.linear(x, self._packed).split(self._widths, -1)
        qkv = qkv.contiguous()
        z = z.reshape(-1, self.hv, self.dv).contiguous()
        decay, beta = fused.gdn_gates(a, b, self.A_log, self.dt_bias)
        kd, vd = self.hk * self.dk, self.hv * self.dv
        m = ctx.batch.attn_metadata
        if ctx.batch.is_decode:
            mixed = fused.conv_decode(
                qkv, self.conv1d.weight, ctx.kv_cache.conv[self.layer_id], m.slots, m.valid
            )
            q, k, v = mixed.split((kd, kd, vd), -1)
            q, k = [fused.l2(t.reshape(-1, self.hk, self.dk)) for t in (q, k)]
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
            q, k = [fused.l2(t.reshape(-1, self.hk, self.dk)) for t in (q, k)]
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
                q, k = [fused.l2(t.reshape(-1, self.hk, self.dk)) for t in (q, k)]
                output[a:b] = prefill_delta(
                    q,
                    k,
                    v.reshape(-1, self.hv, self.dv),
                    decay[a:b],
                    beta[a:b],
                    ctx.kv_cache.recurrent[self.layer_id][slot],
                )
        y = fused.gdn_output(output, z, self.norm.weight, self.eps)
        return self.out_proj.forward(y.flatten(1))

    def forward_sglang(self, x):
        from minisgl.kernel._qwen4_fla.causal_conv import causal_conv1d_fn, causal_conv1d_update
        from minisgl.kernel._qwen4_fla.chunk import chunk_gated_delta_rule
        from minisgl.kernel._qwen4_fla.fused_gdn_gating import fused_gdn_gating
        from minisgl.kernel._qwen4_fla.packed_decode import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        ctx = get_global_ctx()
        m, pool = ctx.batch.attn_metadata, ctx.kv_cache
        # SGLang packs QKV/Z/B/A, and uses two GEMMs beyond 1024 tokens.
        if x.shape[0] <= 1024:
            projected = F.linear(x, self._packed)
        else:
            cut = self._widths[0] + self._widths[1]
            projected = torch.cat(
                (F.linear(x, self._packed[:cut]), F.linear(x, self._packed[cut:])), -1
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
            )
            output = output[0]
        return self.out_proj.forward(
            aligned.gdn_output(output, z, self.norm.weight, self.eps).flatten(1)
        )


class FastQSA(QSAAttention):
    def forward(self, x):
        if not qwen4_fast.enabled(x):
            return super().forward(x)
        ctx, rc = get_global_ctx(), self.config.rotary_config
        pos = ctx.batch.positions
        if qwen4_fast.SGLANG_NUMERICS:
            cut = sum(self._widths[:3])
            qg, k, v = F.linear(x, self._packed[:cut]).split(self._widths[:3], -1)
            iqk = F.linear(x, self._packed[cut:])
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
            if qwen4_fast.SGLANG_NUMERICS:
                return aligned.norm_rope(t, w, pos, self._rope_cache, self.config.rms_norm_eps)
            return fused.norm(t, w, **kwargs)

        q = norm(q, self.q_norm.weight)
        k = norm(k, self.k_norm.weight)
        ix = self.indexer
        iq, ik = iqk.split((ix.heads * ix.dim, ix.dim), -1)
        iq = iq.view(-1, ix.heads, ix.dim)
        if qwen4_fast.SGLANG_NUMERICS:
            iq = aligned.index_norm_rope(
                iq, ix.q_layernorm.weight, pos, self._rope_cache, self.config.rms_norm_eps
            )
        else:
            iq = norm(iq, ix.q_layernorm.weight)
        out = ctx.attn_backend.qsa(q, k, v, iq, ik, ix.k_layernorm.weight, self.layer_id)
        if qwen4_fast.SGLANG_NUMERICS:
            return self.o_proj.forward(aligned.sigmoid_mul(out, gate).flatten(1))
        return self.o_proj.forward((out * gate.sigmoid()).flatten(1))


class FastMoE(SparseMoE):
    def forward(self, x):
        if not qwen4_fast.enabled(x):
            return super().forward(x)
        if qwen4_fast.SGLANG_NUMERICS:
            from minisgl.layers import silu_and_mul
            from minisgl.moe.fused import fused_experts_impl

            # The routed backend writes its result into its input buffer.
            original = x.clone()
            shared = silu_and_mul(F.linear(x, self._packed))
            shared = F.linear(shared, self.shared_expert.down_proj.weight)
            weights, ids = aligned.router(self.gate.forward(x), self.experts.top_k)
            routed = fused_experts_impl(
                x, self.experts.gate_up_proj, self.experts.down_proj, weights, ids
            )
            output = aligned.moe_combine(original, self.shared_expert_gate.weight, shared, routed)
            return self.experts._comm.all_reduce(output) if get_tp_info().size > 1 else output
        shared = self.shared_expert
        y, gate = fused.shared_activation(
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
        return fused.moe_combine(packed, gate)


class FastPLE(PLE):
    def forward(self, residual):
        ctx = get_global_ctx()
        if not qwen4_fast.enabled(residual) or ctx.batch.is_prefill:
            return super().forward(residual)
        cfg = self.config
        c, pe, m = cfg.hybrid, self.ple_embedding, ctx.batch.attn_metadata
        hashed = fused.hash_decode(
            ctx.batch.input_ids,
            ctx.kv_cache.ple_history[self.layer_id],
            pe.layer_multipliers,
            pe.ngram_heads_vocab_sizes,
            pe.ngram_heads_offsets,
            pe.heads,
            c["eos_token_id"],
            m.slots,
            m.valid,
        )
        emb = pe.ngram_embedding.forward(hashed.flatten()).view(residual.shape[0], -1)
        h, n, eps = cfg.hidden_size, c["hc_count"], cfg.rms_norm_eps

        def norm(x, w):
            if qwen4_fast.SGLANG_NUMERICS:
                return aligned.grouped_norm(x, w, h, eps)
            return fused.norm(x, w, eps, h)

        key = norm(self.key_proj.forward(emb), self.norm_key.weight).view(-1, n, h)
        query = norm(residual, self.norm_query.weight).view(-1, n, h)
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(h)
        if qwen4_fast.SGLANG_NUMERICS:
            value = aligned.ple_gate_value(gate, self.value_proj.forward(emb))
        else:
            gate = gate.sign() * gate.abs().clamp_min(1e-6).sqrt()
            value = (gate.sigmoid() * self.value_proj.forward(emb)[:, None]).flatten(1)
        normed = norm(value, self.norm_conv.weight)
        conv = aligned.ple_conv_decode if qwen4_fast.SGLANG_NUMERICS else fused.conv_decode
        out = conv(
            normed,
            self.conv1d.weight,
            ctx.kv_cache.ple_conv[self.layer_id],
            m.slots,
            m.valid,
            c["ngram_size"],
        )
        return value + out


def install(model):
    for layer in model.model.layers.op_list:
        layer.attn_hyper_connection.__class__ = FastGR
        layer.mlp_hyper_connection.__class__ = FastGR
        layer.mlp.__class__ = FastMoE
        if layer.linear_attn is not None:
            layer.linear_attn.__class__ = FastGDN
        else:
            layer.self_attn.__class__ = FastQSA
        if layer.ple is not None:
            layer.ple.__class__ = FastPLE
    model.model.hyper_connection_mixer.__class__ = FastGR


def pack_weights(model):
    """One allocation per merged projection; original parameter names stay views.

    Called after checkpoint validation, before graph capture. No permanent copy
    of the original matrices is retained, and reference forward stays available.
    """
    qwen4_fast.configure_numerics(model.model.hyper_connection_mixer.hc_norm.weight.device)
    if qwen4_fast.SGLANG_NUMERICS and torch.cuda.get_device_capability()[0] == 10:
        from minisgl.kernel._qwen4_cute.hc_mix import permute_pad_up_weight

        groups = [model.model.hyper_connection_mixer]
        for layer in model.model.layers.op_list:
            groups.extend((layer.attn_hyper_connection, layer.mlp_hyper_connection))
        for gr in groups:
            gr._sg_up = permute_pad_up_weight(gr.input_mix_weight_up.weight, gr.n)
    for layer in model.model.layers.op_list:
        attn = layer.linear_attn or layer.self_attn
        if qwen4_fast.SGLANG_NUMERICS and layer.self_attn is not None:
            rc = attn.config.rotary_config
            attn._rope_cache = aligned.rope_cache(
                attn.q_norm.weight.device, rc.rotary_dim, rc.base, rc.max_position
            )
        ops = (
            (
                [attn.in_proj_qkv, attn.in_proj_z, attn.in_proj_b, attn.in_proj_a]
                if qwen4_fast.SGLANG_NUMERICS
                else [attn.in_proj_qkv, attn.in_proj_z, attn.in_proj_a, attn.in_proj_b]
            )
            if layer.linear_attn is not None
            else [attn.q_proj, attn.k_proj, attn.v_proj, attn.indexer.index_qk_proj]
        )
        for owner, parts in [
            (attn, ops),
            (layer.mlp, [layer.mlp.shared_expert.gate_proj, layer.mlp.shared_expert.up_proj]),
        ]:
            owner._widths = [op.weight.shape[0] for op in parts]
            owner._packed = torch.cat([op.weight for op in parts], dim=0)
            for op, view in zip(parts, owner._packed.split(owner._widths, 0)):
                op.weight = view


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
