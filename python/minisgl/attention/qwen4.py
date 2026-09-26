from dataclasses import dataclass

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.models.qwen4_ops import rms_norm, rotary

from .base import BaseAttnBackend, BaseAttnMetadata


@dataclass
class Qwen4Metadata(BaseAttnMetadata):
    # Immutable snapshots: (slot, cached length, end length, input start, input end).
    spans: tuple[tuple[int, int, int, int, int], ...]
    last_indices: torch.Tensor
    slots: torch.Tensor | None = None
    lengths: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None

    def get_last_indices(self, bs):
        return self.last_indices[:bs]


class Qwen4ReferenceBackend(BaseAttnBackend):
    def __init__(self, config):
        self.config = config
        self.runtime = config.qwen4_runtime
        if self.runtime.aligned is None:
            raise ValueError("Qwen4 attention requires a resolved execution policy")

    def prepare_metadata(self, batch):
        spans, offset = [], 0
        for req in batch.reqs:
            end = offset + req.extend_len
            spans.append((req.table_idx, req.cached_len, req.device_len, offset, end))
            offset = end
        batch.attn_metadata = Qwen4Metadata(
            tuple(spans),
            torch.tensor([span[-1] - 1 for span in spans], device=get_global_ctx().kv_cache.device),
        )

    def forward(self, q, k, v, layer_id, batch):
        raise RuntimeError("Qwen4 requires indexer keys as well as core Q/K/V")

    def qsa(self, q, k, v, iq, raw_keys, index_norm, layer_id):
        ctx, c = get_global_ctx(), self.config.hybrid
        batch, pool = ctx.batch, ctx.kv_cache
        pool.store_kv(k, v, batch.out_loc, layer_id)
        kc = pool.k_cache(layer_id).flatten(0, 1)
        vc = pool.v_cache(layer_id).flatten(0, 1)
        li = pool.layer_map[layer_id]
        ratio, budget = c["indexer_compress_ratio"], c["indexer_budget"]
        out = torch.empty_like(q)
        rc = self.config.rotary_config
        for slot, cached, end, a, b in batch.attn_metadata.spans:
            past = cached % ratio
            raw = torch.cat((pool.pending[li, slot, :past], raw_keys[a:b]))
            complete = raw.shape[0] // ratio
            if complete:
                pooled = (
                    raw[: complete * ratio].view(complete, ratio, -1).float().mean(1).to(raw.dtype)
                )
                starts = torch.arange(
                    cached - past, cached - past + complete * ratio, ratio, device=q.device
                )
                pooled = rotary(
                    rms_norm(pooled, index_norm, self.config.rms_norm_eps)[:, None],
                    starts,
                    rc.rotary_dim,
                    rc.base,
                )[:, 0]
                locations = ctx.page_table[slot, starts].long() // ratio
                pool.index_keys[li, locations] = pooled
            tail = raw[complete * ratio :]
            pool.pending[li, slot, : tail.shape[0]].copy_(tail)
            locations = ctx.page_table[slot, :end].long()
            keys, values = kc[locations], vc[locations]
            dense_count = max(0, min(b - a, budget - cached))
            if dense_count:
                dense_end = cached + dense_count
                mask = (
                    torch.arange(dense_end, device=q.device)[None, :]
                    <= torch.arange(cached, dense_end, device=q.device)[:, None]
                )
                result = F.scaled_dot_product_attention(
                    q[a : a + dense_count].transpose(0, 1)[None],
                    keys[:dense_end].transpose(0, 1)[None],
                    values[:dense_end].transpose(0, 1)[None],
                    attn_mask=mask[None, None],
                    enable_gqa=True,
                )
                out[a : a + dense_count] = result[0].transpose(0, 1)
            if dense_count == b - a:
                continue
            # Eager baseline, bounded attention scratch. All full KV remains cached.
            block_locs = locations[: end // ratio * ratio : ratio] // ratio
            block_keys = pool.index_keys[li, block_locs].float()
            for t in range(dense_count, b - a):
                visible = cached + t + 1
                nb = visible // ratio
                if nb > budget // ratio:
                    score = (iq[a + t].float() @ block_keys[:nb].T).relu().sum(0)
                    selected = score.topk(budget // ratio).indices
                else:
                    selected = torch.arange(nb, device=q.device)
                selected = (
                    selected[:, None] * ratio + torch.arange(ratio, device=q.device)
                ).flatten()
                selected = torch.cat((selected, torch.arange(nb * ratio, visible, device=q.device)))
                result = F.scaled_dot_product_attention(
                    q[a + t][None, :, None],
                    keys[selected].transpose(0, 1)[None],
                    values[selected].transpose(0, 1)[None],
                    enable_gqa=True,
                )
                out[a + t] = result[0, :, 0]
        return out

    def init_capture_graph(self, max_seq_len, bs_list):
        if bs_list:
            raise ValueError("Qwen4 eager backend requires --cuda-graph-max-bs 0")

    def prepare_for_capture(self, batch):
        raise ValueError("Qwen4 CUDA graphs are not implemented")

    def prepare_for_replay(self, batch):
        raise ValueError("Qwen4 CUDA graphs are not implemented")


class Qwen4Backend(Qwen4ReferenceBackend):
    def __init__(self, config):
        super().__init__(config)
        from minisgl.kernel.qwen4_qsa import SparseDecode

        self.aligned_decode = SparseDecode()

    def prepare_metadata(self, batch):
        super().prepare_metadata(batch)
        m = batch.attn_metadata
        device = get_global_ctx().kv_cache.device
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        m.slots = torch.tensor([r.table_idx for r in reqs], device=device, dtype=torch.int32)
        m.lengths = torch.tensor([r.device_len for r in reqs], device=device, dtype=torch.int32)
        m.valid = torch.arange(len(reqs), device=device) < len(batch.reqs)
        if not getattr(batch, "is_decode", False):
            m.cu_seqlens = torch.tensor(
                [0] + [span[-1] for span in m.spans], device=device, dtype=torch.int32
            )

    def init_capture_graph(self, max_seq_len, bs_list):
        if not self.runtime.enabled():
            return super().init_capture_graph(max_seq_len, bs_list)
        device = get_global_ctx().kv_cache.device
        n = max(bs_list)
        self.graph_metadata = Qwen4Metadata(
            (),
            torch.arange(n, device=device),
            torch.zeros(n, device=device, dtype=torch.int32),
            torch.ones(n, device=device, dtype=torch.int32),
            torch.zeros(n, device=device, dtype=torch.bool),
        )

    def prepare_for_capture(self, batch):
        m, n = self.graph_metadata, batch.padded_size
        m.valid.zero_()
        batch.attn_metadata = Qwen4Metadata(
            (), m.last_indices[:n], m.slots[:n], m.lengths[:n], m.valid[:n]
        )

    def prepare_for_replay(self, batch):
        dst, src, n = self.graph_metadata, batch.attn_metadata, batch.padded_size
        dst.slots[:n].copy_(src.slots)
        dst.lengths[:n].copy_(src.lengths)
        dst.valid[:n].copy_(src.valid)

    def qsa(self, q, k, v, iq, raw_keys, index_norm, layer_id):
        if not self.runtime.enabled():
            return super().qsa(q, k, v, iq, raw_keys, index_norm, layer_id)
        from minisgl.kernel.qwen4_qsa import (
            compress_decode,
            index_scores_decode,
            select_decode,
            sparse_gqa,
            store_decode,
        )

        ctx, c = get_global_ctx(), self.config.hybrid
        batch, pool, table = ctx.batch, ctx.kv_cache, ctx.page_table
        m, rc = batch.attn_metadata, self.config.rotary_config
        rope_cache = None
        if self.runtime.aligned:
            from minisgl.kernel import qwen4_ops as aligned

            rope_cache = aligned.rope_cache(q.device, rc.rotary_dim, rc.base, rc.max_position)
        ratio, budget = c["indexer_compress_ratio"], c["indexer_budget"]
        li = pool.layer_map[layer_id]
        kc, vc = pool.k_cache(layer_id).flatten(0, 1), pool.v_cache(layer_id).flatten(0, 1)
        if getattr(batch, "is_decode", False):
            store_decode(k, v, kc, vc, batch.out_loc, m.valid)
            compress_decode(
                raw_keys,
                index_norm,
                pool.pending[li],
                pool.index_keys[li],
                table,
                m.slots,
                batch.positions,
                m.valid,
                ratio,
                rc.rotary_dim,
                rc.base,
                self.config.rms_norm_eps,
                rope_cache=rope_cache,
            )
            if self.runtime.aligned:
                scores = index_scores_decode(
                    iq, pool.index_keys[li], table, m.slots, m.lengths, m.valid, ratio, budget
                )
                nb = m.lengths // ratio
                chosen = aligned.index_topk(scores, torch.where(m.valid, nb, 0))
                indices = self.expand(chosen, nb, m.lengths, m.valid, ratio)
                return self.aligned_decode(q, kc, vc, indices, table, m.slots)
            if table.shape[1] // ratio <= 2048 and not self.runtime.aligned:
                indices = select_decode(
                    iq, pool.index_keys[li], table, m.slots, m.lengths, m.valid, ratio, budget
                )
                return sparse_gqa(q, kc, vc, indices, table, m.slots, decode=True)
            blocks = torch.arange(table.shape[1] // ratio, device=q.device)
            loc = table[m.slots.long()[:, None], blocks[None, :] * ratio].long() // ratio
            keys = pool.index_keys[li, loc].float()
            nb = m.lengths // ratio
            visible = (blocks[None, :] < nb[:, None]) & m.valid[:, None]
            keys = torch.where(visible[:, :, None], keys, 0.0)
            scores = torch.matmul(iq.float(), keys.transpose(1, 2)).relu().sum(1)
            scores = scores.masked_fill(~visible, -float("inf"))
            chosen = scores.topk(min(budget // ratio, blocks.numel()), dim=-1).indices
            chosen = torch.where(
                nb[:, None] <= chosen.shape[1],
                torch.arange(chosen.shape[1], device=q.device)[None, :],
                chosen,
            )
            indices = self.expand(chosen, nb, m.lengths, m.valid, ratio)
            return sparse_gqa(q, kc, vc, indices, table, m.slots, decode=True)
        pool.store_kv(k, v, batch.out_loc, layer_id)
        out = torch.empty_like(q)
        for slot, cached, end, a, b in m.spans:
            past = cached % ratio
            raw = torch.cat((pool.pending[li, slot, :past], raw_keys[a:b]))
            complete = raw.shape[0] // ratio
            if complete:
                pooled = (
                    raw[: complete * ratio].view(complete, ratio, -1).float().mean(1).to(raw.dtype)
                )
                starts = torch.arange(
                    cached - past, cached - past + complete * ratio, ratio, device=q.device
                )
                if self.runtime.aligned:
                    pooled = aligned.index_norm_rope(
                        pooled[:, None], index_norm, starts, rope_cache, self.config.rms_norm_eps
                    )[:, 0]
                else:
                    pooled = rotary(
                        rms_norm(pooled, index_norm, self.config.rms_norm_eps)[:, None],
                        starts,
                        rc.rotary_dim,
                        rc.base,
                    )[:, 0]
                pool.index_keys[li, table[slot, starts].long() // ratio] = pooled
            tail = raw[complete * ratio :]
            pool.pending[li, slot, : tail.shape[0]].copy_(tail)
            # Before the sparse budget is reached QSA is exactly dense causal
            # attention. Keep the reference SDPA path for its numerical contract
            # and efficient multi-query prefill tiles.
            dense_count = max(0, min(b - a, budget - cached))
            if self.runtime.aligned:
                dense_count = 0
            if dense_count:
                dense_end = cached + dense_count
                loc = table[slot, :dense_end].long()
                mask = (
                    torch.arange(dense_end, device=q.device)[None, :]
                    <= torch.arange(cached, dense_end, device=q.device)[:, None]
                )
                dense = F.scaled_dot_product_attention(
                    q[a : a + dense_count].transpose(0, 1)[None],
                    kc[loc].transpose(0, 1)[None],
                    vc[loc].transpose(0, 1)[None],
                    attn_mask=mask[None, None],
                    enable_gqa=True,
                )
                out[a : a + dense_count] = dense[0].transpose(0, 1)
            if dense_count == b - a:
                continue
            nb = end // ratio
            blocks = torch.arange(nb, device=q.device)
            keys = pool.index_keys[li, table[slot, blocks * ratio].long() // ratio].float()
            chunk = max(1, min(1024, 2**22 // max(1, nb * iq.shape[1])))
            for start in range(a + dense_count, b, chunk):
                stop = min(start + chunk, b)
                lengths = torch.arange(
                    cached + start - a + 1, cached + stop - a + 1, device=q.device
                )
                counts = lengths // ratio
                if nb:
                    scores = (iq[start:stop].float() @ keys.T).relu().sum(1)
                    scores = scores.masked_fill(blocks[None, :] >= counts[:, None], -float("inf"))
                    if self.runtime.aligned:
                        scores = scores / (iq.shape[-1] ** 0.5)
                        chosen = aligned.index_topk(scores.contiguous(), counts)
                    else:
                        chosen = scores.topk(min(budget // ratio, nb), dim=-1).indices
                        chosen = torch.where(
                            counts[:, None] <= chosen.shape[1],
                            torch.arange(chosen.shape[1], device=q.device)[None, :],
                            chosen,
                        )
                else:
                    chosen = torch.empty((stop - start, 0), device=q.device, dtype=torch.int64)
                indices = self.expand(chosen, counts, lengths, None, ratio)
                slots = torch.full((stop - start,), slot, device=q.device, dtype=torch.int32)
                out[start:stop] = sparse_gqa(
                    q[start:stop],
                    kc,
                    vc,
                    indices,
                    table,
                    slots,
                    sglang_prefill_rows=q.shape[0] if self.runtime.aligned else None,
                )
        return out

    def expand(self, chosen, counts, lengths, valid, ratio):
        if self.runtime.aligned:
            from minisgl.kernel.qwen4_qsa import expand_compact

            return expand_compact(chosen, counts, lengths, valid, ratio)
        selected = chosen[:, :, None] * ratio + torch.arange(ratio, device=chosen.device)
        selected = torch.where(
            (chosen[:, :, None] >= 0) & (chosen[:, :, None] < counts[:, None, None]), selected, -1
        ).flatten(1)
        tail = counts[:, None] * ratio + torch.arange(ratio - 1, device=chosen.device)
        tail = torch.where(tail < lengths[:, None], tail, -1)
        indices = torch.cat((selected, tail), dim=1).to(torch.int32)
        if valid is not None:
            indices = torch.where(valid[:, None], indices, -1)
        return indices.contiguous()
