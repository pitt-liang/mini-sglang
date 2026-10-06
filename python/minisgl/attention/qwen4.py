from dataclasses import dataclass

import torch
from minisgl.core import get_global_ctx
from minisgl.models.qwen4_config import Qwen4Config

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
    track_offsets: tuple[int, ...] = ()
    track_chunk_idx: torch.Tensor | None = None

    def get_last_indices(self, bs):
        return self.last_indices[:bs]


class Qwen4Backend(BaseAttnBackend):
    def __init__(self, config: Qwen4Config) -> None:
        if not isinstance(config, Qwen4Config):
            raise TypeError("Qwen4 attention requires Qwen4Config")
        self.config: Qwen4Config = config

        from minisgl.kernel.qwen4_qsa import SparseDecode

        self.sparse_decode = SparseDecode()

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
        pool = get_global_ctx().kv_cache
        if getattr(pool, "internal_checkpoints", False) and batch.is_prefill:
            offsets = tuple(
                req.checkpoint_len - req.cached_len if req.checkpoint_len else 0
                for req in batch.reqs
            )
            if any(offsets):
                batch.attn_metadata.track_offsets = offsets
                batch.attn_metadata.track_chunk_idx = torch.tensor(
                    [
                        offset // 64 if 0 < offset < req.extend_len else -1
                        for req, offset in zip(batch.reqs, offsets)
                    ],
                    dtype=torch.int32,
                    device=pool.device,
                )

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

    def forward(self, q, k, v, layer_id, batch):
        raise RuntimeError("Qwen4 requires indexer keys as well as core Q/K/V")

    def init_capture_graph(self, max_seq_len, bs_list):
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
        from minisgl.kernel.qwen4_qsa import (
            compress_decode,
            index_scores_decode,
            sparse_gqa,
            store_decode,
        )

        ctx, c = get_global_ctx(), self.config
        batch, pool, table = ctx.batch, ctx.kv_cache, ctx.page_table
        m, rc = batch.attn_metadata, self.config.rotary_config
        from minisgl.kernel import qwen4_ops as ops

        rope_cache = ops.rope_cache(q.device, rc.rotary_dim, rc.base, rc.max_position)
        ratio, budget = c.indexer_compress_ratio, c.indexer_budget
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
            scores = index_scores_decode(
                iq, pool.index_keys[li], table, m.slots, m.lengths, m.valid, ratio, budget
            )
            nb = m.lengths // ratio
            chosen = ops.index_topk(
                scores,
                torch.where(m.valid, nb, 0),
                stable=getattr(pool, "stable_numerics", False),
            )
            indices = self.expand(chosen, nb, m.lengths, m.valid, ratio)
            return self.sparse_decode(q, kc, vc, indices, table, m.slots)
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
                pooled = ops.index_norm_rope(
                    pooled[:, None], index_norm, starts, rope_cache, self.config.rms_norm_eps
                )[:, 0]
                pool.index_keys[li, table[slot, starts].long() // ratio] = pooled
            tail = raw[complete * ratio :]
            pool.pending[li, slot, : tail.shape[0]].copy_(tail)
            nb = end // ratio
            blocks = torch.arange(nb, device=q.device)
            keys = pool.index_keys[li, table[slot, blocks * ratio].long() // ratio].float()
            chunk = max(1, min(1024, 2**22 // max(1, nb * iq.shape[1])))
            for start in range(a, b, chunk):
                stop = min(start + chunk, b)
                lengths = torch.arange(
                    cached + start - a + 1, cached + stop - a + 1, device=q.device
                )
                counts = lengths // ratio
                if nb:
                    scores = (iq[start:stop].float() @ keys.T).relu().sum(1)
                    scores = scores.masked_fill(blocks[None, :] >= counts[:, None], -float("inf"))
                    scores = scores / (iq.shape[-1] ** 0.5)
                    chosen = ops.index_topk(
                        scores.contiguous(),
                        counts,
                        stable=getattr(pool, "stable_numerics", False),
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
                    prefill_rows=q.shape[0],
                    stable_prefill=getattr(pool, "stable_numerics", False),
                )
        return out

    def expand(self, chosen, counts, lengths, valid, ratio):
        from minisgl.kernel.qwen4_qsa import expand_compact

        return expand_compact(chosen, counts, lengths, valid, ratio)
