"""Strict safetensors -> local TP shard streaming loader, including sharded PLE."""

import json
from pathlib import Path

import torch
from minisgl.distributed import get_tp_info
from minisgl.kernel import qwen4_ops as ops
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.embedding import VocabParallelEmbedding
from minisgl.layers.linear import LinearColParallelMerged, LinearOProj
from minisgl.utils import init_logger
from minisgl.utils.hf import download_hf_weight
from safetensors import safe_open

logger = init_logger(__name__)


def _parameters(op, prefix=""):
    if isinstance(op, OPList):
        for i, child in enumerate(op.op_list):
            yield from _parameters(child, f"{prefix}.{i}")
        return
    for attr, value in vars(op).items():
        if attr.startswith("_"):
            continue
        name = f"{prefix}.{attr}" if prefix else attr
        if isinstance(value, torch.Tensor):
            yield name, op, attr, value
        elif isinstance(value, BaseOP):
            yield from _parameters(value, name)


def load_qwen4_weights(model, model_path, device):
    path = Path(download_hf_weight(model_path))
    with (path / "model.safetensors.index.json").open() as f:
        index = json.load(f)["weight_map"]
    used = set()
    rank, tp = get_tp_info().rank, get_tp_info().size
    c = model.config.hybrid
    kd = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    vd = c["linear_num_value_heads"] * c["linear_value_head_dim"]

    def copy_slice(key, dst, source_slices):
        if key not in index:
            raise ValueError(f"Missing checkpoint weight: {key}")
        with safe_open(str(path / index[key]), framework="pt", device="cpu") as f:
            source = f.get_slice(key)[source_slices]
            if source.shape != dst.shape:
                raise ValueError(f"Shape mismatch {key}: {source.shape} != {dst.shape}")
            dst.copy_(source)
        used.add(key)

    for number, (name, owner, attr, meta) in enumerate(_parameters(model)):
        key = "model.language_model." + name[len("model.") :] if name.startswith("model.") else name
        dtype = torch.float32 if attr in ("A_log", "dt_bias") else meta.dtype
        dst = torch.empty(meta.shape, dtype=dtype, device=device)
        if ".ple_embedding.ngram_embedding.weight" in name:
            prefix = key[: -len("weight")]
            shard_keys = sorted(
                (k for k in index if k.startswith(prefix + "shard_")),
                key=lambda k: int(k.split("shard_")[-1].split(".")[0]),
            )
            if len(shard_keys) != c["split_ngram_parts"]:
                raise ValueError(
                    f"Expected {c['split_ngram_parts']} PLE shards, got {len(shard_keys)}"
                )
            lo, count = owner.vocab_range
            hi, offset = lo + count, 0
            dst.zero_()  # padded final vocabulary rows, if any
            for shard in shard_keys:
                with safe_open(str(path / index[shard]), framework="pt", device="cpu") as f:
                    rows, dim = f.get_slice(shard).get_shape()
                if dim != dst.shape[1]:
                    raise ValueError(f"Invalid PLE embedding width: {shard}")
                start, stop = max(lo, offset), min(hi, offset + rows)
                if start < stop:
                    copy_slice(
                        shard,
                        dst[start - lo : stop - lo],
                        (slice(start - offset, stop - offset), slice(None)),
                    )
                used.add(shard)  # other TP ranks own the remaining rows
                offset += rows
            if offset != owner.num_embeddings:
                raise ValueError(f"PLE row count mismatch: {offset} != {owner.num_embeddings}")
        elif name.endswith("experts.gate_up_proj"):
            intermediate = model.config.moe_intermediate_size
            local = intermediate // tp
            for expert in range(model.config.num_experts):
                for half in range(2):
                    lo = half * intermediate + rank * local
                    copy_slice(
                        key,
                        dst[expert, half * local : (half + 1) * local],
                        (expert, slice(lo, lo + local), slice(None)),
                    )
        elif name.endswith("experts.down_proj"):
            local = dst.shape[-1]
            for expert in range(model.config.num_experts):
                copy_slice(
                    key, dst[expert], (expert, slice(None), slice(rank * local, (rank + 1) * local))
                )
        elif name.endswith(("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight")):
            offset, target_offset = 0, 0
            for size in (kd, kd, vd):
                local = size // tp
                copy_slice(
                    key,
                    dst[target_offset : target_offset + local],
                    slice(offset + rank * local, offset + (rank + 1) * local),
                )
                offset += size
                target_offset += local
        elif isinstance(owner, LinearOProj):
            local = dst.shape[1]
            copy_slice(key, dst, (slice(None), slice(rank * local, (rank + 1) * local)))
        elif isinstance(owner, (LinearColParallelMerged, VocabParallelEmbedding)) or attr in (
            "A_log",
            "dt_bias",
        ):
            local = dst.shape[0]
            copy_slice(key, dst, slice(rank * local, (rank + 1) * local))
        else:
            copy_slice(key, dst, slice(None))
        setattr(owner, attr, dst)
        if number % 100 == 0:
            logger.info_rank0(f"Qwen4 streaming load: {number} tensors, {name}")
    unexpected = set(index) - used
    # These are explicitly unsupported independent towers, not silently dropped text layers.
    unexpected = {k for k in unexpected if not k.startswith(("model.visual.", "mtp."))}
    if unexpected:
        raise ValueError(f"Unexpected checkpoint weights: {sorted(unexpected)}")
    logger.info_rank0(
        f"Qwen4 text weights loaded; {len(used)} checkpoint tensors validated (vision/MTP excluded)"
    )


def pack_qwen4_weights(model):
    """One allocation per merged projection; original parameter names stay views.

    Called after checkpoint validation, before graph capture. No permanent copy
    of the original matrices is retained, and reference forward stays available.
    """
    if model.runtime.aligned is None:
        raise ValueError("Qwen4 execution policy must be resolved before packing weights")
    if model.runtime.aligned and torch.cuda.get_device_capability()[0] == 10:
        from minisgl.kernel._qwen4_cute.hc_mix import permute_pad_up_weight

        groups = [model.model.hyper_connection_mixer]
        for layer in model.model.layers.op_list:
            groups.extend((layer.attn_hyper_connection, layer.mlp_hyper_connection))
        for gr in groups:
            gr._sg_up = permute_pad_up_weight(gr.input_mix_weight_up.weight, gr.n)
    for layer in model.model.layers.op_list:
        attn = layer.linear_attn or layer.self_attn
        if model.runtime.aligned and layer.self_attn is not None:
            rc = attn.config.rotary_config
            attn._rope_cache = ops.rope_cache(
                attn.q_norm.weight.device, rc.rotary_dim, rc.base, rc.max_position
            )
        projections = (
            (
                [attn.in_proj_qkv, attn.in_proj_z, attn.in_proj_b, attn.in_proj_a]
                if model.runtime.aligned
                else [attn.in_proj_qkv, attn.in_proj_z, attn.in_proj_a, attn.in_proj_b]
            )
            if layer.linear_attn is not None
            else [attn.q_proj, attn.k_proj, attn.v_proj, attn.indexer.index_qk_proj]
        )
        for owner, parts in [
            (attn, projections),
            (layer.mlp, [layer.mlp.shared_expert.gate_proj, layer.mlp.shared_expert.up_proj]),
        ]:
            owner._widths = [op.weight.shape[0] for op in parts]
            owner._packed = torch.cat([op.weight for op in parts], dim=0)
            for op, view in zip(parts, owner._packed.split(owner._widths, 0)):
                op.weight = view
