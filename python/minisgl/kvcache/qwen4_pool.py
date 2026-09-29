"""Token-paged QSA KV and private working / immutable checkpoint state."""

import torch
from minisgl.distributed import get_tp_info

from .mha_pool import MHAKVCache


def recurrent_bytes(config, slots, tp_size, dtype):
    c = config.hybrid
    hk, hv = c["linear_num_key_heads"] // tp_size, c["linear_num_value_heads"] // tp_size
    dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
    linear = c["layer_types"].count("linear_attention")
    matrix = hv * dv * dk * 4
    conv = (2 * hk * dk + hv * dv) * (c["linear_conv_kernel_dim"] - 1) * dtype.itemsize
    pending = (
        config.num_kv_layers
        * (c["indexer_compress_ratio"] - 1)
        * c["indexer_head_dim"]
        * dtype.itemsize
    )
    ple = len(c["ple_layer_ids"]) * (
        c["hc_count"]
        * config.hidden_size
        * (c["ple_conv_kernel_size"] - 1)
        * c["ngram_size"]
        * dtype.itemsize
        + (c["ngram_size"] - 1) * 8
    )
    return slots * (linear * (matrix + conv) + pending + ple)


class Qwen4KVCache(MHAKVCache):
    def __init__(self, config, num_pages, page_size, dtype, device, slots):
        c, tp = config.hybrid, get_tp_info().size
        ratio = c["indexer_compress_ratio"]
        if page_size % ratio:
            raise ValueError("QSA page_size must be divisible by indexer_compress_ratio")
        super().__init__(
            config.num_kv_heads,
            config.num_kv_layers,
            config.head_dim,
            num_pages,
            page_size,
            dtype,
            device,
        )
        self.layer_map = {
            i: j
            for j, i in enumerate(
                i for i, kind in enumerate(c["layer_types"]) if kind == "full_attention"
            )
        }
        kw = {"dtype": dtype, "device": device}
        hk, hv = c["linear_num_key_heads"] // tp, c["linear_num_value_heads"] // tp
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        self.recurrent, self.conv = {}, {}
        for i, kind in enumerate(c["layer_types"]):
            if kind == "linear_attention":
                self.recurrent[i] = torch.empty(
                    slots, hv, dv, dk, dtype=torch.float32, device=device
                )
                self.conv[i] = torch.empty(
                    slots, 2 * hk * dk + hv * dv, c["linear_conv_kernel_dim"] - 1, **kw
                )
        self.index_keys = torch.empty(
            config.num_kv_layers, num_pages * page_size // ratio, c["indexer_head_dim"], **kw
        )
        self.pending = torch.empty(
            config.num_kv_layers, slots, ratio - 1, c["indexer_head_dim"], **kw
        )
        self.ple_conv, self.ple_history = {}, {}
        for one_based in c["ple_layer_ids"]:
            i = one_based - 1
            self.ple_conv[i] = torch.empty(
                slots,
                c["hc_count"] * config.hidden_size,
                (c["ple_conv_kernel_size"] - 1) * c["ngram_size"],
                **kw,
            )
            self.ple_history[i] = torch.empty(
                slots, c["ngram_size"] - 1, dtype=torch.int64, device=device
            )
        self.eos = c["eos_token_id"]

    def reset(self, slot):
        for state in (*self.recurrent.values(), *self.conv.values(), *self.ple_conv.values()):
            state[slot].zero_()
        for state in self.ple_history.values():
            state[slot].fill_(self.eos)
        self.pending[:, slot].zero_()

    def k_cache(self, index):
        return super().k_cache(self.layer_map[index])

    def v_cache(self, index):
        return super().v_cache(self.layer_map[index])

    def store_kv(self, k, v, out_loc, layer_id):
        return super().store_kv(k.flatten(1), v.flatten(1), out_loc, self.layer_map[layer_id])

    def state_tensors(self):
        # Pending QSA keys are deliberately excluded: reusable checkpoints are
        # compression-group aligned. Compressed keys share the full-KV pages.
        return tuple(
            state
            for group in (self.recurrent, self.conv, self.ple_conv, self.ple_history)
            for state in group.values()
        )


class Qwen4StatePool:
    """Fixed-budget snapshots; allocation/ownership is managed by the radix cache.

    Preserve every source dtype, notably FP32 GDN state. All copies are enqueued
    on the caller's stream; consumers must obey the checkpoint's ready event.
    """

    def __init__(self, working, capacity):
        self.working = working
        self.capacity = capacity
        self.tensors = tuple(
            torch.empty((capacity, *t.shape[1:]), dtype=t.dtype, device=t.device)
            for t in working.state_tensors()
        )
        tensors = iter(self.tensors)
        for name in ("recurrent", "conv", "ple_conv", "ple_history"):
            setattr(self, name, {layer: next(tensors) for layer in getattr(working, name)})
        self.free_slots = list(range(capacity))

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)

    def allocate(self):
        return self.free_slots.pop() if self.free_slots else None

    def free(self, slot):
        assert slot not in self.free_slots, "checkpoint slot freed twice"
        self.free_slots.append(slot)

    def save(self, source, destination, source_tensors=None):
        source_tensors = self.working.state_tensors() if source_tensors is None else source_tensors
        for working, saved in zip(source_tensors, self.tensors):
            saved[destination].copy_(working[source])

    def restore(self, source, destination):
        for working, saved in zip(self.working.state_tensors(), self.tensors):
            working[destination].copy_(saved[source])
        self.working.pending[:, destination].zero_()

    def capture_window(self, name, layer, inputs, metadata):
        """Gather a short-conv boundary before the working window is updated."""
        source = getattr(self.working, name)[layer]
        target = getattr(self, name)[layer]
        width = source.shape[-1]
        for row, (offset, span) in enumerate(zip(metadata.track_offsets, metadata.spans)):
            if not offset:
                continue
            slot, _, _, start, _ = span
            if offset >= width:
                target[row].copy_(inputs[start + offset - width : start + offset].T)
            else:
                target[row].copy_(
                    torch.cat((source[slot, :, offset:], inputs[start : start + offset].T), -1)
                )
