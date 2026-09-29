"""A radix prefix is reusable only at an exact, complete Qwen4 checkpoint.

State is keyed by node identity, not its edge, so splitting an edge leaves the
checkpoint on the old endpoint. KV can outlive a state-only eviction. Working
slots never alias cached states. This cache is model-instance scoped.
"""

from dataclasses import dataclass

import torch
from minisgl.core import get_global_ctx

from .base import InsertResult, MatchResult
from .radix_cache import RadixCacheHandle, RadixPrefixCache, RadixTreeNode


@dataclass(frozen=True)
class StateCheckpoint:
    node: RadixTreeNode
    slot: int
    length: int
    ready: torch.cuda.Event | None


@dataclass(frozen=True)
class HybridCacheHandle(RadixCacheHandle):
    checkpoint: StateCheckpoint | None = None
    raw_hit_len: int = 0


class HybridPrefixCache(RadixPrefixCache):
    def __init__(self, device):
        super().__init__(device)
        pool = get_global_ctx().kv_cache
        self.states = pool.prefix_states
        self.alignment = pool.checkpoint_alignment
        self.interval = pool.checkpoint_interval
        self.internal = getattr(pool, "internal_checkpoints", False)
        self.checkpoints: dict[int, StateCheckpoint] = {}
        self.state_refs: dict[int, int] = {}
        self.stats = dict.fromkeys(
            ("raw_hit_tokens", "hit_tokens", "saves", "restores", "evictions", "skipped_saves"), 0
        )

    def match_prefix(self, input_ids):
        node, length = self._tree_walk(input_ids)
        raw_hit_len = length
        self.stats["raw_hit_tokens"] += length
        while not node.is_root() and node.uuid not in self.checkpoints:
            length -= node.length
            node = node.parent
        checkpoint = self.checkpoints.get(node.uuid)
        assert checkpoint is None or checkpoint.length == length
        self.stats["hit_tokens"] += length
        return MatchResult(HybridCacheHandle(length, node, checkpoint, raw_hit_len))

    def restore(self, handle, destination):
        checkpoint = handle.checkpoint
        if checkpoint is None:
            return
        assert checkpoint.node.ref_count > 0
        if checkpoint.ready is not None:
            torch.cuda.current_stream(self.device).wait_event(checkpoint.ready)
        self.states.restore(checkpoint.slot, destination)
        self.stats["restores"] += 1

    def lock_handle(self, handle, unlock=False):
        super().lock_handle(handle, unlock)
        checkpoint = handle.checkpoint
        if checkpoint is not None:
            slot = checkpoint.slot
            count = self.state_refs.get(slot, 0) + (-1 if unlock else 1)
            assert count >= 0
            if count:
                self.state_refs[slot] = count
            else:
                self.state_refs.pop(slot)

    def _reserve(self):
        slot = self.states.allocate()
        if slot is not None:
            return slot
        # KV path references protect all ancestor pages, but a descendant only
        # needs its own boundary state, not every ancestor's snapshot.
        candidates = [c for c in self.checkpoints.values() if not self.state_refs.get(c.slot)]
        if not candidates:
            self.stats["skipped_saves"] += 1
            return None
        victim = min(candidates, key=lambda c: c.node.timestamp)
        self._on_evict(victim.node)
        return self.states.allocate()

    def insert_prefix(self, input_ids, indices):
        raise RuntimeError("Hybrid insertion requires an exact forward-boundary state")

    def insert_checkpoint(self, input_ids, indices, working_slot, source_tensors=None):
        length = len(input_ids)
        assert length > 0 and length % self.alignment == 0
        # Probe before reserving: an existing immutable endpoint needs no copy.
        node, matched = self._tree_walk(input_ids)
        checkpoint = self.checkpoints.get(node.uuid) if matched == length else None
        if checkpoint is None:
            slot = self._reserve()
            if slot is None:
                return None
            self.states.save(working_slot, slot, source_tensors)
            ready = None
            if self.device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
        result = super().insert_prefix(input_ids, indices)
        node = result.handle.node
        if checkpoint is None:
            checkpoint = StateCheckpoint(node, slot, length, ready)
            self.checkpoints[node.uuid] = checkpoint
            self.stats["saves"] += 1
        return InsertResult(result.cached_len, HybridCacheHandle(length, node, checkpoint))

    def _on_evict(self, node):
        checkpoint = self.checkpoints.pop(node.uuid, None)
        if checkpoint is not None:
            assert not self.state_refs.get(checkpoint.slot)
            # A pending save may be evicted, but its slot cannot be overwritten
            # on another stream until the producer has finished writing it.
            if checkpoint.ready is not None:
                torch.cuda.current_stream(self.device).wait_event(checkpoint.ready)
            self.states.free(checkpoint.slot)
            self.stats["evictions"] += 1

    def check_integrity(self):
        slots = [c.slot for c in self.checkpoints.values()] + self.states.free_slots
        assert sorted(slots) == list(range(self.states.capacity))
        nodes = [(self.root_node, 0)]
        seen = set()
        evictable = protected = 0
        while nodes:
            node, depth = nodes.pop()
            seen.add(node.uuid)
            assert node.ref_count >= 0
            checkpoint = self.checkpoints.get(node.uuid)
            if checkpoint is not None:
                assert checkpoint.length == depth
            if not node.is_root():
                if node.ref_count:
                    protected += node.length
                else:
                    evictable += node.length
            nodes.extend((child, depth + child.length) for child in node.children.values())
        assert self.checkpoints.keys() <= seen
        assert (evictable, protected) == (self.evictable_size, self.protected_size)
        assert self.state_refs.keys() <= {c.slot for c in self.checkpoints.values()}
