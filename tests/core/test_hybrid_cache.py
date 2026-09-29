"""Exact hybrid endpoints, independent state ownership, and KV lifecycle."""

from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.kvcache.qwen4_pool import Qwen4KVCache, Qwen4StatePool
from minisgl.scheduler.cache import CacheManager


@pytest.fixture
def cm():
    old = core._GLOBAL_CTX
    working = SimpleNamespace(
        recurrent={0: torch.zeros(3, 2, 4, 4)},
        conv={0: torch.zeros(3, 4, 3, dtype=torch.bfloat16)},
        ple_conv={1: torch.zeros(3, 8, 9, dtype=torch.bfloat16)},
        ple_history={1: torch.zeros(3, 2, dtype=torch.int64)},
        pending=torch.zeros(1, 3, 3, 4),
        checkpoint_alignment=64,
        checkpoint_interval=128,
    )
    working.state_tensors = lambda: Qwen4KVCache.state_tensors(working)
    working.prefix_states = Qwen4StatePool(working, 2)
    ctx = core.Context(4)
    ctx.kv_cache = working
    core._GLOBAL_CTX = ctx
    manager = CacheManager(128, 4, torch.zeros(3, 512, dtype=torch.int32), "hybrid")
    yield manager
    core._GLOBAL_CTX = old


def insert(cm, length, value=1, ids=None):
    ids = torch.arange(length, dtype=torch.int32) if ids is None else ids
    for tensor in cm.prefix_cache.states.working.state_tensors():
        tensor[0].fill_(value)
    return cm.prefix_cache.insert_checkpoint(ids, torch.arange(length, dtype=torch.int32), 0)


def test_split_cannot_invent_state(cm):
    insert(cm, 128)
    cache = cm.prefix_cache
    assert cache.match_prefix(torch.arange(64, dtype=torch.int32)).cuda_handle.cached_len == 0
    assert cache.match_prefix(torch.arange(128, dtype=torch.int32)).cuda_handle.cached_len == 128
    insert(cm, 64, 2)
    handle = cache.match_prefix(torch.arange(100, dtype=torch.int32)).cuda_handle
    assert handle.cached_len == 64
    assert handle.checkpoint.length == 64
    cache.check_integrity()


def test_restore_is_private_and_preserves_every_dtype(cm):
    insert(cm, 64, 7)
    cache = cm.prefix_cache
    handle = cache.match_prefix(torch.arange(64, dtype=torch.int32)).cuda_handle
    cache.lock_handle(handle)
    cache.states.working.pending.fill_(99)
    cache.restore(handle, 1)
    for tensor in cache.states.working.state_tensors():
        assert (tensor[1] == 7).all()
        tensor[1].fill_(22)
    cache.restore(handle, 2)
    for tensor in cache.states.working.state_tensors():
        assert (tensor[2] == 7).all()
    assert (cache.states.working.pending[:, 2] == 0).all()
    cache.lock_handle(handle, unlock=True)


def test_duplicate_endpoint_does_not_overwrite(cm):
    first = insert(cm, 64, 3)
    second = insert(cm, 64, 9)
    assert first.handle.checkpoint is second.handle.checkpoint
    assert len(cm.prefix_cache.states.free_slots) == 1
    assert (cm.prefix_cache.states.tensors[0][first.handle.checkpoint.slot] == 3).all()


def test_state_eviction_falls_back_and_respects_locks(cm):
    cache = cm.prefix_cache
    first = insert(cm, 64).handle
    cache.lock_handle(first)
    insert(cm, 128)
    insert(cm, 192)
    assert cache.match_prefix(torch.arange(128, dtype=torch.int32)).cuda_handle.cached_len == 64
    second = cache.match_prefix(torch.arange(192, dtype=torch.int32)).cuda_handle
    cache.lock_handle(second)
    assert insert(cm, 256) is None
    assert cache.stats["skipped_saves"] == 1
    cache.check_integrity()
    cache.lock_handle(first, unlock=True)
    cache.lock_handle(second, unlock=True)


def test_kv_eviction_releases_state_slots(cm):
    insert(cm, 128)
    cm.prefix_cache.evict(128)
    assert not cm.prefix_cache.checkpoints
    cm.prefix_cache.check_integrity()


def test_descendant_kv_lock_does_not_pin_every_ancestor_state(cm):
    cache = cm.prefix_cache
    insert(cm, 64)
    descendant = insert(cm, 128).handle
    cache.lock_handle(descendant)
    # Both KV nodes are locked, but only the 128-token state is consumed.
    assert insert(cm, 192) is not None
    assert cache.match_prefix(torch.arange(64, dtype=torch.int32)).cuda_handle.cached_len == 0
    assert cache.match_prefix(torch.arange(128, dtype=torch.int32)).cuda_handle.cached_len == 128
    cache.lock_handle(descendant, unlock=True)
    cache.check_integrity()


@pytest.mark.parametrize(
    "start,end,prompt,expected",
    [
        (0, 64, 64, (64, 0)),
        (0, 128, 128, (64, 64)),
        (0, 129, 129, (128, 128)),
        (128, 256, 256, (192, 192)),
        (0, 3, 129, (3, 0)),
        (3, 129, 129, (128, 128)),
    ],
)
def test_replay_tail_and_small_token_budget(cm, start, end, prompt, expected):
    assert cm.plan_checkpoint(start, end, prompt) == expected


def test_publish_rebinds_duplicate_pages_and_finish_keeps_checkpoint(cm):
    empty = cm.prefix_cache.match_prefix(torch.arange(64, dtype=torch.int32)).cuda_handle
    reqs = [
        Req(
            torch.arange(64, dtype=torch.int32),
            i,
            0,
            1,
            i,
            SamplingParams(),
            empty,
            checkpoint_len=64,
        )
        for i in range(2)
    ]
    for req in reqs:
        cm.lock(empty)
        cm.allocate_paged([req])
        req.complete_one()
        assert cm.publish_checkpoint(req)
    assert torch.equal(cm.page_table[0, :64], cm.page_table[1, :64])
    assert len(cm.free_slots) == 128 - 16
    for req in reqs:
        cm.cache_req(req, finished=True)
    cm.check_integrity()
    assert (
        cm.prefix_cache.match_prefix(torch.arange(64, dtype=torch.int32)).cuda_handle.cached_len
        == 64
    )


def test_reject_non_boundary_or_untyped_insert(cm):
    with pytest.raises(AssertionError):
        insert(cm, 65)
    with pytest.raises(RuntimeError, match="forward-boundary"):
        cm.prefix_cache.insert_prefix(torch.arange(64), torch.arange(64))


def test_internal_checkpoint_avoids_splitting_and_handles_unaligned_start(cm):
    cm.prefix_cache.internal = True
    assert cm.plan_checkpoint(0, 2053, 2053) == (2053, 2048)
    # Tracking uses local chunk starts, not global positions. Fall back to a
    # final-state checkpoint when resuming an unaligned scheduler chunk.
    assert cm.plan_checkpoint(3, 2053, 2053) == (128, 128)


def test_branch_target_does_not_claim_missing_state(cm):
    insert(cm, 256)
    cache = cm.prefix_cache
    handle = cache.match_prefix(torch.arange(200, dtype=torch.int32)).cuda_handle
    assert handle.cached_len == 0
    assert handle.raw_hit_len == 200
    cache.internal = True
    assert cm.plan_checkpoint(0, 300, 300, handle.raw_hit_len) == (300, 192)
    # Once replay captured the actual state, another branch can resume there.
    insert(cm, 192)
    assert cache.match_prefix(torch.arange(200, dtype=torch.int32)).cuda_handle.cached_len == 192


def test_branch_target_survives_small_chunks_and_respects_local_grid(cm):
    cm.prefix_cache.internal = True
    assert cm.plan_checkpoint(0, 64, 300, 200) == (64, 0)
    assert cm.plan_checkpoint(64, 128, 300, 200) == (128, 128)
    assert cm.plan_checkpoint(128, 300, 300, 200) == (300, 192)
    assert cm.plan_checkpoint(129, 300, 300, 200) == (192, 192)


def test_prefill_budget_does_not_shift_recurrent_chunk_boundaries(monkeypatch):
    from minisgl.scheduler.prefill import ChunkedReq, PrefillAdder
    from minisgl.scheduler.utils import PendingReq

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    pending = PendingReq(0, torch.arange(8194, dtype=torch.int32), SamplingParams())
    cache = SimpleNamespace(plan_checkpoint=lambda start, end, prompt, branch: (end, 0))
    tables = SimpleNamespace(token_pool=torch.empty(1, 9000, dtype=torch.int32))
    adder = PrefillAdder(8190, 0, cache, tables, chunk_alignment=64)
    first = adder._add_one_req(pending, None, 0, 0)
    assert isinstance(first, ChunkedReq)
    assert first.device_len == 8128
    assert adder.token_budget == 62
    # Do not acquire a new working slot if the remaining budget cannot fit a
    # recurrent chunk. A later batch gets the full configured budget again.
    assert adder.try_add_one(pending) is None
    adder.token_budget = 64
    second = adder._add_one_req(pending, None, 0, first.device_len)
    assert second.device_len == 8192
    adder.token_budget = 64
    last = adder._add_one_req(pending, None, 0, second.device_len)
    assert not isinstance(last, ChunkedReq)
    assert last.extend_len == 2


@pytest.mark.parametrize("offset", [2, 9, 64])
def test_ple_window_captures_boundary_not_chunk_end(cm, offset):
    pool = cm.prefix_cache.states
    history = pool.working.ple_conv[1]
    history[2].copy_(torch.arange(72).reshape(8, 9))
    x = torch.arange(70 * 8).reshape(70, 8).bfloat16()
    expected = torch.cat((history[2], x.T), -1)[:, offset : offset + 9]
    metadata = SimpleNamespace(track_offsets=(offset,), spans=((2, 0, 70, 0, 70),))
    pool.capture_window("ple_conv", 1, x, metadata)
    torch.testing.assert_close(pool.ple_conv[1][0], expected, atol=0, rtol=0)
