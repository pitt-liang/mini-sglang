"""GPU reference, padding and CUDA graph regression tests for native fast kernels."""

import unittest

import torch
from minisgl.kernel import qwen4_ops as fast
from minisgl.kernel.qwen4 import gated_delta_decode, gated_delta_rule
from minisgl.models import qwen4_ops as ref
from minisgl.models.config import Qwen4RuntimeConfig


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def tensor(self, *shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    def test_ple_lookup_host_and_device_shards(self):
        """UVA and device lookup must preserve all 160 values, including TP masks."""
        from minisgl.models.qwen4 import NGramTable

        weight = self.tensor(31, 160)
        ids = torch.tensor([0, 16, 17, 18, 47, 48, 100], device="cuda")
        expected = torch.zeros((ids.numel(), 160), device="cuda", dtype=weight.dtype)
        expected[2:5] = weight[torch.tensor([0, 1, 30], device="cuda")]
        for storage in ("gpu", "pinned"):
            table = object.__new__(NGramTable)
            table.vocab_range = (17, 31)
            table.tp_size = 1
            table.weight = weight if storage == "gpu" else weight.cpu().pin_memory()
            table.initialize_runtime(torch.device("cuda"))
            actual = table.lookup_local(ids)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            out = torch.empty_like(expected)
            self.assertIs(table.lookup_local(ids, out), out)
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            self.assertEqual(table.lookup_local(ids[:0]).shape, (0, 160))
            with self.assertRaises(ValueError):
                table.lookup_local(ids, out[:, :80].contiguous())

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                table.lookup_local(ids, out)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    table.lookup_local(ids, out)
                torch.cuda.current_stream().wait_stream(stream)
            ids[2] = 47
            graph.replay()
            torch.testing.assert_close(out[2], weight[30], atol=0, rtol=0)
            ids[2] = 17

    def test_ple_prefetch_preserves_history_and_graph_replay(self):
        """All storage/scheduling modes preserve chunk, EOS, padding and slot state."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from minisgl.models.qwen4 import PLE
        from minisgl.models.qwen4_weight import _parameters
        from sympy import nextprime

        def run(storage, prefetch, captured):
            torch.manual_seed(73)
            cfg = SimpleNamespace(
                qwen4_runtime=Qwen4RuntimeConfig(
                    aligned=True, ple_storage=storage, ple_prefetch=prefetch
                ),
                hidden_size=512,
                rms_norm_eps=1e-6,
                hybrid={
                    "ple_layer_ids": [2],
                    "hc_count": 4,
                    "ple_embed_dim": 640,
                    "ple_conv_kernel_size": 4,
                    "heads_per_ngram": 2,
                    "ngram_size": 3,
                    "ngram_vocab_size_base": 97,
                    "make_ngram_vocab_size_divisible_by": 1,
                    "eos_token_id": 2,
                },
            )
            with patch(
                "minisgl.layers.embedding.get_tp_info", return_value=SimpleNamespace(rank=0, size=1)
            ):
                ple = PLE(cfg, 1)
            for _, owner, attr, weight in _parameters(ple):
                if weight.is_floating_point():
                    setattr(owner, attr, self.tensor(*weight.shape) * 0.03)
            pe = ple.ple_embedding
            sizes, prime = [], 96
            for _ in range(4):
                prime = int(nextprime(prime))
                sizes.append(prime)
            pe.layer_multipliers = torch.tensor([31, 139, 241], device="cuda")
            pe.ngram_heads_vocab_sizes = torch.tensor(sizes, device="cuda")
            pe.ngram_heads_offsets = torch.tensor(
                [0, *torch.tensor(sizes).cumsum(0).tolist()[:-1]], device="cuda"
            )
            table = pe.ngram_embedding
            if storage == "pinned":
                table.weight = table.weight.cpu().pin_memory()
            ple.initialize_runtime(torch.device("cuda"))

            def reduce_on_main(x):
                if ple._prefetch_stream is not None:
                    self.assertNotEqual(torch.cuda.current_stream(), ple._prefetch_stream)
                return x

            table.tp_size = 2
            table._comm = SimpleNamespace(all_reduce=reduce_on_main)
            history = torch.full((3, 2), 2, device="cuda", dtype=torch.int64)
            conv = torch.zeros((3, 2048, 9), device="cuda", dtype=torch.bfloat16)
            ctx = SimpleNamespace(
                kv_cache=SimpleNamespace(ple_history={1: history}, ple_conv={1: conv})
            )
            result = []

            def forward(x):
                ple.start_prefetch()
                out = ple.forward(x)
                self.assertIsNone(ple._prefetch_state)
                return out

            with patch("minisgl.models.qwen4.get_global_ctx", return_value=ctx):
                for chunk in range(2):
                    ctx.batch = SimpleNamespace(
                        is_prefill=True,
                        input_ids=torch.tensor(
                            [4, 5, 2, 7, 8, 9, 2], device="cuda", dtype=torch.int32
                        ),
                        attn_metadata=SimpleNamespace(
                            spans=[(0, chunk * 4, 4, 0, 4), (1, chunk * 3, 3, 4, 7)],
                            track_offsets=(),
                        ),
                    )
                    result.extend(
                        [forward(self.tensor(7, 2048)).clone(), history.clone(), conv.clone()]
                    )
                ctx.batch = SimpleNamespace(
                    is_prefill=False,
                    input_ids=torch.tensor([7, 2, 0, 0], device="cuda", dtype=torch.int32),
                    attn_metadata=SimpleNamespace(
                        slots=torch.tensor([1, 0, 2, 2], device="cuda", dtype=torch.int32),
                        valid=torch.tensor([True, True, False, False], device="cuda"),
                    ),
                )
                x = self.tensor(4, 2048)
                if captured:
                    saved_history, saved_conv = history.clone(), conv.clone()
                    forward(x)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = forward(x)
                    history.copy_(saved_history)
                    conv.copy_(saved_conv)
                for step in range(8):
                    ctx.batch.input_ids[:2].copy_(
                        torch.tensor([step + 3, 2 if step == 3 else step + 7], device="cuda")
                    )
                    ctx.batch.attn_metadata.slots[:2].copy_(
                        torch.tensor([step % 2, 1 - step % 2], device="cuda")
                    )
                    if step == 4:  # cancellation followed by slot reuse
                        history[0].fill_(2)
                        conv[0].zero_()
                    if captured:
                        graph.replay()
                    else:
                        output = forward(x)
                    result.extend([output.clone(), history.clone(), conv.clone()])
                torch.testing.assert_close(
                    history[2], torch.full_like(history[2], 2), atol=0, rtol=0
                )
                torch.testing.assert_close(conv[2], torch.zeros_like(conv[2]), atol=0, rtol=0)
            torch.cuda.synchronize()
            return result

        expected = run("gpu", False, False)
        for storage in ("gpu", "pinned"):
            for prefetch in (False, True):
                for captured in (False, True):
                    with self.subTest(storage=storage, prefetch=prefetch, graph=captured):
                        actual = run(storage, prefetch, captured)
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_ple_loader_streams_only_owned_rows_to_pinned_memory(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch

        from minisgl.distributed import DistributedInfo
        from minisgl.models.qwen4_weight import load_qwen4_weights
        from safetensors.torch import save_file

        weight = torch.arange(7 * 160, dtype=torch.float32).reshape(7, 160).bfloat16()
        prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
        shards = {prefix + "shard_0.weight": weight[:3], prefix + "shard_1.weight": weight[3:]}
        table = SimpleNamespace(
            weight=torch.empty((4, 160), device="meta", dtype=torch.bfloat16),
            vocab_range=(3, 4),
            num_embeddings=7,
        )
        model = SimpleNamespace(
            runtime=Qwen4RuntimeConfig(ple_storage="pinned"),
            config=SimpleNamespace(
                hybrid={
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 1,
                    "linear_num_value_heads": 1,
                    "linear_value_head_dim": 1,
                    "split_ngram_parts": 2,
                }
            ),
        )
        parameter = (
            "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
            table,
            "weight",
            table.weight,
        )
        with TemporaryDirectory() as directory:
            save_file(shards, str(Path(directory) / "weights.safetensors"))
            (Path(directory) / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(shards, "weights.safetensors")})
            )
            before = torch.cuda.memory_allocated()
            with (
                patch("minisgl.models.qwen4_weight._parameters", return_value=[parameter]),
                patch("minisgl.models.qwen4_weight.download_hf_weight", return_value=directory),
                patch("minisgl.distributed.info._TP_INFO", DistributedInfo(1, 2)),
            ):
                load_qwen4_weights(model, directory, torch.device("cuda"))
            self.assertEqual(torch.cuda.memory_allocated(), before)
        self.assertEqual(table.weight.device.type, "cpu")
        self.assertTrue(table.weight.is_pinned())
        torch.testing.assert_close(table.weight, weight[3:], atol=0, rtol=0)

    def test_norm_rope(self):
        for h, d, rd in [(12, 256, 64), (4, 128, 64)]:
            x = self.tensor(5, h, d * 2)[..., :d]
            w = self.tensor(d) * 0.1
            p = torch.tensor([0, 3, 2048, 8191, 131071], device="cuda")
            expected = ref.rotary(ref.rms_norm(x, w), p, rd, 10000000.0)
            actual = fast.legacy_norm(x, w, positions=p, rotary_dim=rd, base=10000000)
            torch.testing.assert_close(actual, expected, atol=0.032, rtol=0.02)
        x = self.tensor(7, 10240)
        w = self.tensor(10240) * 0.1
        torch.testing.assert_close(
            fast.legacy_norm(x, w, group_size=2560),
            ref.rms_norm(x, w, group_size=2560),
            atol=0.016,
            rtol=0.01,
        )

    def test_gr(self):
        from types import SimpleNamespace

        from minisgl.models.qwen4 import GatedResidual

        cfg = SimpleNamespace(
            qwen4_runtime=Qwen4RuntimeConfig(aligned=False),
            hidden_size=2560,
            hybrid={"hc_count": 4, "hc_lowrank": 320},
            rms_norm_eps=1e-6,
        )
        op = GatedResidual(cfg)
        for obj in [
            op.hc_norm,
            op.input_mix_weight_down,
            op.input_mix_weight_up,
            op.block_inject_weight,
        ]:
            obj.weight = self.tensor(*obj.weight.shape) * 0.01
        x, y = self.tensor(4, 10240), self.tensor(4, 2560)
        mixed, gate = op._forward_eager(x)
        expected = op._combine_eager(x, y, gate)
        actual, norm = op.forward(x)
        torch.testing.assert_close(actual, mixed, atol=0.008, rtol=0.02)
        torch.testing.assert_close(op.combine(x, y, norm), expected, atol=0.032, rtol=0.02)

    def test_index_compression_padding(self):
        from minisgl.kernel.qwen4_qsa import compress_decode

        slots = torch.tensor([2, 0, 3, 3], device="cuda", dtype=torch.int32)
        valid = torch.tensor([True, True, False, False], device="cuda")
        table = torch.arange(4 * 16, device="cuda", dtype=torch.int32).view(4, 16)
        pending = self.tensor(4, 3, 128)
        cache = self.tensor(16, 128)
        expected_pending, expected_cache = pending.clone(), cache.clone()
        weight = self.tensor(128) * 0.1
        for step in range(8):
            raw = self.tensor(4, 256)[:, :128]
            positions = torch.tensor([step, step + 4, 0, 0], device="cuda")
            for row, slot in enumerate([2, 0]):
                pos = step + row * 4
                if pos % 4 == 3:
                    pooled = torch.cat((expected_pending[slot], raw[row : row + 1]))
                    pooled = pooled.float().mean(0).to(raw.dtype)
                    normalized = ref.rms_norm(pooled, weight).view(1, 1, 128)
                    encoded = ref.rotary(normalized, positions[row : row + 1] - 3, 64, 10000000)
                    expected_cache[table[slot, pos - 3] // 4] = encoded.flatten()
                else:
                    expected_pending[slot, pos % 4] = raw[row]
            compress_decode(
                raw,
                weight,
                pending,
                cache,
                table,
                slots,
                positions,
                valid,
                4,
                64,
                10000000,
                1e-6,
            )
            torch.testing.assert_close(pending, expected_pending, atol=0, rtol=0)
            torch.testing.assert_close(cache, expected_cache, atol=0.016, rtol=0.01)

    def test_conv_hash_padding(self):
        slots = torch.tensor([2, 0, 3, 3], device="cuda", dtype=torch.int32)
        valid = torch.tensor([True, True, False, False], device="cuda")
        for dil in [1, 3]:
            x, w = self.tensor(4, 512), self.tensor(512, 1, 4)
            hist = self.tensor(4, 512, 3 * dil)
            expected = hist.clone()
            ys = []
            for row, slot in enumerate([2, 0]):
                ys.append(ref.causal_conv(x[row : row + 1], w, expected[slot], dil))
            actual = fast.legacy_conv_decode(x, w, hist, slots, valid, dil)
            torch.testing.assert_close(actual[:2], torch.cat(ys), atol=0.063, rtol=0.02)
            torch.testing.assert_close(hist, expected, atol=0, rtol=0)
            self.assertEqual(actual[2:].count_nonzero().item(), 0)
        eos = 17
        hist = torch.tensor([[3, 5], [4, 7], [eos, 8], [6, 9]], device="cuda", dtype=torch.int64)
        expected = hist.clone()
        mul = torch.tensor([123456789012345, 99123123456789, 991234567890123], device="cuda")
        sizes = torch.tensor([101, 103, 107, 109], device="cuda")
        offsets = torch.tensor([0, 101, 204, 311], device="cuda")
        for tokens in [[9, 17, 0, 0], [17, 6, 0, 0], [2, 4, 0, 0]]:
            ids = torch.tensor(tokens, device="cuda")
            ys = [
                ref.ngram_ids(ids[i : i + 1], expected[s], mul, sizes, offsets, 2, eos)
                for i, s in enumerate([2, 0])
            ]
            actual = fast.hash_decode(ids, hist, mul, sizes, offsets, 2, eos, slots, valid)
            torch.testing.assert_close(actual[:2], torch.cat(ys), atol=0, rtol=0)
            torch.testing.assert_close(hist, expected, atol=0, rtol=0)

    def test_sparse_decode(self):
        import torch.nn.functional as F
        from minisgl.kernel.qwen4_qsa import sparse_gqa

        q, k, v = self.tensor(4, 12, 256), self.tensor(4096, 1, 256), self.tensor(4096, 1, 256)
        table = torch.randperm(4096, device="cuda").to(torch.int32).view(4, 1024)
        slots = torch.tensor([2, 0, 1, 3], device="cuda", dtype=torch.int32)
        indices = torch.full((4, 515), -1, device="cuda", dtype=torch.int32)
        for row, count in enumerate([3, 128, 515]):
            indices[row, :count] = torch.randperm(1024, device="cuda")[:count]
        actual = sparse_gqa(q, k, v, indices, table, slots, decode=True)
        for row, count in enumerate([3, 128, 515]):
            loc = table[slots[row], indices[row, :count]].long()
            expected = F.scaled_dot_product_attention(
                q[row][None, :, None],
                k[loc].transpose(0, 1)[None],
                v[loc].transpose(0, 1)[None],
                enable_gqa=True,
            )[0, :, 0]
            torch.testing.assert_close(actual[row], expected, atol=0.016, rtol=0.02)
        self.assertEqual(actual[3].count_nonzero().item(), 0)

    def test_decode_selector(self):
        from minisgl.kernel.qwen4_qsa import select_decode

        table = torch.arange(4 * 4096, device="cuda", dtype=torch.int32).view(4, 4096)
        q, keys = self.tensor(4, 4, 128), self.tensor(4096, 128)
        slots = torch.tensor([1, 0, 2, 2], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([130, 2053, 2049, 1], device="cuda", dtype=torch.int32)
        valid = torch.tensor([True, True, True, False], device="cuda")
        actual = select_decode(q, keys, table, slots, lengths, valid, 4, 2048)
        for row, (slot, length) in enumerate([(1, 130), (0, 2053), (2, 2049)]):
            nb = length // 4
            loc = table[slot, : nb * 4 : 4].long() // 4
            scores = (q[row].float() @ keys[loc].float().T).relu().sum(0)
            chosen = scores.topk(512).indices if nb > 512 else torch.arange(nb, device="cuda")
            expected = torch.cat(
                (
                    (chosen[:, None] * 4 + torch.arange(4, device="cuda")).flatten(),
                    torch.arange(nb * 4, length, device="cuda"),
                )
            )
            torch.testing.assert_close(
                actual[row, : len(expected)].long().sort().values,
                expected.sort().values,
                atol=0,
                rtol=0,
            )
            self.assertTrue((actual[row, len(expected) :] == -1).all())
        self.assertTrue((actual[3] == -1).all())

    def test_shared_activation(self):
        import torch.nn.functional as F

        x, gu, w = self.tensor(4, 2560), self.tensor(4, 640), self.tensor(1, 2560) * 0.01
        a, gate = fast.legacy_shared_activation(x, gu, w)
        g, u = gu.chunk(2, -1)
        torch.testing.assert_close(a, F.silu(g) * u, atol=0.016, rtol=0.01)
        torch.testing.assert_close(
            gate, F.linear(x.float(), w.float()).to(x.dtype).sigmoid(), atol=0.004, rtol=0.01
        )
        packed = self.tensor(4, 5120)
        r, s = packed.chunk(2, -1)
        torch.testing.assert_close(
            fast.legacy_moe_combine(packed, gate), r + s * gate, atol=0, rtol=0
        )

    def test_moe_decode_launch_bound(self):
        from unittest.mock import patch

        from minisgl.moe.fused import FusedMoe

        w1, w2 = self.tensor(512, 128, 128) * 0.02, self.tensor(512, 128, 64) * 0.02
        op = FusedMoe()
        for batch in [1, 4, 16]:
            x, router = self.tensor(batch, 128), self.tensor(batch, 512)
            actual = op.forward(x.clone(), w1, w2, router, 10, True)
            # Restore the pre-optimization worst-case launch, with identical
            # buffers/weights/routing. Bound changes must be value-exact.
            with patch(
                "minisgl.kernel.moe_impl.min", lambda reserved, bound: reserved, create=True
            ):
                expected = op.forward(x.clone(), w1, w2, router, 10, True)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_chunked_prefill_fp32_state(self):
        if (
            int(torch.version.cuda.split(".")[0]) < 13
            or torch.cuda.get_device_capability()[0] != 10
        ):
            self.skipTest("FlashInfer Blackwell chunk prefill requires CUDA 13")
        from minisgl.kernel.qwen4 import prefill_delta

        q, k = [ref.l2_norm(self.tensor(129, 8, 128)) for _ in range(2)]
        v = self.tensor(129, 24, 128)
        g = -torch.rand(129, 24, device="cuda")
        beta = torch.rand(129, 24, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(24, 128, 128, device="cuda") * 0.1
        expected = state.clone()
        target = gated_delta_rule(q, k, v, g, beta, expected)
        actual = prefill_delta(q, k, v, g, beta, state)
        self.assertEqual(state.dtype, torch.float32)
        torch.testing.assert_close(actual, target, atol=0.001, rtol=0.03)
        torch.testing.assert_close(state, expected, atol=0.005, rtol=0.03)

    def test_chunked_prefill_slot_pool(self):
        if (
            int(torch.version.cuda.split(".")[0]) < 13
            or torch.cuda.get_device_capability()[0] != 10
        ):
            self.skipTest("FlashInfer Blackwell chunk prefill requires CUDA 13")
        from minisgl.kernel.qwen4 import prefill_delta_pool

        q, k = [ref.l2_norm(self.tensor(256, 8, 128)) for _ in range(2)]
        v = self.tensor(256, 24, 128)
        g = -torch.rand(256, 24, device="cuda")
        beta = torch.rand(256, 24, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(4, 24, 128, 128, device="cuda") * 0.1
        expected = state.clone()
        cu = torch.tensor([0, 129, 134, 256], device="cuda", dtype=torch.int32)
        slots = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32)
        target = torch.cat(
            [
                gated_delta_rule(q[a:b], k[a:b], v[a:b], g[a:b], beta[a:b], expected[slot])
                for a, b, slot in [(0, 129, 2), (129, 134, 0), (134, 256, 3)]
            ]
        )
        actual = prefill_delta_pool(q, k, v, g, beta, state, slots, cu)
        torch.testing.assert_close(actual, target, atol=0.001, rtol=0.03)
        torch.testing.assert_close(state, expected, atol=0.005, rtol=0.03)
        torch.testing.assert_close(state[1], expected[1], atol=0, rtol=0)

    def test_gdn_gates(self):
        import torch.nn.functional as F

        packed = self.tensor(4, 8240)
        a, b = packed[:, -48:-24], packed[:, -24:]
        log, dt = torch.randn(24, device="cuda"), torch.randn(24, device="cuda")
        g, beta = fast.legacy_gdn_gates(a, b, log, dt)
        torch.testing.assert_close(g, -log.exp() * F.softplus(a.float() + dt), atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(beta, b.sigmoid(), atol=0, rtol=0)

    def test_delta_replay_slots(self):
        slots = torch.tensor([2, 0, 3, 3], device="cuda", dtype=torch.int32)
        valid = torch.tensor([True, True, False, False], device="cuda")
        q, k = [ref.l2_norm(self.tensor(4, 2, 128)) for _ in range(2)]
        v = self.tensor(4, 6, 128)
        g = -torch.rand(4, 6, device="cuda")
        beta = torch.rand(4, 6, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(4, 6, 128, 128, device="cuda") * 0.1
        expected = state.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gated_delta_decode(q, k, v, g, beta, state, slots, valid)
        torch.cuda.current_stream().wait_stream(stream)
        state.copy_(expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = gated_delta_decode(q, k, v, g, beta, state, slots, valid)
        for mapping in [[2, 0, 3, 3], [1, 2, 3, 3], [0, 1, 3, 3]]:
            slots.copy_(torch.tensor(mapping, device="cuda", dtype=torch.int32))
            ys = [
                gated_delta_rule(
                    q[i : i + 1],
                    k[i : i + 1],
                    v[i : i + 1],
                    g[i : i + 1],
                    beta[i : i + 1],
                    expected[s],
                )
                for i, s in enumerate(mapping[:2])
            ]
            graph.replay()
            torch.testing.assert_close(out[:2], torch.cat(ys), atol=0, rtol=0)
            torch.testing.assert_close(state, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
