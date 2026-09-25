"""Independent SGLang operator parity; SGLang is a test-only dependency."""

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
from minisgl.kernel import qwen4_sglang as mini


class TestNumericsSelection(unittest.TestCase):
    def test_default_and_explicit_override(self):
        from minisgl.models import qwen4_fast

        original = qwen4_fast.SGLANG_NUMERICS
        try:
            with (
                patch.dict(os.environ, {}, clear=False),
                patch.object(qwen4_fast, "ENABLED", True),
                patch("torch.cuda.get_device_capability", return_value=(10, 0)),
                patch("torch.version.cuda", "13.0"),
            ):
                os.environ.pop("MINISGL_QWEN4_SGLANG_NUMERICS", None)
                qwen4_fast.configure_numerics(torch.device("cuda"))
                self.assertTrue(qwen4_fast.SGLANG_NUMERICS)
                os.environ["MINISGL_QWEN4_SGLANG_NUMERICS"] = "0"
                qwen4_fast.configure_numerics(torch.device("cuda"))
                self.assertFalse(qwen4_fast.SGLANG_NUMERICS)
                os.environ["MINISGL_QWEN4_SGLANG_NUMERICS"] = "1"
                with self.assertRaises(ValueError):
                    qwen4_fast.configure_numerics(torch.device("cpu"))
        finally:
            qwen4_fast.SGLANG_NUMERICS = original


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(
    importlib.util.find_spec("sglang"), "SGLang checkout required for independent parity"
)
class TestSGLangNumerics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def tensor(self, *shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def test_grouped_norm_and_combine(self):
        from sglang.kernels.ops.elementwise.hc_combine import hc_combine
        from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm

        for rows in (1, 5, 128, 2053):
            x, w = self.tensor(rows, 10240), self.tensor(10240)
            norm = mini.grouped_norm(x, w, 2560)
            torch.testing.assert_close(norm, grouped_gemma_rmsnorm(x, w, 2560), atol=0, rtol=0)
            y, inject = self.tensor(rows, 2560), self.tensor(4, 10240) * 0.01
            torch.testing.assert_close(
                mini.gr_combine(x, y, norm, inject),
                hc_combine(y, x, norm, inject, 4, 2560),
                atol=0,
                rtol=0,
            )

    def test_gdn_l2(self):
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

        for rows in (1, 5, 128, 2053):
            x = self.tensor(rows, 8, 128)
            torch.testing.assert_close(mini.l2(x), l2norm_fwd(x), atol=0, rtol=0)

    def test_ple_decode_gate_conv_and_padding(self):
        import torch.nn.functional as F
        from sglang.kernels.ops.qwen4_ple import fused_qwen4_gate_value

        gate, value = self.tensor(4, 4, 1), self.tensor(4, 2560)
        torch.testing.assert_close(
            mini.ple_gate_value(gate, value).view(4, 4, 2560),
            fused_qwen4_gate_value(gate, value),
            atol=0,
            rtol=0,
        )
        slots = torch.tensor([2, 0, 3, 0], device="cuda", dtype=torch.int32)
        valid = torch.tensor([True, True, True, False], device="cuda")
        state = self.tensor(4, 10240, 6)
        expected = state.clone()
        weight = self.tensor(10240, 1, 3)
        for _ in range(8):
            x = self.tensor(4, 10240)
            x[-1].fill_(float("nan"))
            selected = slots[:3].long()
            full = torch.cat((expected[selected], x[:3, :, None]), -1)
            y = F.silu(F.conv1d(full, weight, groups=10240, dilation=3).squeeze(-1))
            expected[selected] = full[:, :, 1:]
            actual = mini.ple_conv_decode(x, weight, state, slots, valid, 3)
            torch.testing.assert_close(actual[:3], y, atol=0, rtol=0)
            torch.testing.assert_close(state, expected, atol=0, rtol=0)
            self.assertEqual(actual[-1].abs().max().item(), 0)

    def test_qsa_decode_compression_boundary(self):
        from minisgl.kernel.qwen4_qsa import compress_decode
        from sglang.kernels.ops.attention.qsa_indexer import qsa_index_k_compress_store

        batch, dim, capacity = 256, 128, 2060
        slots = torch.randperm(batch, device="cuda").to(torch.int32)
        positions = torch.full((batch,), 2055, device="cuda", dtype=torch.int64)
        valid = torch.arange(batch, device="cuda") % 7 != 0
        table = torch.arange(batch * capacity, device="cuda", dtype=torch.int32).view(
            batch, capacity
        )
        pending, raw, weight = self.tensor(batch, 3, dim), self.tensor(batch, dim), self.tensor(dim)
        before = pending.clone()
        cache = self.tensor(batch * capacity // 4, dim)
        expected = cache.clone()
        rope = mini.rope_cache(torch.device("cuda:0"), 64, 10000000.0, capacity)
        source = torch.cat((pending[slots.long()], raw[:, None]), dim=1).flatten(0, 1)
        group_locs = (
            torch.arange(batch * 4, device="cuda", dtype=torch.int32)
            .view(batch, 4)[valid]
            .contiguous()
        )
        rope_positions = torch.full((batch * 4, 3), 2052, device="cuda", dtype=torch.int64)
        write_locs = (table[slots.long(), 2052] // 4)[valid].contiguous()
        qsa_index_k_compress_store(
            source,
            group_locs,
            rope_positions,
            rope,
            torch.zeros(32, device="cuda", dtype=torch.int32),
            weight,
            write_locs,
            expected,
            4,
            64,
            1e-6,
            True,
        )
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
            10000000.0,
            1e-6,
            rope_cache=rope,
        )
        torch.testing.assert_close(cache, expected, atol=0, rtol=0)
        torch.testing.assert_close(pending, before, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
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
                10000000.0,
                1e-6,
                rope_cache=rope,
            )
        valid.zero_()
        raw.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(cache, expected, atol=0, rtol=0)
        torch.testing.assert_close(pending, before, atol=0, rtol=0)

    def test_moe_combine(self):
        from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add

        for rows in (1, 5, 128, 2053):
            x, w = self.tensor(rows, 2560), self.tensor(2560)
            shared, routed = self.tensor(rows, 2560), self.tensor(rows, 2560)
            expected = routed.clone()
            fused_gate_sigmoid_mul_add(x, w, shared, expected)
            torch.testing.assert_close(
                mini.moe_combine(x, w, shared, routed), expected, atol=0, rtol=0
            )

    def test_router(self):
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

        for rows in (1, 5, 128, 2053):
            logits = self.tensor(rows, 512)
            logits[0].zero_()  # Explicit equal-score tie ordering.
            expected = moe_fused_gate(logits, None, 10, scoring_func="softmax", renormalize=True)
            got = mini.router(logits, 10)
            for a, b in zip(got, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_qsa_norm_rope_and_gate(self):
        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            fused_qk_gemma_rmsnorm_rope_gate,
        )
        from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

        cache = mini.rope_cache(torch.device("cuda:0"), 64, 10000000.0, 4096)
        for rows in (1, 5, 128, 2053):
            qg, k = self.tensor(rows, 16 * 2 * 256), self.tensor(rows, 2 * 256)
            qw, kw = self.tensor(256), self.tensor(256)
            pos = torch.arange(rows, device="cuda") + 7
            q, gate = qg.view(rows, 16, 512).chunk(2, -1)
            expected_q, expected_k, expected_gate = fused_qk_gemma_rmsnorm_rope_gate(
                qg, k, qw, kw, cache, pos, 1e-6, 16, 2, 256, 64
            )
            torch.testing.assert_close(
                mini.norm_rope(q, qw, pos, cache).flatten(1), expected_q, atol=0, rtol=0
            )
            torch.testing.assert_close(
                mini.norm_rope(k.view(rows, 2, 256), kw, pos, cache).flatten(1),
                expected_k,
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                mini.sigmoid_mul(q.contiguous(), gate).flatten(1),
                fused_sigmoid_mul(q.contiguous().flatten(1), expected_gate),
                atol=0,
                rtol=0,
            )

    def test_qsa_index_norm_rope(self):
        from sglang.kernels.ops.attention.qsa_indexer import qsa_index_q_norm_rope_store

        for d in (64, 128, 256):
            cache = mini.rope_cache(torch.device("cuda:0"), 64, 10000000.0, 4096)
            rows, heads = 128, 8
            qk = self.tensor(rows, (heads + 1) * d)
            w = self.tensor(d)
            positions = torch.arange(rows, device="cuda") + 13
            expected = qsa_index_q_norm_rope_store(
                qk,
                positions,
                cache,
                torch.zeros(32, device="cuda", dtype=torch.int32),
                w,
                torch.arange(rows, device="cuda"),
                self.tensor(rows, d),
                torch.empty(rows, 3, device="cuda", dtype=torch.int64),
                heads,
                64,
                1e-6,
                True,
            )
            got = mini.index_norm_rope(
                qk[:, : heads * d].reshape(rows, heads, d), w, positions, cache
            )
            torch.testing.assert_close(got, expected, atol=0, rtol=0)

    def test_qsa_prefill(self):
        from minisgl.kernel.qwen4_qsa import sparse_gqa
        from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_fwd_interface_triton

        for rows in (5, 128, 513):
            q, k, v = (
                self.tensor(rows, 16, 256),
                self.tensor(rows, 2, 256),
                self.tensor(rows, 2, 256),
            )
            ids = torch.arange(rows, device="cuda", dtype=torch.int32)
            indices = torch.where(ids[None, :] <= ids[:, None], ids[None, :], -1).contiguous()
            cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
            expected = sparse_gqa_fwd_interface_triton(q, k, v, rows, indices, cu, 256**-0.5)
            table = ids.flip(0)[None].contiguous()
            got = sparse_gqa(
                q,
                k.flip(0),
                v.flip(0),
                indices,
                table,
                torch.zeros_like(ids),
                sglang_prefill_rows=rows,
            )
            torch.testing.assert_close(got, expected, atol=0, rtol=0)

    def test_qsa_index_compaction(self):
        from minisgl.attention.qwen4 import Qwen4Backend
        from minisgl.models import qwen4_fast

        selected = torch.arange(8, device="cuda").repeat(5, 1)
        lengths = torch.tensor([1, 3, 4, 7, 19], device="cuda")
        with patch.object(qwen4_fast, "SGLANG_NUMERICS", True):
            indices = Qwen4Backend.expand(selected, lengths // 4, lengths, None, 4)
        for row, length in enumerate(lengths.tolist()):
            torch.testing.assert_close(
                indices[row, :length], torch.arange(length, device="cuda", dtype=torch.int32)
            )
            self.assertTrue((indices[row, length:] == -1).all())

    def test_qsa_paged_index_scores(self):
        from minisgl.kernel.qwen4_qsa import index_scores_decode
        from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_decode

        q, keys = self.tensor(4, 4, 128), self.tensor(4 * 1024, 128)
        table = torch.arange(4 * 4096, device="cuda", dtype=torch.int32).view(4, 4096)
        slots = torch.tensor([2, 0, 1, 0], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([2056, 3000, 128, 3000], device="cuda", dtype=torch.int32)
        valid = torch.tensor([True, True, True, False], device="cuda")
        counts = torch.where(valid, lengths // 4, 0)
        expected = qsa_mqa_decode(
            q, keys.view(-1, 16, 1, 128), table[slots.long(), ::64] // 64, counts, 1024
        )
        actual = index_scores_decode(q, keys, table, slots, lengths, valid, 4, 2048)
        torch.testing.assert_close(actual[:2], expected[:2], atol=2e-5, rtol=2e-6)
        # Below budget the score GEMM is skipped, but every visible block is
        # retained; above budget the independent oracle chooses the same IDs.
        torch.testing.assert_close(
            mini.index_topk_deterministic(actual, counts),
            mini.index_topk_deterministic(expected, counts),
            atol=0,
            rtol=0,
        )

    def test_qsa_topk(self):
        from sglang.kernels.ops.elementwise.fast_topk import fast_topk

        scores = torch.rand((5, 1024), device="cuda", dtype=torch.float32)
        lengths = torch.tensor([0, 7, 512, 513, 1024], device="cuda", dtype=torch.int32)
        expected = fast_topk(scores, lengths, 512)
        got = mini.index_topk(scores, lengths)
        # Both radix implementations document atomic/unspecified row order.
        torch.testing.assert_close(got.sort(-1).values, expected.sort(-1).values, atol=0, rtol=0)
        torch.testing.assert_close(got[:3], expected[:3], atol=0, rtol=0)

    def test_qsa_deterministic_topk(self):
        scores = torch.rand((5, 1024), device="cuda", dtype=torch.float32)
        scores[:, 400:] = 0.0
        lengths = torch.tensor([0, 7, 512, 513, 1024], device="cuda", dtype=torch.int32)
        got = mini.index_topk_deterministic(scores, lengths)
        for row, length in enumerate(lengths.tolist()):
            expected = (
                scores[row, :length].argsort(descending=True, stable=True)[:512].sort().values
            )
            expected = torch.nn.functional.pad(expected, (0, 512 - expected.numel()), value=-1).to(
                torch.int32
            )
            torch.testing.assert_close(got[row], expected, atol=0, rtol=0)

    def test_qsa_decode_gather_and_padding(self):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from minisgl.kernel.qwen4_qsa import SparseDecode

        q, k, v = self.tensor(3, 16, 256), self.tensor(256, 2, 256), self.tensor(256, 2, 256)
        slots = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
        table = torch.randperm(256, device="cuda").to(torch.int32).view(4, 64)
        indices = torch.full((3, 67), -1, device="cuda", dtype=torch.int32)
        indices[0, :7] = torch.tensor([13, 9, 2, 3, 4, 5, 1], device="cuda")
        indices[2, :19] = torch.arange(19, device="cuda")
        decode = SparseDecode()
        got = decode(q, k, v, indices, table, slots)
        pk, pv = [torch.zeros((3, 128, 2, 256), device="cuda", dtype=q.dtype) for _ in range(2)]
        for row, count in ((0, 7), (2, 19)):
            loc = table[slots[row].long(), indices[row, :count].long()].long()
            pk[row, :count], pv[row, :count] = k[loc], v[loc]
        expected = trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=tuple(t.view(-1, 64, 2, 256).permute(0, 2, 1, 3) for t in (pk, pv)),
            workspace_buffer=torch.zeros_like(decode.workspace),
            block_tables=torch.arange(6, device="cuda", dtype=torch.int32).view(3, 2),
            seq_lens=torch.tensor([7, 0, 19], device="cuda", dtype=torch.int32),
            max_seq_len=128,
            bmm1_scale=256**-0.5,
            bmm2_scale=1.0,
        )
        self.assertTrue(torch.isfinite(got).all())
        expected[1].zero_()
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = decode(q, k, v, indices, table, slots)
        indices[0].fill_(-1)
        graph.replay()
        eager = decode(q, k, v, indices, table, slots)
        torch.testing.assert_close(captured, eager, atol=0, rtol=0)
        self.assertTrue((captured[:2] == 0).all())

    def test_gdn_output(self):
        from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd_1pass_kernel

        for tokens in (1, 5, 128, 2053):
            x, z, w = self.tensor(tokens, 24, 128), self.tensor(tokens, 24, 128), self.tensor(128)
            rows = tokens * 24
            sm = torch.cuda.get_device_properties(0).multi_processor_count
            block = min(triton.next_power_of_2(triton.cdiv(rows, 2 * sm)), 4)
            ref = torch.empty_like(x)
            rstd = torch.empty(rows, device="cuda")
            _layer_norm_fwd_1pass_kernel[(triton.cdiv(rows, block), 1)](
                x,
                ref,
                w,
                None,
                z,
                None,
                rstd,
                128,
                128,
                128,
                0,
                0,
                rows,
                128,
                1e-6,
                BLOCK_N=128,
                ROWS_PER_BLOCK=block,
                HAS_BIAS=False,
                HAS_Z=True,
                Z_IS_3D=False,
                Z_HEADS=1,
                NORM_BEFORE_GATE=True,
                IS_RMS_NORM=True,
                ACTIVATION="sigmoid",
                num_warps=1,
            )
            torch.testing.assert_close(mini.gdn_output(x, z, w), ref, atol=0, rtol=0)

    def test_gdn_chunk_state_and_slots(self):
        from minisgl.kernel._qwen4_fla.chunk import chunk_gated_delta_rule as actual
        from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule as golden

        q, k, v = (
            self.tensor(1, 128, 8, 128),
            self.tensor(1, 128, 8, 128),
            self.tensor(1, 128, 24, 128),
        )
        g = -torch.rand(1, 128, 24, device="cuda")
        beta = torch.rand(1, 128, 24, device="cuda").bfloat16().float()
        state = torch.randn(3, 24, 128, 128, device="cuda") * 0.1
        reference = state.clone()
        untouched = state[1].clone()
        slots = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 5, 128], device="cuda", dtype=torch.int32)
        kwargs = {"initial_state_indices": slots, "cu_seqlens": cu, "use_qk_l2norm_in_kernel": True}
        expected, _, _ = golden(q, k, v, g, beta, initial_state=reference, **kwargs)
        output, _, _ = actual(q, k, v, g, beta, initial_state=state, **kwargs)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.testing.assert_close(state, reference, atol=0, rtol=0)
        torch.testing.assert_close(state[1], untouched, atol=0, rtol=0)

    def test_gdn_decode_state_and_padding(self):
        from minisgl.kernel._qwen4_fla.packed_decode import (
            fused_recurrent_gated_delta_rule_packed_decode as actual,
        )
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode as golden,
        )

        x, a, b = self.tensor(3, 5120), self.tensor(3, 24), self.tensor(3, 24)
        log, dt = torch.randn(24, device="cuda"), self.tensor(24)
        state = torch.randn(3, 24, 128, 128, device="cuda") * 0.1
        ref_state = state.clone()
        untouched = state[1].clone()
        slots = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int32)
        out, ref = self.tensor(3, 1, 24, 128), self.tensor(3, 1, 24, 128)
        for _ in range(3):
            actual(x, a, b, log, dt, 128**-0.5, state, out, slots, True)
            golden(x, a, b, log, dt, 128**-0.5, ref_state, ref, slots, True)
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
            torch.testing.assert_close(state, ref_state, atol=0, rtol=0)
            torch.testing.assert_close(state[1], untouched, atol=0, rtol=0)
            self.assertEqual(out[1].abs().max().item(), 0)

    @torch.inference_mode()
    def test_gdn_model_path(self):
        from minisgl.models.qwen4_optimized import FastGDN

        pool = SimpleNamespace(
            recurrent={0: torch.zeros(4, 24, 128, 128, device="cuda")},
            conv={0: torch.zeros(4, 5120, 3, device="cuda", dtype=torch.bfloat16)},
        )
        meta = SimpleNamespace(
            slots=torch.tensor([2], device="cuda", dtype=torch.int32),
            valid=torch.ones(1, device="cuda", dtype=torch.bool),
            spans=((2, 0, 5, 0, 5),),
            cu_seqlens=torch.tensor([0, 5], device="cuda", dtype=torch.int32),
        )
        batch = SimpleNamespace(is_decode=False, attn_metadata=meta)
        ctx = SimpleNamespace(batch=batch, kv_cache=pool)
        op = SimpleNamespace(
            _packed=self.tensor(8240, 32) * 0.03,
            _widths=[5120, 3072, 24, 24],
            hv=24,
            dv=128,
            hk=8,
            dk=128,
            layer_id=0,
            conv1d=SimpleNamespace(weight=self.tensor(5120, 1, 4) * 0.1),
            A_log=torch.randn(24, device="cuda"),
            dt_bias=self.tensor(24),
            norm=SimpleNamespace(weight=self.tensor(128)),
            eps=1e-6,
            out_proj=SimpleNamespace(forward=lambda x: x),
        )
        with patch("minisgl.models.qwen4_optimized.get_global_ctx", return_value=ctx):
            out = FastGDN.forward_sglang(op, self.tensor(5, 32))
            self.assertEqual(out.shape, (5, 3072))
            self.assertTrue(torch.isfinite(out).all())
            batch.is_decode = True
            out = FastGDN.forward_sglang(op, self.tensor(1, 32))
            self.assertEqual(out.shape, (1, 3072))
            self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(pool.recurrent[0][0].abs().max().item(), 0)
        self.assertEqual(pool.conv[0][0].abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
