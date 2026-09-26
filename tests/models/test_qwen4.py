"""Run: python -m unittest discover -s tests/models -p test_qwen4.py -v"""

import ast
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from minisgl.models.config import ModelConfig, Qwen4RuntimeConfig
from minisgl.models.qwen4_ops import causal_conv, l2_norm, ngram_ids, rms_norm, rotary
from minisgl.utils.hf import cached_load_hf_config


def independent_hf_reference():
    """Execute selected unmodified upstream definitions, without importing its new deps.

    The path is configurable; no reference implementation is copied into mini-sglang.
    Missing external source skips ONLY the independent-reference tests.
    """
    source = os.environ.get("QWEN4_HF_SOURCE")
    if not source or not Path(source).is_file():
        raise unittest.SkipTest(
            "Set QWEN4_HF_SOURCE to the independent Transformers implementation"
        )
    path = Path(source)
    names = {
        "l2norm",
        "torch_recurrent_gated_delta_rule",
        "torch_chunk_gated_delta_rule",
        "Qwen4ExpTextRMSNorm",
        "Qwen4ExpTextGatedResidual",
        "Qwen4ExpTextNGramEmbedding",
        "Qwen4ExpTextExperts",
    }
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    ns = {"torch": torch, "nn": torch.nn, "F": F, "Qwen4ExpTextConfig": object, "Cache": object}
    ns["ACT2FN"] = {"silu": F.silu}
    code = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), ns)
    return ns


class Qwen4OpsTest(unittest.TestCase):
    def test_overlap_length_uses_committed_tokens(self):
        from minisgl.core import Req, SamplingParams
        from minisgl.scheduler.scheduler import Scheduler

        req = Req(torch.tensor([1, 2]), 0, 0, 3, 0, SamplingParams(ignore_eos=True), None)
        # Two device forwards finished; only first host result is being committed.
        req.complete_one()
        req.complete_one()
        replies = []
        scheduler = object.__new__(Scheduler)
        scheduler.finished_reqs = set()
        scheduler.eos_token_ids = {17}
        scheduler.cache_manager = SimpleNamespace(
            lazy_free_region=nullcontext, cache_req=lambda *a, **kw: None
        )
        scheduler.decode_manager = SimpleNamespace(remove_req=lambda req: None)
        scheduler._free_req_resources = lambda req: None
        scheduler.send_result = lambda reply: replies.extend(reply)
        batch = SimpleNamespace(reqs=[req], is_prefill=False)
        event = SimpleNamespace(synchronize=lambda: None)

        def commit(token):
            scheduler._process_last_data(
                (SimpleNamespace(batch=batch), (None, torch.tensor([token]), event))
            )

        commit(3)
        self.assertFalse(replies[-1].finished)
        req.complete_one()  # device can_decode is now false, but second host token isn't the last
        commit(4)
        self.assertFalse(replies[-1].finished)
        commit(5)
        self.assertTrue(replies[-1].finished)
        self.assertEqual(req.input_ids.tolist(), [1, 2, 3, 4, 5])
        commit(6)  # extra overlapped result after EOS/completion must be discarded
        self.assertEqual(len(replies), 3)

    def test_existing_llama_config(self):
        from transformers import LlamaConfig

        c = ModelConfig.from_hf(LlamaConfig(architectures=["LlamaForCausalLM"]))
        self.assertFalse(c.is_qwen4 or c.is_moe)
        self.assertEqual(c.num_kv_layers, c.num_layers)
        self.assertEqual(c.rotary_config.rotary_dim, c.head_dim)

    def test_metadata_is_a_request_snapshot(self):
        from minisgl.attention.qwen4 import Qwen4Backend

        req = SimpleNamespace(table_idx=3, cached_len=5, device_len=9, extend_len=4)
        batch = SimpleNamespace(reqs=[req])
        ctx = SimpleNamespace(kv_cache=SimpleNamespace(device="cpu"))
        with patch("minisgl.attention.qwen4.get_global_ctx", return_value=ctx):
            Qwen4Backend(
                SimpleNamespace(qwen4_runtime=Qwen4RuntimeConfig(aligned=False))
            ).prepare_metadata(batch)
        req.cached_len, req.device_len = 9, 10
        self.assertEqual(batch.attn_metadata.spans, ((3, 5, 9, 0, 4),))
        self.assertEqual(batch.attn_metadata.last_indices.tolist(), [3])

    def test_real_config(self):
        path = os.environ.get("QWEN4_MODEL")
        if not path or not Path(path).is_dir():
            self.skipTest("Set QWEN4_MODEL to a local Qwen4 checkpoint")
        c = ModelConfig.from_hf(cached_load_hf_config(path))
        self.assertEqual(cached_load_hf_config(path).dtype, torch.bfloat16)
        self.assertTrue(c.is_qwen4 and c.is_moe and c.norm_topk_prob)
        self.assertEqual((c.num_layers, c.num_kv_layers, c.rotary_config.rotary_dim), (48, 12, 64))
        from minisgl.kvcache.qwen4_pool import recurrent_bytes

        self.assertGreater(recurrent_bytes(c, 1, 1, torch.bfloat16), 108 * 2**20)
        self.assertLess(recurrent_bytes(c, 1, 2, torch.bfloat16), 56 * 2**20)

    def test_norm_and_gr_against_hf(self):
        from minisgl.models.qwen4 import GatedResidual

        ref = independent_hf_reference()
        torch.manual_seed(5)
        cfg = SimpleNamespace(
            qwen4_runtime=Qwen4RuntimeConfig(reference=True, aligned=False),
            hidden_size=32,
            hybrid={"hc_count": 4, "hc_lowrank": 8},
            hc_count=4,
            hc_lowrank=8,
            rms_norm_eps=1e-6,
        )
        mine = GatedResidual(cfg)
        golden = ref["Qwen4ExpTextGatedResidual"](cfg)
        for name, value in golden.state_dict().items():
            value.normal_(0, 0.1)
        mine.load_state_dict(dict(golden.state_dict()))
        x = torch.randn(7, 128)
        expected, residual, injection = golden(x)
        actual, gate = mine.forward(x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(gate, injection)
        torch.testing.assert_close(
            mine.combine(x, actual, gate),
            (residual.view(7, 4, 32) + injection[..., None] * expected[:, None]).flatten(1),
        )

    def test_ngram_eos_and_chunks_against_hf(self):
        ref = independent_hf_reference()
        cls = ref["Qwen4ExpTextNGramEmbedding"]
        obj = object.__new__(cls)
        obj.eos_token_id = eos = 17
        ids = torch.tensor([2, 5, eos, 9, 8, 7, eos, eos, 4])
        multipliers = torch.tensor([123456789012345, 123123123456789, 991234567890123])
        sizes, offsets = torch.tensor([101, 103, 107, 109]), torch.tensor([0, 101, 204, 311])
        history = torch.full((2,), eos)
        tokens = torch.cat((history, ids))[None]
        shifted = [obj._shift_right_ignore_eos(tokens, i)[0] for i in range(3)]
        expected = []
        for n in (2, 3):
            mixed = shifted[0] * multipliers[0]
            for i in range(1, n):
                mixed ^= shifted[i] * multipliers[i]
            sl = slice((n - 2) * 2, (n - 1) * 2)
            expected.append(mixed[:, None].remainder(sizes[sl]) + offsets[sl])
        expected = torch.cat(expected, -1)[2:]
        actual = ngram_ids(ids, history, multipliers, sizes, offsets, 2, eos)
        torch.testing.assert_close(actual, expected)
        history.fill_(eos)
        chunks = [
            ngram_ids(chunk, history, multipliers, sizes, offsets, 2, eos)
            for chunk in ids.split([1, 3, 2, 3])
        ]
        torch.testing.assert_close(torch.cat(chunks), actual)

    def test_conv_arbitrary_chunks(self):
        torch.manual_seed(4)
        for dilation in (1, 3):
            x, weight = torch.randn(19, 8), torch.randn(8, 1, 4)
            state = torch.zeros(8, 3 * dilation)
            full = causal_conv(x, weight, state, dilation)
            final = state.clone()
            state.zero_()
            chunks = [
                causal_conv(chunk, weight, state, dilation) for chunk in x.split([1, 7, 2, 9])
            ]
            torch.testing.assert_close(torch.cat(chunks), full)
            torch.testing.assert_close(state, final)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_moe_against_hf_experts(self):
        from minisgl.moe.fused import FusedMoe

        ref = independent_hf_reference()
        cfg = SimpleNamespace(
            qwen4_runtime=Qwen4RuntimeConfig(reference=True, aligned=False),
            num_experts=8,
            hidden_size=128,
            moe_intermediate_size=64,
            hidden_act="silu",
        )
        golden = ref["Qwen4ExpTextExperts"](cfg).to(device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(21)
        with torch.no_grad():
            for parameter in golden.parameters():
                parameter.normal_(0, 0.03)
            x = torch.randn(7, 128, dtype=torch.bfloat16, device="cuda")
            router = torch.randn(7, 8, dtype=torch.bfloat16, device="cuda")
            probability, selected = router.float().softmax(-1).topk(2, -1)
            probability /= probability.sum(-1, keepdim=True)
            expected = golden(x, selected, probability.to(x.dtype))
            actual = FusedMoe().forward(
                x.clone(), golden.gate_up_proj, golden.down_proj, router, topk=2, renormalize=True
            )
            torch.testing.assert_close(actual, expected, atol=5e-4, rtol=0.03)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_bf16_narrow_gate_chunk_stability(self):
        from minisgl.models.qwen4 import NarrowLinear

        torch.manual_seed(19)
        op = NarrowLinear(10240, 4, False)
        op.weight = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16) * 0.02
        x = torch.randn(5, 10240, device="cuda", dtype=torch.bfloat16)
        whole = op.forward(x)
        chunked = torch.cat([op.forward(t) for t in x.split([3, 2])])
        torch.testing.assert_close(whole, chunked, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_qsa_sparse_chunks_and_paged_storage(self):
        from minisgl.attention.qwen4 import Qwen4Backend, Qwen4Metadata
        from minisgl.kvcache.qwen4_pool import Qwen4KVCache

        torch.manual_seed(12)
        c = {
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "layer_types": ["full_attention"],
            "linear_conv_kernel_dim": 4,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 8,
            "indexer_budget": 8,
            "ple_layer_ids": [],
            "eos_token_id": 17,
        }
        cfg = SimpleNamespace(
            qwen4_runtime=Qwen4RuntimeConfig(aligned=False),
            hybrid=c,
            num_kv_heads=2,
            num_kv_layers=1,
            head_dim=32,
            rms_norm_eps=1e-6,
            rotary_config=SimpleNamespace(rotary_dim=4, base=10000.0),
        )
        tp = SimpleNamespace(rank=0, size=1)
        with (
            patch("minisgl.kvcache.qwen4_pool.get_tp_info", return_value=tp),
            patch("minisgl.kvcache.mha_pool.get_tp_info", return_value=tp),
        ):
            pool = Qwen4KVCache(cfg, 8, 4, torch.float32, torch.device("cuda"), 2)
        # Noncontiguous page order catches logical/physical compressed-block mixups.
        table = (
            (
                (torch.tensor([[3, 0, 6, 2], [1, 7, 4, 5]], device="cuda")[:, :, None] * 4)
                + torch.arange(4, device="cuda")
            )
            .flatten(1)
            .int()
        )
        ctx = SimpleNamespace(kv_cache=pool, page_table=table)
        backend = Qwen4Backend(cfg)
        n = 13
        q = torch.randn(n, 4, 32, device="cuda")
        k, v = [torch.randn(n, 2, 32, device="cuda") for _ in range(2)]
        iq, raw = torch.randn(n, 4, 8, device="cuda"), torch.randn(n, 8, device="cuda")
        weight = torch.randn(8, device="cuda") * 0.1
        pooled = raw[:12].view(3, 4, 8).mean(1)
        pooled = rotary(
            rms_norm(pooled, weight)[:, None], torch.arange(0, 12, 4, device="cuda"), 4, 10000.0
        )[:, 0]
        expected = []
        for t in range(n):
            complete = (t + 1) // 4
            score = (iq[t] @ pooled[:complete].T).relu().sum(0)
            blocks = score.topk(min(2, complete)).indices
            selected = torch.cat(
                (
                    (blocks[:, None] * 4 + torch.arange(4, device="cuda")).flatten(),
                    torch.arange(complete * 4, t + 1, device="cuda"),
                )
            )
            keys = k[selected].repeat_interleave(2, 1).transpose(0, 1)
            values = v[selected].repeat_interleave(2, 1).transpose(0, 1)
            probs = (q[t, :, None] @ keys.transpose(-1, -2) / 32**0.5).softmax(-1)
            expected.append((probs @ values)[:, 0])
        expected = torch.stack(expected)

        def run(chunks, slot):
            pool.reset(slot)
            result, start = [], 0
            with patch("minisgl.attention.qwen4.get_global_ctx", return_value=ctx):
                for length in chunks:
                    end = start + length
                    ctx.batch = SimpleNamespace(
                        out_loc=table[slot, start:end],
                        attn_metadata=Qwen4Metadata(
                            ((slot, start, end, 0, length),),
                            torch.tensor([length - 1], device="cuda"),
                        ),
                    )
                    result.append(
                        backend.qsa(
                            q[start:end],
                            k[start:end],
                            v[start:end],
                            iq[start:end],
                            raw[start:end],
                            weight,
                            0,
                        )
                    )
                    start = end
            return torch.cat(result)

        torch.testing.assert_close(run([13], 0), expected, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(run([3, 2, 6, 2], 1), expected, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(run([1] * 13, 0), expected, atol=2e-6, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gdn_cuda_against_hf(self):
        from minisgl.kernel.qwen4 import gated_delta_rule

        ref = independent_hf_reference()
        torch.manual_seed(3)
        for length in (1, 7, 65):
            q, k = [
                torch.randn(length, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
            ]
            v = torch.randn(length, 6, 128, device="cuda", dtype=torch.bfloat16)
            g = -torch.rand(length, 6, device="cuda")
            beta = torch.rand(length, 6, device="cuda", dtype=torch.bfloat16)
            initial = torch.randn(6, 128, 128, device="cuda") * 0.1
            state = initial.clone()
            actual = gated_delta_rule(l2_norm(q), l2_norm(k), v, g, beta, state)
            for algorithm in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
                expected, final = ref[algorithm](
                    q.repeat_interleave(3, 1)[None],
                    k.repeat_interleave(3, 1)[None],
                    v[None],
                    g[None],
                    beta[None],
                    initial_state=initial.transpose(-1, -2)[None],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                torch.testing.assert_close(actual, expected[0], atol=0.002, rtol=0.02)
                torch.testing.assert_close(state, final[0].transpose(-1, -2), atol=2e-5, rtol=2e-4)
            if length > 1:
                state2 = initial.clone()
                split = min(3, length - 1)
                chunks = []
                for sl in (slice(0, split), slice(split, length)):
                    chunks.append(
                        gated_delta_rule(
                            l2_norm(q[sl]), l2_norm(k[sl]), v[sl], g[sl], beta[sl], state2
                        )
                    )
                torch.testing.assert_close(torch.cat(chunks), actual, atol=0, rtol=0)
                torch.testing.assert_close(state2, state, atol=0, rtol=0)


class Qwen4PackingTest(unittest.TestCase):
    def test_projection_order_and_parameter_views(self):
        from minisgl.models.qwen4_weight import pack_qwen4_weights

        def projection(value, rows=1):
            return SimpleNamespace(weight=torch.full((rows, 4), float(value)))

        for aligned in (False, True):
            with self.subTest(aligned=aligned):
                attn = SimpleNamespace(
                    in_proj_qkv=projection(1, 3),
                    in_proj_z=projection(2, 2),
                    in_proj_a=projection(3),
                    in_proj_b=projection(4),
                )
                mlp = SimpleNamespace(
                    shared_expert=SimpleNamespace(gate_proj=projection(5), up_proj=projection(6))
                )
                model = SimpleNamespace(
                    runtime=Qwen4RuntimeConfig(aligned=aligned),
                    model=SimpleNamespace(
                        layers=SimpleNamespace(
                            op_list=[SimpleNamespace(linear_attn=attn, self_attn=None, mlp=mlp)]
                        )
                    ),
                )
                # Isolate projection layout from the SM100-specific HC permutation.
                with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                    pack_qwen4_weights(model)
                expected = [1, 1, 1, 2, 2] + ([4, 3] if aligned else [3, 4])
                torch.testing.assert_close(attn._packed[:, 0], torch.tensor(expected).float())
                torch.testing.assert_close(mlp._packed[:, 0], torch.tensor([5.0, 6.0]))
                for name in ("qkv", "z", "a", "b"):
                    weight = getattr(attn, f"in_proj_{name}").weight
                    self.assertEqual(
                        weight.untyped_storage().data_ptr(),
                        attn._packed.untyped_storage().data_ptr(),
                    )
                attn.in_proj_a.weight.fill_(9)
                self.assertEqual(attn._packed[-1 if aligned else -2, 0].item(), 9)

    def test_unresolved_policy_rejected(self):
        from minisgl.models.qwen4_weight import pack_qwen4_weights

        with self.assertRaises(ValueError):
            pack_qwen4_weights(SimpleNamespace(runtime=Qwen4RuntimeConfig()))


if __name__ == "__main__":
    unittest.main()
