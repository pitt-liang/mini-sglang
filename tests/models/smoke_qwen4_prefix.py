"""Real-weight hybrid prefix accuracy and warm/cold TTFT regression.

Run with torchrun (TP2). Uses the real scheduler, including chunked prefill,
overlap scheduling, CUDA graphs and B3 padding. Performance runs never capture
logits. --controlled-topk is for numerical diagnosis, not stock performance.
"""

import argparse
import json
import os
import time
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine
from minisgl.kernel import qwen4_ops
from minisgl.message import UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.scheduler.prefill import ChunkedReq
from qwen4_test_utils import index_topk_deterministic


def exercise(
    engine,
    config,
    groups,
    cache_type,
    capture=False,
    trace_components=False,
    trace_layers=(0, 1, 2, 3),
    output_tokens=4,
):
    torch.cuda.set_stream(engine.stream)
    torch.cuda.synchronize()
    # The previous scheduler is drained; discard its tree before starting an
    # independent baseline. Model weights and graph captures remain unchanged.
    pool = engine.kv_cache.prefix_states
    pool.free_slots = list(range(pool.capacity))
    pending = list(groups)
    results, done, active, starts, ttft, hits, logits = {}, set(), set(), {}, {}, {}, {}
    arrivals = {}
    current = {}
    layers = {}
    components = {}
    calls = 0

    class Finished(Exception):
        pass

    class TestScheduler(Scheduler):
        def offline_receive_msg(self, blocking=False):
            nonlocal calls
            calls += 1
            if calls > 10000:
                raise RuntimeError("Scheduler did not drain")
            if active <= done and pending:
                group = pending.pop(0)
                active.clear()
                active.update(uid for uid, _ in group)
                messages = []
                for uid, ids in group:
                    starts[uid] = time.perf_counter()
                    messages.append(
                        UserMsg(
                            uid,
                            torch.tensor(ids, dtype=torch.int32),
                            SamplingParams(max_tokens=output_tokens, ignore_eos=True),
                        )
                    )
                return messages
            if blocking and not pending and active <= done:
                raise Finished()
            return []

        def offline_send_result(self, reply):
            for msg in reply:
                if msg.uid in done:
                    continue
                ttft.setdefault(msg.uid, (time.perf_counter() - starts[msg.uid]) * 1000)
                arrivals.setdefault(msg.uid, []).append(time.perf_counter())
                results.setdefault(msg.uid, []).append(msg.next_token)
                if msg.finished:
                    done.add(msg.uid)

        def _forward(self, forward_input):
            current["batch"] = forward_input.batch
            if forward_input.batch.is_prefill:
                for req in forward_input.batch.reqs:
                    hits.setdefault(req.uid, req.cached_len)
            return super()._forward(forward_input)

    original_sample = engine.sampler.sample

    def sample(values, args):
        if capture:
            for row, req in enumerate(current["batch"].reqs):
                if not isinstance(req, ChunkedReq):
                    logits.setdefault(req.uid, []).append(values[row].detach().float().cpu())
        return original_sample(values, args)

    def capture_rows(values, destination, name):
        batch = current["batch"]
        if not batch.is_prefill:
            return
        if isinstance(values, tuple):
            for index, value in enumerate(values):
                capture_rows(value, destination, f"{name}.{index}")
            return
        if not isinstance(values, torch.Tensor):
            return
        for row, req in enumerate(batch.reqs):
            if not isinstance(req, ChunkedReq):
                start, end = batch.attn_metadata.spans[row][-2:]
                destination.setdefault(req.uid, {})[name] = (
                    values[max(start, end - 16) : end].detach().float().cpu()
                )

    cfg = replace(config, cache_type=cache_type)
    with patch("minisgl.engine.Engine", return_value=engine):
        scheduler = TestScheduler(cfg)
    begin = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(engine.sampler, "sample", new=sample))
        if capture:
            if trace_components:
                original_gdn_output = qwen4_ops.gdn_output

                def traced_gdn_output(x, z, weight, eps=1e-6, **kwargs):
                    layer_id = current.get("layer")
                    result = original_gdn_output(x, z, weight, eps, **kwargs)
                    if layer_id in trace_layers:
                        capture_rows(x, components, f"{layer_id}.gdn_norm.input")
                        capture_rows(z, components, f"{layer_id}.gdn_norm.z")
                        capture_rows(result, components, f"{layer_id}.gdn_norm.output")
                    return result

                stack.enter_context(patch.object(qwen4_ops, "gdn_output", new=traced_gdn_output))
            for layer_id, layer in enumerate(engine.model.model.layers.op_list):
                original = layer.forward

                def traced(x, original=original, layer_id=layer_id):
                    current["layer"] = layer_id
                    result = original(x)
                    capture_rows(result, layers, layer_id)
                    return result

                # Plain wrappers do not retain full GPU inputs in Mock.call_args_list.
                stack.enter_context(patch.object(layer, "forward", new=traced))
                if trace_components and layer_id in trace_layers:
                    modules = {
                        "attn_hc": layer.attn_hyper_connection,
                        "attn": layer.linear_attn or layer.self_attn,
                        "mlp_hc": layer.mlp_hyper_connection,
                        "mlp": layer.mlp,
                    }
                    if layer.ple is not None:
                        modules.update(
                            ple=layer.ple,
                            ple_key=layer.ple.key_proj,
                            ple_value=layer.ple.value_proj,
                        )
                    for name, module in modules.items():
                        method = module.forward
                        label = f"{layer_id}.{name}"

                        def component(x, method=method, label=label):
                            capture_rows(x, components, f"{label}.input")
                            result = method(x)
                            capture_rows(result, components, f"{label}.output")
                            return result

                        stack.enter_context(patch.object(module, "forward", new=component))
        try:
            scheduler.run_forever()
        except Finished:
            pass
    torch.cuda.synchronize()
    scheduler.cache_manager.check_integrity()
    assert scheduler.table_manager.available_size == config.max_running_req
    assert len(done) == sum(len(group) for group in groups)
    stats = getattr(scheduler.cache_manager.prefix_cache, "stats", {})
    return {
        "tokens": results,
        "ttft_ms": ttft,
        "tpot_ms": {
            uid: (times[-1] - times[0]) * 1000 / (len(times) - 1)
            for uid, times in arrivals.items()
            if len(times) > 1
        },
        "hits": hits,
        "logits": logits,
        "layers": layers,
        "components": components,
        "cache_stats": stats,
        "elapsed_ms": (time.perf_counter() - begin) * 1000,
    }


def compare(reference, actual):
    metrics = []
    for uid, baseline in reference["logits"].items():
        candidate = actual["logits"][uid]
        assert len(baseline) == len(candidate)
        for step, (a, b) in enumerate(zip(baseline, candidate)):
            a, b = a.double(), b.double()
            assert torch.isfinite(b).all()
            cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            rmse = (a - b).square().mean().sqrt().item()
            metrics.append(
                {
                    "uid": uid,
                    "step": step,
                    "cosine": cosine,
                    "rmse": rmse,
                    "max_abs": (a - b).abs().max().item(),
                    "top1_equal": a.argmax().item() == b.argmax().item(),
                    "kl": (a.softmax(0) * (a.log_softmax(0) - b.log_softmax(0))).sum().item(),
                }
            )
    return {
        "tokens_equal": reference["tokens"] == actual["tokens"],
        "min_cosine": min(m["cosine"] for m in metrics),
        "max_rmse": max(m["rmse"] for m in metrics),
        "max_abs": max(m["max_abs"] for m in metrics),
        "top1_equal": sum(m["top1_equal"] for m in metrics),
        "rows": len(metrics),
        "details": metrics,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=2053)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--max-extend-tokens", type=int, default=8192)
    parser.add_argument("--prompt-pattern", choices=("repeat", "text"), default="repeat")
    parser.add_argument(
        "--branch-at", type=int, help="Replay and cache a previously unsaved junction"
    )
    parser.add_argument("--controlled-topk", action="store_true")
    parser.add_argument("--trace-components", action="store_true")
    parser.add_argument("--trace-layers", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert size == 2
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    config = SchedulerConfig(
        args.model,
        DistributedInfo(rank, size),
        torch.bfloat16,
        max_running_req=4,
        max_seq_len_override=16384,
        page_size=64,
        num_page_override=1024,
        cuda_graph_max_bs=4,
        use_pynccl=False,
        distributed_timeout=600,
        offline_mode=True,
        cache_type="hybrid",
        qwen4_state_cache_mb=512,
        max_extend_tokens=args.max_extend_tokens,
    )
    selection = (
        patch.object(qwen4_ops, "index_topk", index_topk_deterministic)
        if args.controlled_topk
        else nullcontext()
    )
    with selection:
        engine = Engine(config)
        try:
            ids = ([760, 6511, 314, 9338, 369] * ((args.length + 4) // 5))[: args.length]
            if args.prompt_pattern == "text":
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(args.model)
                paragraphs = [
                    f"Record {i}: A prefix cache lets requests reuse computation for shared "
                    "documents. Recurrent layers must restore their state at the same token "
                    "boundary as attention and convolution layers. Explain the tradeoff "
                    "between checkpoint memory, latency, and numerical accuracy.\n"
                    for i in range(args.length // 20 + 1)
                ]
                ids = tokenizer.encode("".join(paragraphs), add_special_tokens=False)[: args.length]
                assert len(ids) == args.length
            groups = [[(0, ids)], [(1, ids)], [(2, ids + [100]), (3, ids + [200]), (4, ids)]]
            if args.branch_at is not None:
                assert 64 <= args.branch_at < args.length - 1
                groups += [
                    [(5, ids[: args.branch_at] + [101] * 257)],
                    [(6, ids[: args.branch_at] + [201] * 257)],
                ]
            cold = exercise(
                engine,
                config,
                groups,
                "naive",
                capture=True,
                trace_components=args.trace_components,
                trace_layers=args.trace_layers,
                output_tokens=args.output_tokens,
            )
            cached = exercise(
                engine,
                config,
                groups,
                "hybrid",
                capture=True,
                trace_components=args.trace_components,
                trace_layers=args.trace_layers,
                output_tokens=args.output_tokens,
            )
            accuracy = compare(cold, cached)
            baseline_repeat = compare(
                {"logits": {1: cold["logits"][0]}, "tokens": {1: cold["tokens"][0]}},
                {"logits": {1: cold["logits"][1]}, "tokens": {1: cold["tokens"][1]}},
            )
            restored = compare(
                {"logits": {1: cached["logits"][0]}, "tokens": {1: cached["tokens"][0]}},
                {"logits": {1: cached["logits"][1]}, "tokens": {1: cached["tokens"][1]}},
            )
            print(
                f"rank {rank} accuracy: {accuracy}, restore: {restored}, hits={cached['hits']}",
                flush=True,
            )
            assert cached["hits"][1] == (args.length - 1) // 64 * 64
            assert all(cached["hits"][uid] > 0 for uid in (2, 3, 4))
            if args.branch_at is not None:
                assert cached["hits"][6] == args.branch_at // 64 * 64
            # Kernel repartitioning can alter BF16 rounding, but a correct
            # restore must remain close and preserve these greedy generations.
            performance_groups = [[(i, ids)] for i in range(args.repeats + 1)]
            baseline_perf = exercise(
                engine, config, performance_groups, "naive", output_tokens=args.output_tokens
            )
            cached_perf = exercise(
                engine, config, performance_groups, "hybrid", output_tokens=args.output_tokens
            )
            report = {
                "rank": rank,
                "length": args.length,
                "prompt_pattern": args.prompt_pattern,
                "branch_at": args.branch_at,
                "max_extend_tokens": args.max_extend_tokens,
                "output_tokens": args.output_tokens,
                "runtime": {
                    "tp_size": size,
                    "stable_numerics": config.qwen4_stable_numerics,
                    "ple_storage": config.model_config.qwen4_runtime.ple_storage,
                    "ple_prefetch": config.model_config.qwen4_runtime.ple_prefetch,
                    "page_size": config.page_size,
                    "checkpoint_interval": config.qwen4_checkpoint_interval,
                    "state_cache_mb": config.qwen4_state_cache_mb,
                },
                "controlled_topk": args.controlled_topk,
                "accuracy": accuracy,
                "baseline_repeat_accuracy": baseline_repeat,
                "restore_accuracy": restored,
                "hits": cached["hits"],
                "baseline": {
                    k: v
                    for k, v in baseline_perf.items()
                    if k not in ("logits", "layers", "components")
                },
                "hybrid": {
                    k: v
                    for k, v in cached_perf.items()
                    if k not in ("logits", "layers", "components")
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"naive": cold, "hybrid": cached}, f"{args.output}.rank{rank}.pt")
            Path(f"{args.output}.rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"rank {rank} TTFT naive={baseline_perf['ttft_ms']}, hybrid={cached_perf['ttft_ms']}",
                flush=True,
            )
            assert accuracy["min_cosine"] > 0.999, accuracy
            assert accuracy["tokens_equal"], accuracy
        finally:
            engine.shutdown()


if __name__ == "__main__":
    main()
