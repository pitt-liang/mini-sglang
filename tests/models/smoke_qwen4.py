"""Real-weight TP smoke test; run with torchrun, not pytest.

Checks eager/graph parity (including padding), slot recycling and the actual
overlap scheduler. Optional captures compare against a separately run baseline.
Uses a controlled top-k policy, so this is not a stock sparse-top-k or performance
benchmark. The reference SGLang framework is never imported.
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.kernel import qwen4_ops
from qwen4_test_utils import index_topk_deterministic


def forward(engine, inputs, cached, graph):
    bs, count = inputs.shape
    reqs = [
        Req(
            torch.empty(cached + count, dtype=torch.int32), i, cached, 64, i, SamplingParams(), None
        )
        for i in range(bs)
    ]
    batch = Batch(reqs, "prefill" if cached == 0 else "decode")
    # Pad eager decode identically to graph decode: B3 must exercise B4 masking.
    engine.graph_runner.pad_batch(batch)
    device = engine.device
    batch.input_ids = inputs.to(device=device, dtype=torch.int32).flatten()
    batch.positions = torch.arange(cached, cached + count, device=device, dtype=torch.int32).repeat(
        bs
    )
    batch.out_loc = engine.page_table[:bs, cached : cached + count].flatten().contiguous()
    if batch.padded_size > bs:
        padding = batch.padded_size - bs
        zeros = torch.zeros(padding, device=device, dtype=torch.int32)
        batch.input_ids = torch.cat((batch.input_ids, zeros))
        batch.positions = torch.cat((batch.positions, zeros))
        batch.out_loc = torch.cat(
            (batch.out_loc, engine.page_table[engine.dummy_req.table_idx, :padding])
        )
    engine.attn_backend.prepare_metadata(batch)
    with engine.ctx.forward_batch(batch):
        logits = (
            engine.graph_runner.replay(batch)
            if graph and engine.graph_runner.can_use_cuda_graph(batch)
            else engine.model.forward()
        )[:bs]
    assert torch.isfinite(logits).all(), "Non-finite logits"
    # GraphRunner exposes an FP32 output buffer; eager returns BF16. Widening
    # BF16 to FP32 is exact and leaves the zero-tolerance value check unchanged.
    return logits.detach().float().cpu().clone()


def workload(engine, bs, length, steps, graph):
    ids = ([760, 6511, 314, 9338, 369] * ((length + 4) // 5))[:length]
    logits = [forward(engine, torch.tensor([ids] * bs), 0, graph)]
    for step in range(steps):
        logits.append(forward(engine, torch.full((bs, 1), 11751), length + step, graph))
    return logits


def scheduler_smoke(engine, config):
    from minisgl.message import AbortBackendMsg, UserMsg
    from minisgl.scheduler import Scheduler, SchedulerConfig

    class Finished(Exception):
        pass

    results, done = {}, set()

    class TestScheduler(Scheduler):
        stage = 0

        def offline_receive_msg(self, blocking=False):
            def request(uid):
                return UserMsg(
                    uid=uid,
                    input_ids=torch.tensor([760, 6511, 314, 9338, 369], dtype=torch.int32),
                    sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
                )

            if self.stage == 0:
                self.stage = 1
                return [request(0), request(1)]
            if self.stage == 1 and {0, 1} <= done:
                self.stage = 2
                return [request(2)]
            if self.stage == 2 and 2 in done:
                self.stage = 3
                return [request(3)]
            if self.stage == 3:
                self.stage = 4
                return [AbortBackendMsg(uid=3), request(4)]
            if blocking and 4 in done:
                raise Finished()
            return []

        def offline_send_result(self, reply):
            for msg in reply:
                if msg.uid not in done:
                    results.setdefault(msg.uid, []).append(msg.next_token)
                    if msg.finished:
                        done.add(msg.uid)

    values = asdict(config)
    values["tp_info"] = config.tp_info
    cfg = SchedulerConfig(**values, offline_mode=True, cache_type="naive", max_extend_tokens=3)
    with patch("minisgl.engine.Engine", return_value=engine):
        scheduler = TestScheduler(cfg)
    try:
        scheduler.run_forever()
    except Finished:
        pass
    scheduler.cache_manager.check_integrity()
    assert set(results) == {0, 1, 2, 4}, results
    assert results[0] == results[1] == results[2] == results[4], results
    assert len(results[0]) == 3, results
    assert scheduler.table_manager.available_size == config.max_running_req
    return results


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Per-rank capture prefix")
    parser.add_argument("--compare", type=Path, help="Previously captured baseline prefix")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 64:
        parser.error("--steps must be in [1, 64]")
    if args.compare is not None and args.compare.resolve() == args.output.resolve():
        parser.error("--compare and --output must differ")
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if size != 2:
        parser.error("This smoke configuration requires TP=2")
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    config = EngineConfig(
        str(Path(args.model).resolve()),
        DistributedInfo(rank, size),
        torch.bfloat16,
        max_running_req=4,
        max_seq_len_override=8192,
        page_size=64,
        num_page_override=512,
        cuda_graph_max_bs=4,
        use_pynccl=False,
        distributed_timeout=600,
    )
    with patch.object(qwen4_ops, "index_topk", index_topk_deterministic):
        engine = Engine(config)
        try:
            for slot in range(4):
                engine.page_table[slot, :8192] = torch.arange(
                    slot * 8192, (slot + 1) * 8192, device=engine.device
                )
            report = {
                "model": config.model_path,
                "torch": str(torch.__version__),
                "python": __import__("sys").executable,
                "controlled_topk": True,
                "steps": args.steps,
                "rank": rank,
                "cases": [],
            }
            for bs, length in ((1, 5), (1, 128), (1, 2053), (3, 7)):
                captured = workload(engine, bs, length, args.steps, graph=True)
                eager = workload(engine, bs, length, args.steps, graph=False)
                for a, b in zip(captured, eager):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                report["cases"].append({"batch": bs, "length": length, "logits": captured})
                print(f"rank {rank}: B{bs}/L{length} finite + eager/graph exact PASS", flush=True)
            if args.compare:
                old = torch.load(f"{args.compare}.rank{rank}.pt", weights_only=True)
                for key in ("model", "controlled_topk", "steps", "rank"):
                    assert old[key] == report[key], (key, old[key], report[key])
                assert len(old["cases"]) == len(report["cases"])
                for old_case, new_case in zip(old["cases"], report["cases"]):
                    assert (old_case["batch"], old_case["length"]) == (
                        new_case["batch"],
                        new_case["length"],
                    )
                    assert len(old_case["logits"]) == len(new_case["logits"])
                    for a, b in zip(old_case["logits"], new_case["logits"]):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                report["baseline_exact"] = True
                print(f"rank {rank}: separate baseline exact PASS", flush=True)
            report["scheduler"] = scheduler_smoke(engine, config)
            if args.compare:
                assert report["scheduler"] == old["scheduler"], (
                    report["scheduler"],
                    old["scheduler"],
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(report, f"{args.output}.rank{rank}.pt")
            summary = {k: v for k, v in report.items() if k != "cases"}
            summary["cases"] = [
                {k: v for k, v in c.items() if k != "logits"} for c in report["cases"]
            ]
            Path(f"{args.output}.rank{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"rank {rank}: scheduler cancellation/reuse PASS", flush=True)
        finally:
            engine.shutdown()


if __name__ == "__main__":
    main()
