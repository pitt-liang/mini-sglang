"""Run with torchrun --standalone --nproc-per-node=2 on NVLink multicast GPUs."""

import os

import torch
import torch.distributed as dist
from minisgl.distributed import DistributedInfo
from minisgl.distributed.flashinfer import FlashInferAllReduce
from minisgl.distributed.impl import TorchDistributedImpl


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    comm = FlashInferAllReduce(
        DistributedInfo(rank, dist.get_world_size()), cpu, TorchDistributedImpl(), 5120
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.full((4, 5120), rank + 1, device="cuda", dtype=torch.bfloat16)
        comm.all_reduce(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = comm.all_reduce(x)
        for step in range(5):
            x.fill_(rank + 1 + step)
            graph.replay()
            torch.testing.assert_close(y, torch.full_like(y, 3 + 2 * step), atol=0, rtol=0)
    torch.cuda.synchronize()
    del graph
    comm.destroy()
    dist.destroy_process_group()
    print(f"rank {rank}: multicast eager/graph replay PASS", flush=True)


if __name__ == "__main__":
    main()
