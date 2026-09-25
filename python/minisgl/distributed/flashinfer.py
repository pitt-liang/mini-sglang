"""Optional NVLink multicast all-reduce; large/unhandled tensors retain NCCL.

This uses FlashInfer's public API, not the SGLang runtime. Enable explicitly with
MINISGL_QWEN4_MNNVL=1 on a multicast-capable deployment before graph capture.
"""

import torch

from .impl import DistributedCommunicator, DistributedImpl


class FlashInferAllReduce(DistributedImpl):
    def __init__(self, tp_info, cpu_group, fallback, hidden_size, max_tokens=128):
        from flashinfer.comm import create_allreduce_fusion_workspace
        from flashinfer.comm.comm_backend import TorchDistBackend

        self.fallback = fallback
        self.max_elements = max_tokens * hidden_size
        self.workspace = create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=tp_info.size,
            rank=tp_info.rank,
            max_token_num=max_tokens,
            hidden_dim=hidden_size,
            dtype=torch.bfloat16,
            comm_backend=TorchDistBackend(cpu_group),
            force_oneshot_support=True,
        )

    def all_reduce(self, x):
        if (
            x.dtype != torch.bfloat16
            or x.ndim != 2
            or not x.is_contiguous()
            or x.numel() > self.max_elements
            or x.shape[1] % 16
        ):
            return self.fallback.all_reduce(x)
        from flashinfer.comm import allreduce_fusion

        out = torch.empty_like(x)
        return allreduce_fusion(
            x,
            self.workspace,
            pattern=0,
            output=out,
            use_oneshot=True,
        )

    def all_gather(self, x):
        return self.fallback.all_gather(x)

    def destroy(self):
        self.workspace.destroy()


def enable_flashinfer_allreduce(tp_info, cpu_group, hidden_size):
    if tp_info.size > 1:
        DistributedCommunicator.plugins.append(
            FlashInferAllReduce(
                tp_info,
                cpu_group,
                DistributedCommunicator.plugins[-1],
                hidden_size,
            )
        )
