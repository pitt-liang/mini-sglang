"""Switch for same-weight reference comparisons; reference never uses graphs."""

import os

ENABLED = os.environ.get("MINISGL_QWEN4_REFERENCE", "0") != "1"
# Resolved after weights are on the target GPU, before graph capture. Do not
# initialize CUDA at import time in the scheduler's parent process.
SGLANG_NUMERICS = os.environ.get("MINISGL_QWEN4_SGLANG_NUMERICS", "0") == "1"


def configure_numerics(device):
    import torch

    global SGLANG_NUMERICS
    requested = os.environ.get("MINISGL_QWEN4_SGLANG_NUMERICS")
    if requested not in (None, "0", "1"):
        raise ValueError("MINISGL_QWEN4_SGLANG_NUMERICS must be 0 or 1")
    supported = (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] == 10
        and torch.version.cuda is not None
        and int(torch.version.cuda.split(".")[0]) >= 13
    )
    if requested == "1" and not supported:
        raise ValueError("Qwen4 SGLang-aligned kernels currently require SM100 and CUDA 13+")
    SGLANG_NUMERICS = requested == "1" or (requested is None and ENABLED and supported)


def enabled(x=None):
    return ENABLED and (x is None or x.is_cuda)
