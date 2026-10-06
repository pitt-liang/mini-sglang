from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Callable, Generic, Literal, NamedTuple, TypeVar

if TYPE_CHECKING:
    import torch


class BaseEnv:
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class StartupEnvVar(BaseEnv, Generic[T]):
    """Strict opt-in read at worker startup, not module import or forward time.

    Unrelated models need not validate Qwen4 flags. Unlike the legacy EnvVar,
    invalid values are never silently replaced with defaults.
    """

    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.default_value = default_value
        self.fn = fn

    def _init(self, name: str) -> None:
        self.name = name

    def read(self) -> T:
        value = os.getenv(self.name)
        if value is None:
            return self.default_value
        try:
            return self.fn(value)
        except ValueError as exc:
            raise ValueError(f"{self.name}: {exc}") from exc


class EnvVar(BaseEnv, Generic[T]):
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


MINISGL_ENV_PREFIX = "MINISGL_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)


def _strict_bool(value: str) -> bool:
    if value not in ("0", "1"):
        raise ValueError("must be 0 or 1")
    return value == "1"


def _ple_storage(value: str) -> Literal["gpu", "pinned"]:
    if value == "gpu":
        return "gpu"
    if value == "pinned":
        return "pinned"
    raise ValueError("must be gpu or pinned")


class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None

    # shell
    SHELL_MAX_TOKENS = EnvInt(2048)
    SHELL_TOP_K = EnvInt(-1)
    SHELL_TOP_P = EnvFloat(1.0)
    SHELL_TEMPERATURE = EnvFloat(0.6)

    # backend runtime
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)

    # Qwen4 startup flags: read together after the worker selects its device.
    QWEN4_SGLANG_NUMERICS = StartupEnvVar[bool | None](None, _strict_bool)
    QWEN4_PLE_STORAGE = StartupEnvVar[Literal["gpu", "pinned"]]("gpu", _ple_storage)
    QWEN4_PLE_PREFETCH = StartupEnvVar(False, _strict_bool)

    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()


class Qwen4Execution(NamedTuple):
    """Internal resolved worker state, not model metadata or a user config API.

    The engine, weight loader and layers share this immutable snapshot. Never
    re-read env during execution: packing and captured graphs depend on it.
    """

    aligned: bool
    ple_storage: Literal["gpu", "pinned"]
    ple_prefetch: bool


def resolve_qwen4_execution(device: torch.device) -> Qwen4Execution:
    """Called once per Qwen4 worker, before construction, packing and capture."""
    import torch

    if "MINISGL_QWEN4_REFERENCE" in os.environ:
        raise ValueError(
            "MINISGL_QWEN4_REFERENCE has been removed; unset it and use external "
            "Transformers/SGLang tests for reference validation"
        )
    if device.type != "cuda":
        raise ValueError("Qwen4 execution requires a CUDA device")
    requested = ENV.QWEN4_SGLANG_NUMERICS.read()
    storage = ENV.QWEN4_PLE_STORAGE.read()
    prefetch = ENV.QWEN4_PLE_PREFETCH.read()
    supported = (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] == 10
        and torch.version.cuda is not None
        and int(torch.version.cuda.split(".")[0]) >= 13
    )
    if requested is True and not supported:
        raise ValueError("Qwen4 aligned kernels currently require SM100 and CUDA 13+")
    aligned = supported if requested is None else requested
    return Qwen4Execution(aligned, storage, prefetch)
