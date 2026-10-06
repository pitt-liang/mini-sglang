"""Qwen4 env lifecycle: import-safe, startup-only, strict and worker-local."""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from minisgl.env import ENV, resolve_qwen4_execution


def test_import_does_not_parse_qwen4_flags_or_import_torch():
    environment = dict(os.environ)
    environment["MINISGL_QWEN4_SGLANG_NUMERICS"] = "invalid"
    result = subprocess.run(
        [sys.executable, "-c", "import minisgl.env, sys; assert 'torch' not in sys.modules"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_worker_reads_current_env_once_and_keeps_a_snapshot(monkeypatch):
    for name in ("REFERENCE", "SGLANG_NUMERICS", "PLE_STORAGE", "PLE_PREFETCH"):
        monkeypatch.delenv("MINISGL_QWEN4_" + name, raising=False)
    # ENV has already been imported: late startup env must still take effect.
    monkeypatch.setenv("MINISGL_QWEN4_SGLANG_NUMERICS", "1")
    with (
        patch.object(
            ENV.QWEN4_SGLANG_NUMERICS, "read", wraps=ENV.QWEN4_SGLANG_NUMERICS.read
        ) as read,
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        patch("torch.version.cuda", "13.0"),
    ):
        first = resolve_qwen4_execution(torch.device("cuda"))
        read.assert_called_once_with()
        monkeypatch.setenv("MINISGL_QWEN4_SGLANG_NUMERICS", "0")
        second = resolve_qwen4_execution(torch.device("cuda"))
    assert first.aligned and not second.aligned
    with pytest.raises(AttributeError):
        first.aligned = False


@pytest.mark.parametrize("value", ["0", "1", "invalid"])
def test_removed_reference_flag_requires_migration(value, monkeypatch):
    monkeypatch.setenv("MINISGL_QWEN4_REFERENCE", value)
    with pytest.raises(ValueError, match="has been removed; unset it"):
        resolve_qwen4_execution(torch.device("cuda"))


@pytest.mark.parametrize("capability,cuda", [((9, 0), "13.0"), ((10, 0), "12.9"), ((10, 0), None)])
def test_auto_fallback_and_explicit_unsupported_device(capability, cuda, monkeypatch):
    for name in ("REFERENCE", "SGLANG_NUMERICS", "PLE_STORAGE", "PLE_PREFETCH"):
        monkeypatch.delenv("MINISGL_QWEN4_" + name, raising=False)
    with (
        patch("torch.cuda.get_device_capability", return_value=capability),
        patch("torch.version.cuda", cuda),
    ):
        execution = resolve_qwen4_execution(torch.device("cuda:0"))
        assert not execution.aligned
        monkeypatch.setenv("MINISGL_QWEN4_SGLANG_NUMERICS", "1")
        with pytest.raises(ValueError, match="SM100 and CUDA 13"):
            resolve_qwen4_execution(torch.device("cuda:0"))
