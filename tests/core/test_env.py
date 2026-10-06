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
    monkeypatch.setenv("MINISGL_QWEN4_PLE_PREFETCH", "1")
    with (
        patch.object(ENV.QWEN4_PLE_PREFETCH, "read", wraps=ENV.QWEN4_PLE_PREFETCH.read) as read,
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        patch("torch.version.cuda", "13.0"),
    ):
        first = resolve_qwen4_execution(torch.device("cuda"))
        read.assert_called_once_with()
        monkeypatch.setenv("MINISGL_QWEN4_PLE_PREFETCH", "0")
        second = resolve_qwen4_execution(torch.device("cuda"))
    assert first.ple_prefetch and not second.ple_prefetch
    assert not hasattr(first, "aligned")
    with pytest.raises(AttributeError):
        first.ple_prefetch = False


@pytest.mark.parametrize("value", ["0", "1", "invalid"])
@pytest.mark.parametrize("flag", ["REFERENCE", "SGLANG_NUMERICS"])
def test_removed_execution_flags_require_migration(flag, value, monkeypatch):
    monkeypatch.setenv("MINISGL_QWEN4_" + flag, value)
    with pytest.raises(ValueError, match="has been removed; unset it"):
        resolve_qwen4_execution(torch.device("cuda"))


@pytest.mark.parametrize("capability,cuda", [((9, 0), "13.0"), ((10, 0), "12.9"), ((10, 0), None)])
def test_unsupported_device_fails_without_legacy_fallback(capability, cuda, monkeypatch):
    for name in ("REFERENCE", "SGLANG_NUMERICS", "PLE_STORAGE", "PLE_PREFETCH"):
        monkeypatch.delenv("MINISGL_QWEN4_" + name, raising=False)
    with (
        patch("torch.cuda.get_device_capability", return_value=capability),
        patch("torch.version.cuda", cuda),
    ):
        with pytest.raises(ValueError, match="SM100 and CUDA 13"):
            resolve_qwen4_execution(torch.device("cuda:0"))
