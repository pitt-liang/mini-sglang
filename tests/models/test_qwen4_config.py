"""Typed configuration dispatch and construction, without external checkpoints."""

from dataclasses import FrozenInstanceError, asdict, replace
from unittest.mock import patch

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.models import ModelConfig, Qwen4Config, create_model
from qwen4_test_utils import make_qwen4_config
from transformers import LlamaConfig, PretrainedConfig, Qwen3Config, Qwen3MoeConfig


def hf_text_config(**overrides) -> PretrainedConfig:
    values = asdict(make_qwen4_config())
    for source, target in (
        ("num_layers", "num_hidden_layers"),
        ("num_qo_heads", "num_attention_heads"),
        ("num_kv_heads", "num_key_value_heads"),
    ):
        values[target] = values.pop(source)
    for key in ("rotary_config", "reference", "aligned", "ple_storage", "ple_prefetch"):
        values.pop(key)
    values.update(
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    )
    values.update(overrides)
    return PretrainedConfig.from_dict(values)


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_factory_dispatches_to_typed_config(model_type, monkeypatch):
    monkeypatch.setenv("MINISGL_QWEN4_SGLANG_NUMERICS", "0")
    monkeypatch.setenv("MINISGL_QWEN4_PLE_STORAGE", "pinned")
    monkeypatch.setenv("MINISGL_QWEN4_PLE_PREFETCH", "1")
    with patch("torch.cuda.get_device_capability", side_effect=AssertionError("CUDA queried")):
        config = ModelConfig.from_hf(hf_text_config(model_type=model_type))
    assert isinstance(config, Qwen4Config)
    assert config == Qwen4Config.from_hf(hf_text_config(model_type=model_type))
    assert (config.num_layers, config.num_kv_layers, config.rotary_config.rotary_dim) == (4, 1, 4)
    assert config.hc_count == 4 and config.linear_num_value_heads == 4
    assert config.ple_layer_ids == (2,) and config.indexer_compress_ratio == 4
    assert config.aligned is False and config.ple_storage == "pinned" and config.ple_prefetch
    assert not hasattr(config, "qwen4_runtime") and not hasattr(config, "hybrid")


@pytest.mark.parametrize("as_dict", [False, True])
def test_nested_text_config_and_top_level_architecture(as_dict):
    text = hf_text_config(architectures=None)
    top = PretrainedConfig(
        text_config=dict(vars(text)) if as_dict else text,
        architectures=["Qwen4ExpForConditionalGeneration"],
    )
    config = ModelConfig.from_hf(top)
    assert isinstance(config, Qwen4Config)
    assert config.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert config.layer_types == ("linear_attention",) * 3 + ("full_attention",)


@pytest.mark.parametrize("hf_class", [LlamaConfig, Qwen3Config, Qwen3MoeConfig])
def test_generic_models_do_not_read_or_store_qwen4_options(hf_class, monkeypatch):
    monkeypatch.setenv("MINISGL_QWEN4_PLE_STORAGE", "invalid")
    monkeypatch.setenv("MINISGL_QWEN4_SGLANG_NUMERICS", "invalid")
    config = ModelConfig.from_hf(hf_class())
    assert type(config) is ModelConfig
    assert config.num_kv_layers == config.num_layers and not config.is_qwen4
    for field in ("qwen4_runtime", "hybrid", "ple_storage", "aligned", "linear_num_key_heads"):
        assert not hasattr(config, field)
    with pytest.raises(ValueError, match="requires a Qwen4"):
        Qwen4Config.from_hf(hf_class())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"layer_types": ["full_attention"]}, "one layer_types entry"),
        ({"layer_types": ["unknown"] * 4}, "Unsupported Qwen4 layer type"),
        ({"rope_parameters": {"rope_type": "linear"}}, "default text RoPE"),
    ],
)
def test_invalid_architecture_is_rejected(overrides, message):
    # Start from valid HF metadata so this exercises mini's validation, not
    # Transformers' own constructor-time validators.
    config = hf_text_config()
    for name, value in overrides.items():
        setattr(config, name, value)
    with pytest.raises(ValueError, match=message):
        ModelConfig.from_hf(config)


def test_frozen_replace_preserves_subtype_and_architecture():
    config = make_qwen4_config()
    resolved = config.resolve_runtime(torch.device("cpu"))
    assert type(resolved) is Qwen4Config
    assert resolved.aligned is False and config.aligned is None
    changed = replace(resolved, ple_prefetch=True)
    assert changed.layer_types == config.layer_types
    assert changed.ple_prefetch and not config.ple_prefetch
    with pytest.raises(FrozenInstanceError):
        resolved.aligned = True


def test_model_backend_and_pool_consume_the_same_typed_config():
    from minisgl.attention import create_attention_backend
    from minisgl.kvcache import create_kvcache_pool
    from minisgl.kvcache.qwen4_pool import Qwen4KVCache

    config = make_qwen4_config(reference=True).resolve_runtime(torch.device("cpu"))
    with patch("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1)):
        with torch.device("meta"):
            model = create_model(config)
        backend = create_attention_backend("qwen4", config)
        pool = create_kvcache_pool(config, 2, 4, torch.float32, torch.device("cpu"), 1)
    assert model.config is backend.config is config
    for layer in model.model.layers.op_list:
        assert layer.attn_hyper_connection.config is config
        assert layer.mlp.config is config
        assert (layer.linear_attn or layer.self_attn).config is config
        if layer.ple is not None:
            assert layer.ple.config is config
    assert isinstance(pool, Qwen4KVCache)
    assert len(pool.recurrent) == 3 and len(pool.ple_history) == 1
    pool.reset(0)
    assert torch.all(pool.ple_history[1][0] == config.eos_token_id)


def test_generic_config_cannot_construct_qwen4_components():
    from minisgl.attention import create_attention_backend
    from minisgl.models.qwen4 import Qwen4ExpForCausalLM

    generic = ModelConfig.from_hf(LlamaConfig())
    with pytest.raises(TypeError, match="requires Qwen4Config"):
        Qwen4ExpForCausalLM(generic)
    with pytest.raises(TypeError, match="requires Qwen4Config"):
        create_attention_backend("qwen4", generic)
