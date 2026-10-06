"""Typed configuration dispatch and construction, without external checkpoints."""

from dataclasses import FrozenInstanceError, asdict, replace
from unittest.mock import patch

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.env import StartupEnvVar
from minisgl.models import ModelConfig, Qwen4Config, create_model
from qwen4_test_utils import make_qwen4_config, qwen4_execution
from transformers import LlamaConfig, PretrainedConfig, Qwen3Config, Qwen3MoeConfig


def hf_text_config(**overrides) -> PretrainedConfig:
    values = asdict(make_qwen4_config())
    for source, target in (
        ("num_layers", "num_hidden_layers"),
        ("num_qo_heads", "num_attention_heads"),
        ("num_kv_heads", "num_key_value_heads"),
    ):
        values[target] = values.pop(source)
    values.pop("rotary_config")
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
    # Even malformed execution env must not affect architecture parsing.
    monkeypatch.setenv("MINISGL_QWEN4_SGLANG_NUMERICS", "invalid")
    monkeypatch.setenv("MINISGL_QWEN4_PLE_STORAGE", "invalid")
    monkeypatch.setenv("MINISGL_QWEN4_PLE_PREFETCH", "invalid")
    with patch("torch.cuda.get_device_capability", side_effect=AssertionError("CUDA queried")):
        config = ModelConfig.from_hf(hf_text_config(model_type=model_type))
    assert isinstance(config, Qwen4Config)
    assert config == Qwen4Config.from_hf(hf_text_config(model_type=model_type))
    assert (config.num_layers, config.num_kv_layers, config.rotary_config.rotary_dim) == (4, 1, 4)
    assert config.hc_count == 4 and config.linear_num_value_heads == 4
    assert config.ple_layer_ids == (2,) and config.indexer_compress_ratio == 4
    for name in (
        "reference",
        "aligned",
        "ple_storage",
        "ple_prefetch",
        "qwen4_runtime",
        "hybrid",
        "with_env_overrides",
        "resolve_runtime",
        "use_optimized_kernels",
    ):
        assert not hasattr(config, name)


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
    changed = replace(config, indexer_budget=16)
    assert type(changed) is Qwen4Config
    assert changed.layer_types == config.layer_types
    assert changed.indexer_budget == 16 and config.indexer_budget == 8
    with pytest.raises(FrozenInstanceError):
        config.indexer_budget = 16


def test_model_backend_and_pool_consume_the_same_typed_config():
    from minisgl.attention import create_attention_backend
    from minisgl.kvcache import create_kvcache_pool
    from minisgl.kvcache.qwen4_pool import Qwen4KVCache

    config = make_qwen4_config()
    with (
        patch("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1)),
        qwen4_execution() as execution,
        patch.object(
            StartupEnvVar, "read", side_effect=AssertionError("env re-read after startup")
        ),
    ):
        with torch.device("meta"):
            model = create_model(config)
        backend = create_attention_backend("qwen4", config)
        pool = create_kvcache_pool(config, 2, 4, torch.float32, torch.device("cpu"), 1)
    assert model.config is backend.config is config
    assert model.execution is backend.execution is execution
    for layer in model.model.layers.op_list:
        assert layer.attn_hyper_connection.config is config
        assert layer.mlp.config is config
        assert layer.mlp.execution is execution
        assert layer.attn_hyper_connection.execution is execution
        assert (layer.linear_attn or layer.self_attn).config is config
        assert (layer.linear_attn or layer.self_attn).execution is execution
        if layer.ple is not None:
            assert layer.ple.config is config
            assert layer.ple.execution is execution
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


@pytest.mark.parametrize("aligned", [False, True])
def test_engine_graph_defaults_use_execution_without_replacing_model_config(aligned):
    from minisgl.engine import EngineConfig
    from minisgl.engine.engine import _adjust_config

    model_config = make_qwen4_config()
    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16)
    object.__setattr__(config, "model_config", model_config)
    with (
        qwen4_execution(aligned=aligned) as execution,
        patch("minisgl.distributed.info._TP_INFO", config.tp_info),
    ):
        _adjust_config(config, execution)
    assert config.model_config is model_config
    assert config.cuda_graph_max_bs == 4
    assert config.attention_backend == "qwen4"


def test_engine_can_disable_graph_capture():
    from minisgl.engine import EngineConfig
    from minisgl.engine.engine import _adjust_config

    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, cuda_graph_max_bs=0)
    object.__setattr__(config, "model_config", make_qwen4_config())
    with qwen4_execution() as execution:
        _adjust_config(config, execution)
    assert config.cuda_graph_max_bs == 0
