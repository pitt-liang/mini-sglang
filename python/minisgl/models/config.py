from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    hybrid: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_qwen4(self) -> bool:
        return self.model_type in ("qwen4_exp", "qwen4_exp_text")

    @property
    def num_kv_layers(self) -> int:
        return (
            self.hybrid["layer_types"].count("full_attention") if self.is_qwen4 else self.num_layers
        )

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            if isinstance(config, dict):
                config = PretrainedConfig.from_dict(config)
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = config.to_dict().get("model_type", "llama")
        # Generic PretrainedConfig.to_dict uses its class model_type (empty).
        model_type = getattr(config, "model_type", model_type)
        is_qwen4 = model_type in ("qwen4_exp", "qwen4_exp_text")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", is_qwen4)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_parameters = getattr(config, "rope_parameters", None) or rope_scaling or {}
        rope_theta = getattr(config, "rope_theta", None) or rope_parameters.get(
            "rope_theta", 10000.0
        )
        rotary_dim = int(
            head_dim
            * rope_parameters.get(
                "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
            )
        )
        hybrid = dict(vars(config)) if is_qwen4 else {}
        if is_qwen4:
            if len(hybrid.get("layer_types", [])) != config.num_hidden_layers:
                raise ValueError("Qwen4 requires one layer_types entry per decoder layer")
            if set(hybrid["layer_types"]) - {"linear_attention", "full_attention"}:
                raise ValueError("Unsupported Qwen4 layer type")
            if rope_parameters.get("rope_type", "default") != "default":
                raise ValueError("Qwen4 currently supports default text RoPE only")

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=getattr(config, "intermediate_size", moe_intermediate_size),
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            hybrid=hybrid,
        )
