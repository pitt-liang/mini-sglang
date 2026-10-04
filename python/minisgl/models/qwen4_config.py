"""Typed Qwen4 text architecture, independent of the inference environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .config import ModelConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass(frozen=True, kw_only=True)
class Qwen4Config(ModelConfig):
    # Architecture: no opaque copy of the Hugging Face config is retained.
    layer_types: tuple[Literal["linear_attention", "full_attention"], ...]
    hc_count: int
    hc_lowrank: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    output_gate_type: str
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_compress_ratio: int
    indexer_budget: int
    shared_expert_intermediate_size: int
    ple_layer_ids: tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    heads_per_ngram: int
    ngram_size: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    eos_token_id: int

    def __post_init__(self) -> None:
        if self.model_type not in ("qwen4_exp", "qwen4_exp_text"):
            raise ValueError("Qwen4Config requires a Qwen4 text model_type")
        if len(self.layer_types) != self.num_layers:
            raise ValueError("Qwen4 requires one layer_types entry per decoder layer")
        if set(self.layer_types) - {"linear_attention", "full_attention"}:
            raise ValueError("Unsupported Qwen4 layer type")

    @property
    def is_qwen4(self) -> bool:
        return True

    @property
    def num_kv_layers(self) -> int:
        return self.layer_types.count("full_attention")

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> Qwen4Config:
        parsed = ModelConfig.from_hf(config)
        if not isinstance(parsed, cls):
            raise ValueError("Qwen4Config.from_hf requires a Qwen4 configuration")
        return parsed

    @classmethod
    def _from_hf(
        cls, config: PretrainedConfig, common: dict[str, Any], rope_parameters: dict[str, Any]
    ) -> Qwen4Config:
        if rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("Qwen4 currently supports default text RoPE only")
        # Keep required architecture fields explicit: missing checkpoint metadata
        # must fail here, rather than select a silently different architecture.
        return cls(
            **common,
            layer_types=tuple(config.layer_types),
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            linear_num_key_heads=config.linear_num_key_heads,
            linear_num_value_heads=config.linear_num_value_heads,
            linear_key_head_dim=config.linear_key_head_dim,
            linear_value_head_dim=config.linear_value_head_dim,
            linear_conv_kernel_dim=config.linear_conv_kernel_dim,
            output_gate_type=config.output_gate_type,
            indexer_n_heads=config.indexer_n_heads,
            indexer_kv_heads=config.indexer_kv_heads,
            indexer_head_dim=config.indexer_head_dim,
            indexer_compress_ratio=config.indexer_compress_ratio,
            indexer_budget=config.indexer_budget,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
            ple_layer_ids=tuple(config.ple_layer_ids),
            ple_embed_dim=config.ple_embed_dim,
            ple_conv_kernel_size=config.ple_conv_kernel_size,
            heads_per_ngram=config.heads_per_ngram,
            ngram_size=config.ngram_size,
            ngram_vocab_size_base=config.ngram_vocab_size_base,
            make_ngram_vocab_size_divisible_by=config.make_ngram_vocab_size_divisible_by,
            split_ngram_parts=config.split_ngram_parts,
            eos_token_id=config.eos_token_id,
        )
