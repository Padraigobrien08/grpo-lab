"""Policy loading: a Hugging Face checkpoint, or a tiny random Qwen2 for tests."""

from __future__ import annotations

import contextlib

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from grpo_lab.config import ModelConfig

TOY_PAD_ID = 0  # reserved in toy vocabularies; toy prompts use ids >= 1

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device: torch.device, amp_dtype: str | None) -> contextlib.AbstractContextManager:
    """Mixed-precision forward passes; a no-op when ``amp_dtype`` is None."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=_DTYPES[amp_dtype])


def load_policy(
    cfg: ModelConfig, device: torch.device
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase | None]:
    if cfg.toy is not None:
        toy = cfg.toy
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=toy.vocab_size,
                hidden_size=toy.hidden_size,
                intermediate_size=toy.hidden_size * 2,
                num_hidden_layers=toy.num_layers,
                num_attention_heads=toy.num_heads,
                num_key_value_heads=toy.num_heads,
                max_position_embeddings=256,
                tie_word_embeddings=True,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=TOY_PAD_ID,
            )
        )
        tokenizer = None
    else:
        assert cfg.name_or_path is not None
        model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, dtype=_DTYPES[cfg.dtype])
        tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model.to(device), tokenizer
