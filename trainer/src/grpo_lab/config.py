"""Typed experiment config, loaded from YAML with optional ``dotted.key=value`` overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from grpo_lab.algo.advantages import AdvantageScale
from grpo_lab.algo.losses import Aggregation


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # typos in YAML fail loudly


class ToyModelConfig(_Strict):
    vocab_size: int = 16
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4


class ModelConfig(_Strict):
    name_or_path: str | None = None  # Hugging Face id or local path
    toy: ToyModelConfig | None = None  # tiny random Qwen2: no download, for tests and smoke runs
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    gradient_checkpointing: bool = False

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ModelConfig:
        if (self.name_or_path is None) == (self.toy is None):
            raise ValueError("set exactly one of model.name_or_path or model.toy")
        return self


class TaskConfig(_Strict):
    name: Literal["toy_target", "gsm8k"]
    params: dict[str, Any] = {}


class AlgoConfig(_Strict):
    group_size: int = 8
    beta: float = 0.04  # KL coefficient; 0 disables the reference model entirely
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2  # > clip_eps_low gives DAPO clip-higher
    loss_aggregation: Aggregation = "sequence_mean"
    advantage_scale: AdvantageScale = "std"
    num_iterations: int = 1  # optimisation passes per batch of rollouts (mu in the GRPO paper)
    temperature: float = 1.0
    max_new_tokens: int = 256


class TrainConfig(_Strict):
    steps: int = 100
    prompts_per_step: int = 8
    micro_batch_size: int = 8  # sequences per forward/backward
    lr: float = 1e-6
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.99)
    max_grad_norm: float = 1.0
    warmup_steps: int = 0


class LogConfig(_Strict):
    store_dir: str = "runs"
    rollout_every: int = 10  # also always logged on the first and last step
    print_every: int = 1


class Config(_Strict):
    name: str
    seed: int = 0
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    model: ModelConfig
    task: TaskConfig
    algo: AlgoConfig = AlgoConfig()
    train: TrainConfig = TrainConfig()
    log: LogConfig = LogConfig()


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides. Values are parsed as YAML: ``0.0``, ``true``, ``[1,2]``."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = data
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return data


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return Config.model_validate(apply_overrides(data, overrides or []))
