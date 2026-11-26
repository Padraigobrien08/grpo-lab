"""Typed experiment config, loaded from YAML with optional ``dotted.key=value`` overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"  # weights + optimizer state
    # Compute dtype for forward passes (torch.autocast); weights stay in ``dtype``. bf16 only:
    # fp16 would also need loss scaling.
    amp_dtype: Literal["bfloat16"] | None = None
    gradient_checkpointing: bool = False

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ModelConfig:
        if (self.name_or_path is None) == (self.toy is None):
            raise ValueError("set exactly one of model.name_or_path or model.toy")
        return self

    @model_validator(mode="after")
    def _trainable_weights_in_fp32(self) -> ModelConfig:
        # Full fine-tuning updates are ~lr (1e-6) per step; bf16 can't represent changes below
        # ~1e-4 of a typical weight, so Adam steps silently round to zero and nothing learns.
        if self.dtype != "float32":
            raise ValueError(
                f"model.dtype={self.dtype} would lose small optimizer updates to rounding; "
                "keep model.dtype=float32 and set model.amp_dtype=bfloat16 for speed"
            )
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


class EvalConfig(_Strict):
    # Greedy held-out eval every N steps, plus before step 1 and after the last. 0 = off.
    every: int = 0
    num_examples: int = 500  # fixed held-out subset, the same at every eval
    batch_size: int = 64
    max_new_tokens: int | None = None  # None = algo.max_new_tokens


class LogConfig(_Strict):
    store_dir: str = "runs"
    rollout_every: int = 10  # also always logged on the first and last step
    # On rollout-logging steps, also log per-token traces for the first N groups. 0 = off.
    trace_groups: int = Field(default=2, ge=0)
    print_every: int = 1
    save_final: bool = False  # save the final policy (HF format) to checkpoint_dir/<run_id>
    checkpoint_dir: str = "checkpoints"


class Config(_Strict):
    name: str
    seed: int = 0
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    model: ModelConfig
    task: TaskConfig
    algo: AlgoConfig = AlgoConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
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
