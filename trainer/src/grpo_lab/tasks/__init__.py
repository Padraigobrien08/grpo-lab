from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from grpo_lab.config import Config
from grpo_lab.tasks.base import Prompt, Task


def build_task(cfg: Config, tokenizer: PreTrainedTokenizerBase | None) -> Task:
    if cfg.task.name == "toy_target":
        from grpo_lab.tasks.toy import ToyTargetTask

        if cfg.model.toy is None:
            raise ValueError("toy_target task needs model.toy")
        return ToyTargetTask(vocab_size=cfg.model.toy.vocab_size, **cfg.task.params)
    if cfg.task.name == "gsm8k":
        from grpo_lab.tasks.gsm8k import GSM8KTask

        if tokenizer is None:
            raise ValueError("gsm8k task needs a tokenizer (use model.name_or_path)")
        return GSM8KTask(tokenizer, **cfg.task.params)
    raise ValueError(f"unknown task {cfg.task.name}")


__all__ = ["Prompt", "Task", "build_task"]
