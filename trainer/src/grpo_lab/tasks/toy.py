"""Toy task for checking that the gradient points the right way: emit a target token.

A random-init policy emits the target about 1/vocab of the time. GRPO should push that toward
1.0 within tens of steps on CPU. If it doesn't, the bug is in the algorithm, not the LLM setup.
"""

from __future__ import annotations

import random

from grpo_lab.models import TOY_PAD_ID
from grpo_lab.tasks.base import Prompt, Task


class ToyTargetTask(Task):
    def __init__(self, vocab_size: int, prompt_len: int = 4, target_token: int = 3) -> None:
        if not 1 <= target_token < vocab_size:
            raise ValueError("target_token must be a non-pad id inside the vocab")
        self.vocab_size = vocab_size
        self.prompt_len = prompt_len
        self.target_token = target_token
        self.reward_weights = {"target_frac": 1.0}
        self.gold_component = "target_frac"
        self.eos_ids = []
        self.pad_id = TOY_PAD_ID

    def sample_prompts(self, n: int, rng: random.Random) -> list[Prompt]:
        prompts = []
        for _ in range(n):
            ids = [rng.randrange(1, self.vocab_size) for _ in range(self.prompt_len)]
            prompts.append(Prompt(id="-".join(map(str, ids)), input_ids=ids, text=self.decode(ids)))
        return prompts

    def eval_prompts(self, n: int) -> list[Prompt]:
        return self.sample_prompts(n, random.Random(12345))  # fixed seed: same set every call

    def score(self, prompts: list[Prompt], completions: list[list[int]]) -> dict[str, list[float]]:
        return {
            "target_frac": [
                sum(t == self.target_token for t in c) / max(len(c), 1) for c in completions
            ]
        }

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"t{t}" for t in token_ids)
