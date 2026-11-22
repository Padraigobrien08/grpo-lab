"""Task interface. The trainer only sees token ids; text and verification live in tasks."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Prompt:
    id: str
    input_ids: list[int]
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    #: weight of each reward component in the total reward the policy is trained on
    reward_weights: dict[str, float]
    #: component that measures true success, logged separately as gold_reward (None = no gold)
    gold_component: str | None = None
    #: token ids that end a completion (empty = fixed-length completions)
    eos_ids: list[int]
    pad_id: int

    @abstractmethod
    def sample_prompts(self, n: int, rng: random.Random) -> list[Prompt]: ...

    @abstractmethod
    def score(self, prompts: list[Prompt], completions: list[list[int]]) -> dict[str, list[float]]:
        """Reward components per completion; ``prompts[i]`` is the prompt of ``completions[i]``."""

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str: ...

    def total_reward(self, components: dict[str, list[float]]) -> list[float]:
        n = len(next(iter(components.values())))
        return [
            sum(self.reward_weights[k] * components[k][i] for k in self.reward_weights)
            for i in range(n)
        ]
