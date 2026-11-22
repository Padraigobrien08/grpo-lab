"""GSM8K with a binary correctness reward plus a small format reward (answer in \\boxed{})."""

from __future__ import annotations

import random

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from grpo_lab.rewards.math import gold_answer, has_final_answer_format, is_correct
from grpo_lab.tasks.base import Prompt, Task

# Qwen's own math prompt. Keeping the model in its trained output style matters at 0.5B:
# see grpo_lab.rewards.math for the numbers.
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


class GSM8KTask(Task):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "train",
        max_examples: int | None = None,
        correct_weight: float = 1.0,
        format_weight: float = 0.1,
    ) -> None:
        self.tokenizer = tokenizer
        ds = load_dataset("openai/gsm8k", "main", split=split)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self.examples = [
            {"id": f"{split}-{i}", "question": ex["question"], "gold": gold_answer(ex["answer"])}
            for i, ex in enumerate(ds)
        ]
        self.reward_weights = {"correct": correct_weight, "format": format_weight}
        self.gold_component = "correct"
        assert tokenizer.pad_token_id is not None
        self.pad_id = tokenizer.pad_token_id
        # Chat models end a turn with an end-of-turn token that may differ from eos_token.
        eos = {tokenizer.eos_token_id}
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
            eos.add(im_end)
        self.eos_ids = sorted(i for i in eos if i is not None)

    def _encode(self, question: str) -> tuple[list[int], str]:
        # Render to text, then tokenize: the return type of apply_chat_template(tokenize=True)
        # changed across transformers versions.
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(text, add_special_tokens=False)["input_ids"], question

    def _prompt(self, ex: dict) -> Prompt:
        ids, text = self._encode(ex["question"])
        return Prompt(id=ex["id"], input_ids=ids, text=text, meta={"gold": ex["gold"]})

    def sample_prompts(self, n: int, rng: random.Random) -> list[Prompt]:
        return [self._prompt(ex) for ex in rng.sample(self.examples, k=min(n, len(self.examples)))]

    def eval_prompts(self, n: int) -> list[Prompt]:
        """The first ``n`` problems of the GSM8K test split (1319 total)."""
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.select(range(min(n, len(ds))))
        examples = [
            {"id": f"test-{i}", "question": ex["question"], "gold": gold_answer(ex["answer"])}
            for i, ex in enumerate(ds)
        ]
        return [self._prompt(ex) for ex in examples]

    def score(self, prompts: list[Prompt], completions: list[list[int]]) -> dict[str, list[float]]:
        texts = [self.decode(c) for c in completions]
        return {
            "correct": [
                float(is_correct(t, p.meta["gold"])) for t, p in zip(texts, prompts, strict=True)
            ],
            "format": [float(has_final_answer_format(t)) for t in texts],
        }

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
