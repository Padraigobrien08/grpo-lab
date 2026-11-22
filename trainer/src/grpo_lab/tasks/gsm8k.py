"""GSM8K with a binary correctness reward plus a small format reward."""

from __future__ import annotations

import random

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from grpo_lab.rewards.math import gold_answer, has_final_answer_format, is_correct
from grpo_lab.tasks.base import Prompt, Task

SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "On the final line, write the answer as '#### <number>' and nothing else."
)


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

    def sample_prompts(self, n: int, rng: random.Random) -> list[Prompt]:
        prompts = []
        for ex in rng.sample(self.examples, k=min(n, len(self.examples))):
            ids, text = self._encode(ex["question"])
            prompts.append(Prompt(id=ex["id"], input_ids=ids, text=text, meta={"gold": ex["gold"]}))
        return prompts

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
