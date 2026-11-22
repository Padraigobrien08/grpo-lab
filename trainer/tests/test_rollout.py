import torch

from grpo_lab.rollout import completion_mask_from_eos, generate_group_rollouts, left_pad


def test_left_pad():
    ids, mask = left_pad([[1, 2], [3, 4, 5]], pad_id=0, device=torch.device("cpu"))
    assert ids.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert mask.tolist() == [[0, 1, 1], [1, 1, 1]]


def test_completion_mask_includes_first_eos_only():
    completions = torch.tensor([[4, 9, 5, 9], [4, 4, 4, 4], [9, 0, 0, 0]])
    mask, truncated = completion_mask_from_eos(completions, eos_ids=[9])
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]]
    assert truncated.tolist() == [False, True, False]


def test_completion_mask_without_eos_is_full_length():
    mask, truncated = completion_mask_from_eos(torch.ones(2, 3, dtype=torch.long), eos_ids=[])
    assert mask.all() and not truncated.any()


def test_group_rollouts_shape_and_grouping(tiny_model):
    torch.manual_seed(0)
    prompts = [[5, 6], [7, 8, 9]]
    batch = generate_group_rollouts(
        tiny_model, prompts, group_size=3, max_new_tokens=5, temperature=1.0, pad_id=0, eos_ids=[]
    )
    assert batch.input_ids.shape == (6, 3 + 5)
    assert batch.prompt_len == 3
    # rows 0-2 share prompt 0, rows 3-5 share prompt 1
    assert all(batch.input_ids[i, 1:3].tolist() == [5, 6] for i in range(3))
    assert all(batch.input_ids[i, :3].tolist() == [7, 8, 9] for i in range(3, 6))
    assert batch.attention_mask[:3, 0].tolist() == [0, 0, 0]
    assert len(batch.completion_token_lists()[0]) == 5
