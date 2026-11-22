import torch

from grpo_lab.algo.logprobs import completion_logprobs
from grpo_lab.rollout import left_pad


def test_matches_manual_log_softmax(tiny_model):
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids)
    lp, ent = completion_logprobs(tiny_model, ids, mask, prompt_len=3, return_entropy=True)
    with torch.no_grad():
        logits = tiny_model(input_ids=ids).logits[0]
    expected = torch.log_softmax(logits, -1)[2:5].gather(-1, ids[0, 3:].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(lp[0], expected, atol=1e-5, rtol=0)
    probs = torch.softmax(logits[2:5], -1)
    torch.testing.assert_close(ent[0], -(probs * probs.log()).sum(-1), atol=1e-5, rtol=0)


def test_left_padding_does_not_change_logprobs(tiny_model):
    short, long = [5, 6], [7, 8, 9, 10, 11]
    completion = [12, 13, 14]
    ids, mask = left_pad([short, long], pad_id=0, device=torch.device("cpu"))
    batch_ids = torch.cat([ids, torch.tensor([completion, completion])], dim=1)
    batch_mask = torch.cat([mask, torch.ones(2, 3, dtype=torch.long)], dim=1)
    lp_batch, _ = completion_logprobs(tiny_model, batch_ids, batch_mask, prompt_len=ids.shape[1])

    alone = torch.tensor([short + completion])
    lp_alone, _ = completion_logprobs(tiny_model, alone, torch.ones_like(alone), prompt_len=2)
    torch.testing.assert_close(lp_batch[0], lp_alone[0], atol=1e-5, rtol=0)


def test_scored_distribution_matches_sampling_distribution(tiny_model):
    """The log-probs we train on must be those of the distribution generate() sampled from."""
    torch.manual_seed(0)
    temperature = 0.7
    ids, mask = left_pad([[5, 6], [7, 8, 9, 10]], pad_id=0, device=torch.device("cpu"))
    out = tiny_model.generate(
        input_ids=ids,
        attention_mask=mask,
        do_sample=True,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
        max_new_tokens=6,
        pad_token_id=0,
        eos_token_id=None,
        output_logits=True,
        return_dict_in_generate=True,
    )
    raw = torch.stack(out.logits, dim=1).float() / temperature  # (B, C, V)
    sampled = out.sequences[:, ids.shape[1] :]
    expected = torch.log_softmax(raw, -1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

    full_mask = torch.cat([mask, torch.ones_like(sampled)], dim=1)
    lp, _ = completion_logprobs(
        tiny_model, out.sequences, full_mask, prompt_len=ids.shape[1], temperature=temperature
    )
    torch.testing.assert_close(lp, expected, atol=1e-4, rtol=0)
