import math

import pytest
import torch

from grpo_lab.algo.losses import grpo_loss


def _grad(logp_values, old_values, adv, mask, **kw):
    logp = torch.tensor(logp_values, requires_grad=True)
    out = grpo_loss(logp, torch.tensor(old_values), torch.tensor(adv), torch.tensor(mask), **kw)
    out.loss.backward()
    return logp.grad, out


def test_on_policy_gradient_is_minus_advantage_over_tokens():
    # At ratio == 1 the clipped objective reduces to the vanilla policy gradient.
    lp = [[-1.0, -2.0, -0.5]]
    grad, _ = _grad(lp, lp, [2.0], [[1, 1, 0]], aggregation="token_mean")
    torch.testing.assert_close(grad, torch.tensor([[-1.0, -1.0, 0.0]]))


@pytest.mark.parametrize(
    ("ratio", "adv", "expect_zero_grad"),
    [
        (1.5, +1.0, True),  # pushing an already-raised prob further up: clipped
        (1.5, -1.0, False),  # pessimistic bound keeps the unclipped term: gradient flows
        (0.5, -1.0, True),  # pushing an already-lowered prob further down: clipped
        (0.5, +1.0, False),
    ],
)
def test_clipping(ratio, adv, expect_zero_grad):
    grad, out = _grad([[0.0]], [[-math.log(ratio)]], [adv], [[1]])
    assert (grad.abs().sum().item() == 0.0) == expect_zero_grad
    clipped = out.stat_sums["clip_frac_low"] + out.stat_sums["clip_frac_high"]
    assert (clipped == 1.0) == expect_zero_grad


def test_clip_higher_unblocks_upward_updates():
    old = [[-math.log(1.25)]]
    grad_sym, _ = _grad([[0.0]], old, [1.0], [[1]], clip_eps_high=0.2)
    grad_dapo, _ = _grad([[0.0]], old, [1.0], [[1]], clip_eps_high=0.28)
    assert grad_sym.abs().sum() == 0
    assert grad_dapo.abs().sum() > 0


def test_masked_tokens_do_not_affect_loss():
    mask = torch.tensor([[1, 1, 0, 0]])
    adv = torch.tensor([1.0])
    a = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    b = torch.tensor([[-1.0, -2.0, 99.0, -99.0]])
    for agg in ("sequence_mean", "token_mean", "constant"):
        la = grpo_loss(a, a - 0.1, adv, mask, aggregation=agg, max_completion_len=4).loss
        lb = grpo_loss(b, b - 0.1, adv, mask, aggregation=agg, max_completion_len=4).loss
        torch.testing.assert_close(la, lb)


def test_aggregation_token_weights():
    # Two sequences of length 2 and 6, same advantage: per-token gradient weight shows the bias.
    lp = [[0.0] * 6, [0.0] * 6]
    mask = [[1, 1, 0, 0, 0, 0], [1] * 6]
    g_seq, _ = _grad(lp, lp, [1.0, 1.0], mask, aggregation="sequence_mean")
    g_tok, _ = _grad(lp, lp, [1.0, 1.0], mask, aggregation="token_mean")
    g_const, _ = _grad(lp, lp, [1.0, 1.0], mask, aggregation="constant", max_completion_len=6)
    # sequence_mean: 1 / (n_seq * len_i) -> short-sequence tokens weigh 3x more
    assert g_seq[0, 0].item() == pytest.approx(-1 / 4)
    assert g_seq[1, 0].item() == pytest.approx(-1 / 12)
    # token_mean: 1 / total_tokens for everyone
    assert g_tok[0, 0].item() == pytest.approx(-1 / 8)
    assert g_tok[1, 0].item() == pytest.approx(-1 / 8)
    # constant: 1 / (n_seq * max_len) for everyone
    assert g_const[0, 0].item() == pytest.approx(-1 / 12)
    assert g_const[1, 0].item() == pytest.approx(-1 / 12)


@pytest.mark.parametrize("agg", ["sequence_mean", "token_mean", "constant"])
def test_micro_batch_losses_sum_to_full_batch_loss(agg):
    torch.manual_seed(0)
    lp = torch.randn(6, 5)
    old = lp + 0.1 * torch.randn(6, 5)
    ref = lp + 0.1 * torch.randn(6, 5)
    adv = torch.randn(6)
    mask = (torch.rand(6, 5) > 0.3).long()
    mask[:, 0] = 1
    kw = dict(ref_logprobs=ref, beta=0.1, aggregation=agg, max_completion_len=5)
    full = grpo_loss(lp, old, adv, mask, **kw).loss
    totals = dict(num_sequences_total=6, num_tokens_total=mask.sum().item())
    parts = sum(
        grpo_loss(lp[s], old[s], adv[s], mask[s], **{**kw, "ref_logprobs": ref[s]}, **totals).loss
        for s in (slice(0, 2), slice(2, 3), slice(3, 6))
    )
    torch.testing.assert_close(parts, full)


def test_kl_term_has_zero_gradient_at_reference():
    lp = [[-1.0, -2.0]]
    grad_with_kl, out = _grad(lp, lp, [0.0], [[1, 1]], ref_logprobs=torch.tensor(lp), beta=1.0)
    assert grad_with_kl.abs().sum() == 0
    assert out.stat_sums["kl"] == pytest.approx(0.0)


def test_argument_validation():
    x = torch.zeros(1, 2)
    with pytest.raises(ValueError):
        grpo_loss(x, x, torch.zeros(1), torch.ones(1, 2), beta=0.1)
    with pytest.raises(ValueError):
        grpo_loss(x, x, torch.zeros(1), torch.ones(1, 2), aggregation="constant")
