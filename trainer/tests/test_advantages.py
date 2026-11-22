import pytest
import torch

from grpo_lab.algo.advantages import group_advantages, zero_advantage_fraction


def test_std_scaled_advantages_are_standardised_per_group():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 5.0, 3.0, 4.0, 4.0])
    adv = group_advantages(rewards, group_size=4).view(2, 4)
    torch.testing.assert_close(adv.mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(adv.std(dim=1), torch.ones(2), atol=1e-3, rtol=0)


def test_none_scale_only_centres():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    adv = group_advantages(rewards, group_size=4, scale="none")
    torch.testing.assert_close(adv, torch.tensor([0.75, -0.25, -0.25, -0.25]))


def test_uniform_group_gives_zero_not_nan():
    adv = group_advantages(torch.ones(8), group_size=4)
    assert torch.isfinite(adv).all()
    assert (adv == 0).all()


def test_groups_are_independent():
    a = group_advantages(torch.tensor([0.0, 1.0, 10.0, 20.0]), group_size=2)
    b = group_advantages(torch.tensor([0.0, 1.0]), group_size=2)
    torch.testing.assert_close(a[:2], b)


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(5), group_size=2)


def test_zero_advantage_fraction():
    rewards = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    assert zero_advantage_fraction(rewards, group_size=2) == pytest.approx(2 / 3)
