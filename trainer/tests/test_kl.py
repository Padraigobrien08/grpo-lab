import torch

from grpo_lab.algo.kl import kl_k1, kl_k3


def test_k3_is_nonnegative_and_zero_at_equality():
    logp = torch.randn(1000)
    ref = torch.randn(1000)
    assert (kl_k3(logp, ref) >= 0).all()
    assert torch.allclose(kl_k3(logp, logp), torch.zeros(1000))


def test_estimators_match_exact_kl_in_expectation():
    torch.manual_seed(0)
    p = torch.softmax(torch.tensor([2.0, 0.5, -1.0, 0.0]), dim=0)
    q = torch.softmax(torch.tensor([0.0, 1.0, 0.0, -0.5]), dim=0)
    exact = (p * (p.log() - q.log())).sum()
    samples = torch.multinomial(p, 200_000, replacement=True)  # tokens sampled from the policy
    logp, ref = p.log()[samples], q.log()[samples]
    torch.testing.assert_close(kl_k1(logp, ref).mean(), exact, atol=5e-3, rtol=0)
    torch.testing.assert_close(kl_k3(logp, ref).mean(), exact, atol=5e-3, rtol=0)
    # k3 should be the lower-variance estimator
    assert kl_k3(logp, ref).var() < kl_k1(logp, ref).var()
