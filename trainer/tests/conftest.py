import pytest
import torch

from grpo_lab.config import ModelConfig, ToyModelConfig
from grpo_lab.models import load_policy


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model, _ = load_policy(
        ModelConfig(toy=ToyModelConfig(vocab_size=32, hidden_size=32, num_layers=2, num_heads=4)),
        torch.device("cpu"),
    )
    return model.eval()
