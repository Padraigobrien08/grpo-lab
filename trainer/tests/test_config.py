from pathlib import Path

import pytest
from pydantic import ValidationError

from grpo_lab.config import load_config

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_validate(path):
    load_config(path)


def test_overrides_are_typed():
    cfg = load_config(CONFIGS / "toy.yaml", ["algo.beta=0.0", "seed=3", "algo.clip_eps_high=0.28"])
    assert cfg.algo.beta == 0.0 and cfg.seed == 3 and cfg.algo.clip_eps_high == 0.28


def test_unknown_keys_fail_loudly():
    with pytest.raises(ValidationError):
        load_config(CONFIGS / "toy.yaml", ["algo.btea=0.1"])


def test_model_needs_exactly_one_source():
    with pytest.raises(ValidationError):
        load_config(CONFIGS / "toy.yaml", ["model.name_or_path=foo"])
