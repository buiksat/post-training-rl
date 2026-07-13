import pytest
import torch

from post_training_rl.eval import pass_at_k
from post_training_rl.grpo import group_normalize_rewards
from post_training_rl.ppo import ppo_clipped_loss
from post_training_rl.reward_modeling import detect_reward_hacking


def test_pass_at_k_rejects_invalid_counts():
    with pytest.raises(ValueError):
        pass_at_k(4, 5, 1)
    with pytest.raises(ValueError):
        pass_at_k(4, 0, 0)


def test_group_normalization_has_zero_signal_for_singleton_group():
    advantages = group_normalize_rewards(torch.tensor([3.0]), torch.tensor([7]))
    assert advantages.tolist() == [0.0]


def test_ppo_clips_positive_and_negative_advantages():
    logp = torch.tensor([0.0, 0.0])
    old_logp = torch.tensor([-2.0, 2.0])
    advantages = torch.tensor([1.0, -1.0])
    _, metrics = ppo_clipped_loss(logp, old_logp, advantages)
    assert metrics["clip_fraction"].item() == 1.0


def test_reward_hacking_detects_divergent_trends():
    result = detect_reward_hacking([0.1, 0.8], [0.9, 0.4], [0.01, 0.2])
    assert result["reward_up_task_down"] is True
    assert result["kl_increased"] is True
