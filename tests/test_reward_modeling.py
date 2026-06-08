import pytest
import torch

from post_training_rl.reward_modeling import (
    detect_reward_hacking,
    pairwise_preference_loss,
    reward_accuracy,
)


def call_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as error:
        pytest.skip(str(error))


def assert_scalar_finite(value):
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    assert tensor.ndim == 0
    assert torch.isfinite(tensor).item()


def test_pairwise_preference_loss_rewards_correct_ordering():
    good_loss = call_or_skip(
        pairwise_preference_loss,
        torch.tensor([2.0, 3.0]),
        torch.tensor([0.0, 1.0]),
    )
    bad_loss = call_or_skip(
        pairwise_preference_loss,
        torch.tensor([0.0, 1.0]),
        torch.tensor([2.0, 3.0]),
    )

    assert_scalar_finite(good_loss)
    assert_scalar_finite(bad_loss)
    assert good_loss < bad_loss


def test_reward_accuracy_counts_preference_ordering():
    accuracy = call_or_skip(
        reward_accuracy,
        torch.tensor([2.0, 0.0, 1.0, 4.0]),
        torch.tensor([1.0, 1.0, 1.0, 3.0]),
    )

    assert accuracy == pytest.approx(0.5)


def test_detect_reward_hacking_returns_named_diagnostics():
    diagnostics = call_or_skip(
        detect_reward_hacking,
        reward_scores=[0.1, 0.3, 0.8],
        task_scores=[0.7, 0.6, 0.4],
        kl_values=[0.01, 0.05, 0.20],
    )

    assert isinstance(diagnostics, dict)
    assert "reward_up_task_down" in diagnostics
    assert "kl_increased" in diagnostics
