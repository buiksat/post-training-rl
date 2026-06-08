import pytest
import torch

from post_training_rl.grpo import group_normalize_rewards, grpo_clipped_loss


def call_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as error:
        pytest.skip(str(error))


def assert_scalar_finite(value):
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    assert tensor.ndim == 0
    assert torch.isfinite(tensor).item()


def test_group_normalize_rewards_normalizes_within_prompt_group():
    rewards = torch.tensor([1.0, 3.0, 10.0, 14.0])
    group_ids = torch.tensor([0, 0, 1, 1])

    advantages = call_or_skip(group_normalize_rewards, rewards, group_ids)

    assert advantages.shape == rewards.shape
    assert advantages.tolist() == pytest.approx([-1.0, 1.0, -1.0, 1.0])


def test_grpo_clipped_loss_outputs_metrics():
    logp = torch.tensor([-1.0, -1.5, -0.7])
    old_logp = torch.tensor([-1.1, -1.4, -0.8])
    group_advantages = torch.tensor([1.0, -0.5, 0.25])

    loss, metrics = call_or_skip(
        grpo_clipped_loss,
        logp,
        old_logp,
        group_advantages,
    )

    assert_scalar_finite(loss)
    for key in {"loss", "clip_fraction", "approx_kl", "ratio_mean"}:
        assert key in metrics
        assert_scalar_finite(metrics[key])
