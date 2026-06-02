import pytest
import torch

from post_training_rl.ppo import ppo_clipped_loss


# These tests define the target behavior, but skip while functions are stubbed.
# Once each NotImplementedError is replaced, the skipped checks become real tests.


REQUIRED_PPO_METRICS = {
    "loss",
    "clip_fraction",
    "approx_kl",
    "ratio_mean",
}


def call_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip("pending implementation")


def assert_scalar_finite(value):
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    assert tensor.ndim == 0
    assert torch.isfinite(tensor).item()


def scalar_float(value):
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.detach().cpu().item()


def test_ppo_outputs_scalar_loss_and_metrics_are_finite():
    logp = torch.tensor([-1.0, -1.5, -0.7])
    old_logp = torch.tensor([-1.1, -1.4, -0.8])
    advantages = torch.tensor([1.0, -0.5, 0.25])

    loss, metrics = call_or_skip(ppo_clipped_loss, logp, old_logp, advantages)

    assert_scalar_finite(loss)
    assert isinstance(metrics, dict)
    assert REQUIRED_PPO_METRICS.issubset(metrics)
    for key in REQUIRED_PPO_METRICS:
        assert_scalar_finite(metrics[key])


def test_ppo_clip_fraction_is_between_zero_and_one():
    logp = torch.tensor([-1.0, -1.0, -1.0])
    old_logp = torch.tensor([-1.0, -1.5, -0.5])
    advantages = torch.tensor([1.0, 1.0, 1.0])

    _, metrics = call_or_skip(
        ppo_clipped_loss,
        logp,
        old_logp,
        advantages,
        clip_eps=0.2,
    )

    clip_fraction = scalar_float(metrics["clip_fraction"])
    assert 0.0 <= clip_fraction <= 1.0


def test_ppo_zero_advantage_case_behaves_sanely():
    logp = torch.tensor([-1.0, -2.0, -3.0])
    old_logp = logp.clone()
    advantages = torch.zeros(3)

    loss, metrics = call_or_skip(ppo_clipped_loss, logp, old_logp, advantages)

    assert_scalar_finite(loss)
    assert scalar_float(loss) == pytest.approx(0.0)
    assert scalar_float(metrics["ratio_mean"]) == pytest.approx(1.0)
    assert scalar_float(metrics["approx_kl"]) == pytest.approx(0.0)
    assert 0.0 <= scalar_float(metrics["clip_fraction"]) <= 1.0
