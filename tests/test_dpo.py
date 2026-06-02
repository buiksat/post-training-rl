import pytest
import torch

from post_training_rl.dpo import dpo_loss
from post_training_rl.utils import compute_sequence_logps


# These tests define the target behavior, but skip while functions are stubbed.
# Once each NotImplementedError is replaced, the skipped checks become real tests.


REQUIRED_DPO_METRICS = {
    "loss",
    "dpo_loss",
    "nll_loss",
    "reward_chosen",
    "reward_rejected",
    "reward_margin",
    "reward_accuracy",
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


def test_dpo_outputs_scalar_loss_and_metrics_are_finite():
    policy_chosen_logps = torch.tensor([-1.0, -0.5, -0.25])
    policy_rejected_logps = torch.tensor([-2.0, -1.5, -1.25])
    reference_chosen_logps = torch.tensor([-1.2, -0.7, -0.4])
    reference_rejected_logps = torch.tensor([-1.8, -1.0, -0.9])

    loss, metrics = call_or_skip(
        dpo_loss,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )

    assert_scalar_finite(loss)
    assert isinstance(metrics, dict)
    assert REQUIRED_DPO_METRICS.issubset(metrics)
    for key in REQUIRED_DPO_METRICS:
        assert_scalar_finite(metrics[key])


def test_dpo_perfect_chosen_rejected_ordering_has_reward_accuracy_one():
    policy_chosen_logps = torch.tensor([-0.5, -0.25, -0.1])
    policy_rejected_logps = torch.tensor([-2.0, -1.5, -1.0])
    reference_chosen_logps = torch.zeros(3)
    reference_rejected_logps = torch.zeros(3)

    _, metrics = call_or_skip(
        dpo_loss,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )

    assert scalar_float(metrics["reward_accuracy"]) == pytest.approx(1.0)


def test_dpo_use_nll_without_chosen_lengths_raises_value_error():
    policy_chosen_logps = torch.tensor([-1.0, -0.5])
    policy_rejected_logps = torch.tensor([-2.0, -1.5])
    reference_chosen_logps = torch.tensor([-1.0, -0.5])
    reference_rejected_logps = torch.tensor([-2.0, -1.5])

    try:
        with pytest.raises(ValueError):
            dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                use_nll=True,
            )
    except NotImplementedError:
        pytest.skip("pending implementation")


def test_compute_sequence_logps_shapes_finite_and_fully_masked_zero():
    logits = torch.randn(2, 4, 5)
    labels = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ])
    loss_mask = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    seq_logp_sum, seq_len = call_or_skip(
        compute_sequence_logps,
        logits,
        labels,
        loss_mask,
    )

    assert seq_logp_sum.shape == (2,)
    assert seq_len.shape == (2,)
    assert torch.isfinite(seq_logp_sum).all().item()
    assert torch.isfinite(seq_len).all().item()
    assert seq_logp_sum[1].item() == pytest.approx(0.0)
    assert seq_len[1].item() == pytest.approx(0.0)
