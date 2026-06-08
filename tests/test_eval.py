import pytest

from post_training_rl.eval import bootstrap_mean_ci, paired_win_rate, pass_at_k


def call_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as error:
        pytest.skip(str(error))


def test_pass_at_k_known_values():
    assert call_or_skip(pass_at_k, num_samples=4, num_correct=0, k=1) == 0.0
    assert call_or_skip(pass_at_k, num_samples=4, num_correct=4, k=1) == 1.0
    assert call_or_skip(pass_at_k, num_samples=4, num_correct=1, k=1) == pytest.approx(0.25)
    assert call_or_skip(pass_at_k, num_samples=4, num_correct=1, k=2) == pytest.approx(0.5)
    assert call_or_skip(pass_at_k, num_samples=4, num_correct=1, k=4) == 1.0


def test_bootstrap_mean_ci_contains_empirical_mean():
    mean, lower, upper = call_or_skip(
        bootstrap_mean_ci,
        [0.0, 1.0, 1.0, 1.0],
        num_bootstrap=200,
        confidence=0.90,
        seed=1,
    )

    assert mean == pytest.approx(0.75)
    assert lower <= mean <= upper
    assert 0.0 <= lower <= upper <= 1.0


def test_paired_win_rate_counts_wins_losses_and_ties():
    result = call_or_skip(
        paired_win_rate,
        before_scores=[0.0, 1.0, 0.5, 0.5],
        after_scores=[1.0, 0.0, 0.5, 1.0],
    )

    assert result["win_rate"] == pytest.approx(0.5)
    assert result["loss_rate"] == pytest.approx(0.25)
    assert result["tie_rate"] == pytest.approx(0.25)
