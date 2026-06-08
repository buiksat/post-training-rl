def pass_at_k(num_samples, num_correct, k):
    """Estimate pass@k from n sampled completions with c correct completions.

    Exercise 8:
    Implement the standard unbiased pass@k estimator used for code/reasoning
    evals. Think through edge cases:
    - no correct samples
    - k >= num_samples
    - all samples correct
    """
    raise NotImplementedError("Exercise 8: implement pass@k estimator.")


def bootstrap_mean_ci(values, num_bootstrap=1000, confidence=0.95, seed=0):
    """Bootstrap confidence interval for a mean metric.

    Exercise 9:
    Given a list/tensor of scalar per-example outcomes, return
    (mean, lower, upper). This is interview-relevant because post-training
    gains are often small and need uncertainty estimates, not just point
    estimates.
    """
    raise NotImplementedError("Exercise 9: implement bootstrap mean CI.")


def paired_win_rate(before_scores, after_scores):
    """Compare two systems on matched examples.

    Exercise 10:
    Return win_rate, loss_rate, tie_rate for after vs before. This is useful
    for A/B-style evals where every prompt has a baseline and candidate result.
    """
    raise NotImplementedError("Exercise 10: implement paired win-rate eval.")
