import math
import random


def pass_at_k(num_samples, num_correct, k):
    """Estimate pass@k from n sampled completions with c correct completions.

    Exercise 8:
    Implement the standard unbiased pass@k estimator used for code/reasoning
    evals. Think through edge cases:
    - no correct samples
    - k >= num_samples
    - all samples correct
    """
    n, c, k = int(num_samples), int(num_correct), int(k)
    if n <= 0 or c < 0 or c > n or k <= 0:
        raise ValueError("require n > 0, 0 <= c <= n, and k > 0")
    if c == 0:
        return 0.0
    if k >= n or c == n:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def bootstrap_mean_ci(values, num_bootstrap=1000, confidence=0.95, seed=0):
    """Bootstrap confidence interval for a mean metric.

    Exercise 9:
    Given a list/tensor of scalar per-example outcomes, return
    (mean, lower, upper). This is interview-relevant because post-training
    gains are often small and need uncertainty estimates, not just point
    estimates.
    """
    if num_bootstrap <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("num_bootstrap must be positive and confidence must be in (0, 1)")
    values = [float(value) for value in values]
    if not values:
        raise ValueError("values cannot be empty")
    rng = random.Random(seed)
    mean = sum(values) / len(values)
    samples = []
    for _ in range(num_bootstrap):
        samples.append(sum(rng.choice(values) for _ in values) / len(values))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, min(len(samples) - 1, int(alpha * len(samples))))
    upper_index = max(0, min(len(samples) - 1, int((1.0 - alpha) * len(samples)) - 1))
    return mean, samples[lower_index], samples[upper_index]


def paired_win_rate(before_scores, after_scores):
    """Compare two systems on matched examples.

    Exercise 10:
    Return win_rate, loss_rate, tie_rate for after vs before. This is useful
    for A/B-style evals where every prompt has a baseline and candidate result.
    """
    if len(before_scores) != len(after_scores) or not before_scores:
        raise ValueError("before_scores and after_scores must be non-empty and aligned")
    wins = losses = ties = 0
    for before, after in zip(before_scores, after_scores):
        if after > before:
            wins += 1
        elif after < before:
            losses += 1
        else:
            ties += 1
    total = len(before_scores)
    return {"win_rate": wins / total, "loss_rate": losses / total, "tie_rate": ties / total}
