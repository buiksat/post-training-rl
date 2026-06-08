import torch


def ppo_clipped_loss(logp, old_logp, advantages, clip_eps=0.2):
    """PPO clipped surrogate objective with diagnostics. logp, old_logp,
    advantages are all shape (B,). Returns (loss scalar, metrics dict).
    metrics should include: loss, clip_fraction, approx_kl, ratio_mean."""
    # Exercise 5:
    # 1. Compute log_ratio = logp - old_logp and ratio = exp(log_ratio).
    # 2. Build unclipped and clipped policy objectives.
    # 3. PPO maximizes the lower objective, so return negative mean as loss.
    # 4. Report loss, clip_fraction, approx_kl, and ratio_mean.
    raise NotImplementedError("Exercise 5: implement PPO clipped loss.")
