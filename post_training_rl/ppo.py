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
    if logp.shape != old_logp.shape or logp.shape != advantages.shape:
        raise ValueError("logp, old_logp, and advantages must have the same shape")
    if logp.numel() == 0:
        raise ValueError("PPO batch cannot be empty")
    if clip_eps < 0:
        raise ValueError("clip_eps must be non-negative")
    log_ratio = logp - old_logp
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    metrics = {
        "loss": loss.detach(),
        "clip_fraction": ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float().mean().detach(),
        "approx_kl": (old_logp - logp).mean().detach(),
        "ratio_mean": ratio.mean().detach(),
    }
    return loss, metrics
