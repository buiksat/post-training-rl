def ppo_clipped_loss(logp, old_logp, advantages, clip_eps=0.2):
    """PPO clipped surrogate objective with diagnostics. logp, old_logp,
    advantages are all shape (B,). Returns (loss scalar, metrics dict).
    metrics should include: loss, clip_fraction, approx_kl, ratio_mean."""
    raise NotImplementedError("Implement PPO clipped loss by hand.")
