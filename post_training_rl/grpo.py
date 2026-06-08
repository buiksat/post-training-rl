def group_normalize_rewards(rewards, group_ids, eps=1e-8):
    """Normalize rewards within each prompt group.

    Exercise 11:
    GRPO-style training samples multiple completions per prompt, then computes
    relative advantages inside each prompt group. Implement:
    advantage_i = (reward_i - group_mean) / (group_std + eps)
    """
    raise NotImplementedError("Exercise 11: implement grouped advantages.")


def grpo_clipped_loss(logp, old_logp, group_advantages, clip_eps=0.2):
    """GRPO-style clipped policy loss without a learned value function.

    Exercise 12:
    This should look similar to PPO's clipped policy loss, but advantages come
    from within-group reward normalization instead of a value model.
    Return loss plus diagnostics: loss, clip_fraction, approx_kl, ratio_mean.
    """
    raise NotImplementedError("Exercise 12: implement GRPO clipped loss.")
