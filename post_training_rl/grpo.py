import torch


def group_normalize_rewards(rewards, group_ids, eps=1e-8):
    """Normalize rewards within each prompt group.

    Exercise 11:
    GRPO-style training samples multiple completions per prompt, then computes
    relative advantages inside each prompt group. Implement:
    advantage_i = (reward_i - group_mean) / (group_std + eps)
    """
    rewards = torch.as_tensor(rewards)
    group_ids = torch.as_tensor(group_ids, device=rewards.device)
    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must be one-dimensional and aligned")
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    for group in torch.unique(group_ids):
        mask = group_ids == group
        values = rewards[mask].float()
        mean = values.mean()
        std = values.std(unbiased=False)
        advantages[mask] = (values - mean) / (std + eps)
    return advantages


def grpo_clipped_loss(logp, old_logp, group_advantages, clip_eps=0.2):
    """GRPO-style clipped policy loss without a learned value function.

    Exercise 12:
    This should look similar to PPO's clipped policy loss, but advantages come
    from within-group reward normalization instead of a value model.
    Return loss plus diagnostics: loss, clip_fraction, approx_kl, ratio_mean.
    """
    from .ppo import ppo_clipped_loss
    return ppo_clipped_loss(logp, old_logp, group_advantages, clip_eps)
