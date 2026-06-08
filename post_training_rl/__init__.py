"""From-scratch post-training RL primitives."""

from .dpo import dpo_loss
from .eval import bootstrap_mean_ci, paired_win_rate, pass_at_k
from .grpo import grpo_clipped_loss, group_normalize_rewards
from .hf_logprobs import compute_hf_completion_logps
from .ppo import ppo_clipped_loss
from .reward_modeling import (
    detect_reward_hacking,
    pairwise_preference_loss,
    reward_accuracy,
)
from .utils import compute_sequence_logps

__all__ = [
    "bootstrap_mean_ci",
    "compute_sequence_logps",
    "compute_hf_completion_logps",
    "detect_reward_hacking",
    "dpo_loss",
    "grpo_clipped_loss",
    "group_normalize_rewards",
    "paired_win_rate",
    "pairwise_preference_loss",
    "pass_at_k",
    "ppo_clipped_loss",
    "reward_accuracy",
]
