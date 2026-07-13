import torch


def pairwise_preference_loss(chosen_scores, rejected_scores, margin=0.0):
    """Bradley-Terry/logistic preference-model loss.

    Exercise 13:
    Implement the reward-model training loss for preference pairs. The chosen
    response should receive a higher scalar score than the rejected response.
    """
    if chosen_scores.shape != rejected_scores.shape:
        raise ValueError("chosen_scores and rejected_scores must have the same shape")
    if chosen_scores.numel() == 0:
        raise ValueError("preference batch cannot be empty")
    return -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores - margin).mean()


def reward_accuracy(chosen_scores, rejected_scores):
    """Fraction of preference pairs ranked correctly.

    Exercise 14:
    Return the fraction where chosen_score > rejected_score. This is a basic
    reward-model diagnostic and a common interview warmup.
    """
    if chosen_scores.shape != rejected_scores.shape or chosen_scores.numel() == 0:
        raise ValueError("scores must be non-empty and have matching shapes")
    return (chosen_scores > rejected_scores).float().mean().item()


def detect_reward_hacking(reward_scores, task_scores, kl_values):
    """Surface simple reward-hacking warning signals.

    Exercise 15:
    Return a dict of diagnostics that can flag cases like reward increasing
    while task success falls, or KL drifting sharply upward. Keep this simple
    and explainable; the point is failure-analysis thinking.
    """
    reward_scores = [float(x) for x in reward_scores]
    task_scores = [float(x) for x in task_scores]
    kl_values = [float(x) for x in kl_values]
    if not reward_scores or len({len(reward_scores), len(task_scores), len(kl_values)}) != 1:
        raise ValueError("diagnostic series must be non-empty and equally sized")
    reward_delta = reward_scores[-1] - reward_scores[0]
    task_delta = task_scores[-1] - task_scores[0]
    kl_delta = kl_values[-1] - kl_values[0]
    return {
        "reward_up_task_down": reward_delta > 0 and task_delta < 0,
        "kl_increased": kl_delta > 0,
        "reward_delta": reward_delta,
        "task_delta": task_delta,
        "kl_delta": kl_delta,
        "reward_final": reward_scores[-1],
        "task_final": task_scores[-1],
        "kl_final": kl_values[-1],
    }
