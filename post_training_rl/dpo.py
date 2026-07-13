import torch


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1, label_smoothing=0.0,
             use_nll=False, chosen_lengths=None, nll_alpha=1.0):
    """DPO loss (Rafailov et al. 2023) with optional NLL-on-chosen
    regularizer (use_nll). All *_logps are SUMMED sequence log-probs,
    shape (B,). Returns (loss scalar, metrics dict). metrics should
    include: loss, dpo_loss, nll_loss, reward_chosen, reward_rejected,
    reward_margin, reward_accuracy. chosen_lengths is required when
    use_nll=True; raise ValueError otherwise."""
    # Exercise 7:
    # 1. Compute policy and reference chosen-vs-rejected log-ratios.
    # 2. Form beta * (policy_logratio - reference_logratio).
    # 3. Apply the DPO logistic loss, with optional label smoothing.
    # 4. Optionally add chosen-response NLL regularization.
    # 5. Return reward diagnostics from beta * (policy_logp - reference_logp).
    tensors = [policy_chosen_logps, policy_rejected_logps,
               reference_chosen_logps, reference_rejected_logps]
    if any(t.shape != tensors[0].shape for t in tensors[1:]):
        raise ValueError("all log-prob tensors must have the same shape")
    if beta <= 0:
        raise ValueError("beta must be positive")
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    smoothing = float(label_smoothing)
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("label_smoothing must be in [0, 1]")
    dpo = -((1.0 - smoothing) * torch.nn.functional.logsigmoid(logits)
            + smoothing * torch.nn.functional.logsigmoid(-logits)).mean()
    if use_nll:
        if chosen_lengths is None:
            raise ValueError("chosen_lengths is required when use_nll=True")
        if torch.any(chosen_lengths <= 0):
            raise ValueError("chosen_lengths must be positive")
        nll = (-policy_chosen_logps / chosen_lengths).mean()
    else:
        nll = logits.new_zeros(())
    loss = dpo + nll_alpha * nll if use_nll else dpo
    chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps)
    margin = chosen_reward - rejected_reward
    metrics = {
        "loss": loss.detach(), "dpo_loss": dpo.detach(), "nll_loss": nll.detach(),
        "reward_chosen": chosen_reward.mean().detach(),
        "reward_rejected": rejected_reward.mean().detach(),
        "reward_margin": margin.mean().detach(),
        "reward_accuracy": (margin > 0).float().mean().detach(),
    }
    return loss, metrics
