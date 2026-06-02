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
    raise NotImplementedError("Implement DPO loss by hand.")
