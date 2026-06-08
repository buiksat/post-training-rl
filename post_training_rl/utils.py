def compute_sequence_logps(logits, labels, loss_mask):
    """Sum of per-token log-probs over a response, plus token counts.
    Args: logits (B,T,V) PRE-ALIGNED so logits[:,t] scores labels[:,t]
    (caller owns the causal shift); labels (B,T); loss_mask (B,T) with
    1.0 for response tokens, 0.0 for prompt/padding.
    Returns: (seq_logp_sum (B,), seq_len (B,))."""
    # Exercise 1:
    # 1. Validate shapes.
    # 2. Convert logits to log-probs with log_softmax.
    # 3. Gather the log-prob assigned to each label token.
    # 4. Apply loss_mask so prompt/padding tokens do not count.
    # 5. Return summed response log-probs and response token counts.
    raise NotImplementedError("Exercise 1: implement response sequence log-probs.")
