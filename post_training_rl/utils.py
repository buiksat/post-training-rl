def compute_sequence_logps(logits, labels, loss_mask):
    """Sum of per-token log-probs over a response, plus token counts.
    Args: logits (B,T,V) PRE-ALIGNED so logits[:,t] scores labels[:,t]
    (caller owns the causal shift); labels (B,T); loss_mask (B,T) with
    1.0 for response tokens, 0.0 for prompt/padding.
    Returns: (seq_logp_sum (B,), seq_len (B,))."""
    raise NotImplementedError("Implement sequence log-prob aggregation by hand.")
