import torch

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
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, V)")
    B, T, _ = logits.shape
    if labels.shape != (B, T) or loss_mask.shape != (B, T):
        raise ValueError("labels and loss_mask must have shape (B, T)")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("labels must contain integer token ids")
    if torch.any(loss_mask < 0):
        raise ValueError("loss_mask must be non-negative")
    log_probs = torch.log_softmax(logits, dim=-1)
    chosen_log_probs = torch.gather(
        log_probs, 
        dim=-1, 
        index=labels.unsqueeze(-1)
        ).squeeze(-1)
    masked_log_probs = chosen_log_probs * loss_mask
    seq_logp_sum = torch.sum(masked_log_probs, dim=-1)    
    seq_len = torch.sum(loss_mask, dim=-1)        
    return (seq_logp_sum, seq_len)
