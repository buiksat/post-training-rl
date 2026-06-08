def compute_hf_completion_logps(model, tokenizer, prompts, completions, device="cpu"):
    """Compute summed log-probs for Hugging Face model completions.

    Exercise HF-1:
    This is the bridge from `experiments/hf_probe.py` to RL training with a real
    causal LM. For each prompt/completion pair:
    - tokenize prompt + completion
    - shift inputs/labels for causal LM next-token prediction
    - mask out prompt tokens
    - return summed completion log-probs and completion token counts
    """
    raise NotImplementedError("Exercise HF-1: implement HF completion log-probs.")
