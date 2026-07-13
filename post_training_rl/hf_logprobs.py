import torch

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
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")
    rows = []
    prompt_lengths = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        completion_ids = tokenizer(
            completion,
            add_special_tokens=False,
        )["input_ids"]

        if not completion_ids:
            raise ValueError("completion must contain at least one token")
        rows.append(prompt_ids + completion_ids)
        prompt_lengths.append(len(prompt_ids))



    raise NotImplementedError("Exercise HF-1: implement HF completion log-probs.")
