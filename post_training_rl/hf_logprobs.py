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
        raise ValueError("prompts and completions must have equal length")
    if not prompts:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    rows = []
    prompt_lengths = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if not completion_ids:
            raise ValueError("each completion must contain at least one token")
        rows.append(torch.tensor(prompt_ids + completion_ids, dtype=torch.long))
        prompt_lengths.append(len(prompt_ids))

    max_len = max(row.numel() for row in rows)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, : row.numel()] = row
        attention_mask[index, : row.numel()] = 1
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        labels = input_ids[:, 1:]
        token_logps = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

    sums = []
    lengths = []
    for index, prompt_length in enumerate(prompt_lengths):
        response_start = max(prompt_length - 1, 0)
        response_end = rows[index].numel() - 1
        selected = token_logps[index, response_start:response_end]
        sums.append(selected.sum())
        lengths.append(float(selected.numel()))
    if model_was_training:
        model.train()
    return torch.stack(sums), torch.tensor(lengths, device=device)
