import torch

from .utils import compute_sequence_logps


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
    if not prompts:
        raise ValueError("prompts and completions must not be empty")
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")
    rows = []
    prompt_lengths = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")

        completion_ids = tokenizer(
            completion,
            add_special_tokens=False,
        )["input_ids"]

        if not completion_ids:
            raise ValueError("completion must contain at least one token")
        rows.append(prompt_ids + completion_ids)
        prompt_lengths.append(len(prompt_ids))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")

    batch_size = len(rows)
    max_length = max(len(row) for row in rows)
    input_ids = torch.full(
        (batch_size, max_length),
        pad_id,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
        device=device,
    )

    loss_mask = torch.zeros(
        (batch_size, max_length - 1),
        dtype=torch.float32,
        device=device,
    )
    for row_idx, row in enumerate(rows):
        row_length = len(row)

        input_ids[row_idx, :row_length] = torch.tensor(
            row,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row_idx, :row_length] = 1
        completion_start = prompt_lengths[row_idx] - 1
        prediction_end = row_length - 1
        loss_mask[row_idx, completion_start:prediction_end] = 1.0

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    shifted_logits = outputs.logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]

    return compute_sequence_logps(
        shifted_logits,
        shifted_labels,
        loss_mask,
    )
