from types import SimpleNamespace

import pytest
import torch

from post_training_rl.hf_logprobs import compute_hf_completion_logps


class TinyTokenizer:
    pad_token_id = None
    eos_token_id = 0
    token_to_id = {"a": 2, "b": 3, "c": 4, "d": 5}

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [self.token_to_id[character] for character in text]}


class TinyCausalLM(torch.nn.Module):
    vocab_size = 8

    def forward(self, input_ids, attention_mask):
        self.last_attention_mask = attention_mask.detach().clone()
        logits = torch.zeros(
            (*input_ids.shape, self.vocab_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        preferred_tokens = (input_ids + 1) % self.vocab_size
        logits.scatter_(dim=-1, index=preferred_tokens.unsqueeze(-1), value=4.0)
        return SimpleNamespace(logits=logits)


def test_hf_completion_logps_masks_prompt_and_padding():
    model = TinyCausalLM()
    tokenizer = TinyTokenizer()

    sequence_logps, token_counts = compute_hf_completion_logps(
        model,
        tokenizer,
        prompts=["ab", "a"],
        completions=["cd", "d"],
    )

    log_normalizer = torch.log(torch.exp(torch.tensor(4.0)) + model.vocab_size - 1)
    preferred_logp = 4.0 - log_normalizer
    other_logp = -log_normalizer
    expected_logps = torch.stack((2 * preferred_logp, other_logp))

    assert torch.allclose(sequence_logps, expected_logps)
    assert token_counts.tolist() == [2.0, 1.0]
    assert model.last_attention_mask.tolist() == [
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ]


@pytest.mark.parametrize(
    ("prompts", "completions", "message"),
    [
        ([], [], "must not be empty"),
        (["a"], [], "same length"),
        ([""], ["c"], "prompt must contain"),
        (["a"], [""], "completion must contain"),
    ],
)
def test_hf_completion_logps_validates_inputs(prompts, completions, message):
    with pytest.raises(ValueError, match=message):
        compute_hf_completion_logps(
            TinyCausalLM(),
            TinyTokenizer(),
            prompts,
            completions,
        )
