import torch

from post_training_rl.hf_logprobs import compute_hf_completion_logps


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) % 7 + 1 for char in text]}


class FakeModel(torch.nn.Module):
    def __init__(self, vocab_size=16):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        logits = torch.zeros(
            *input_ids.shape, self.vocab_size, device=input_ids.device
        )
        logits.scatter_(-1, input_ids.unsqueeze(-1).clamp_max(self.vocab_size - 1), 1.0)
        return type("Output", (), {"logits": logits})()


def test_hf_completion_logps_scores_only_completion_tokens():
    logps, lengths = compute_hf_completion_logps(
        FakeModel(), FakeTokenizer(), ["ab", "abc"], ["xy", "z"]
    )

    assert logps.shape == (2,)
    assert lengths.tolist() == [2.0, 1.0]
    assert torch.isfinite(logps).all().item()
