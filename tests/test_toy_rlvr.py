import pytest
import torch

from post_training_rl.toy_rlvr import (
    CharTokenizer,
    ExperimentConfig,
    TinyCausalLM,
    arithmetic_reward,
    generate_completion,
    make_sft_batch,
    run_experiment,
    sequence_logps_for_completions,
)


def call_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as error:
        pytest.skip(str(error))


def test_arithmetic_reward_prefers_exact_answer():
    exact = call_or_skip(arithmetic_reward, "7", 7)
    wrong_numeric = call_or_skip(arithmetic_reward, "8", 7)
    invalid = call_or_skip(arithmetic_reward, "abc", 7)

    assert exact == 1.0
    assert wrong_numeric < exact
    assert invalid < wrong_numeric


def test_sequence_logps_for_completions_shapes_are_valid():
    tokenizer = CharTokenizer()
    model = TinyCausalLM(tokenizer.vocab_size, d_model=8, hidden_size=12)
    prompt = tokenizer.encode_text("1+2=")
    completion = tokenizer.encode_text("3") + [tokenizer.eos_token_id]

    seq_logps, seq_lens = call_or_skip(
        sequence_logps_for_completions,
        model,
        tokenizer,
        [prompt],
        [completion],
        device=torch.device("cpu"),
    )

    assert seq_logps.shape == (1,)
    assert seq_lens.tolist() == [2.0]
    assert torch.isfinite(seq_logps).all().item()


def test_sft_batch_masks_only_answer_tokens():
    tokenizer = CharTokenizer()
    _, labels, loss_mask = make_sft_batch(
        tokenizer,
        [(1, 2)],
        device=torch.device("cpu"),
    )

    assert labels.shape == loss_mask.shape
    assert loss_mask.sum().item() == 2.0


def test_generate_completion_returns_at_least_one_token():
    tokenizer = CharTokenizer()
    model = TinyCausalLM(tokenizer.vocab_size, d_model=8, hidden_size=12)
    prompt = tokenizer.encode_text("1+2=")

    completion = call_or_skip(
        generate_completion,
        model,
        tokenizer,
        prompt,
        max_new_tokens=3,
        temperature=0.0,
        device=torch.device("cpu"),
    )

    assert 1 <= len(completion) <= 3


def test_run_experiment_smoke():
    config = ExperimentConfig(
        seed=1,
        max_operand=3,
        sft_steps=1,
        rl_steps=1,
        batch_size=4,
        rollout_batch_size=4,
        ppo_epochs=1,
        eval_every=1,
        d_model=8,
        hidden_size=12,
        eval_prompts=4,
        pass_k=2,
        max_new_tokens=3,
    )

    result = call_or_skip(run_experiment, config, progress=False)

    assert "initial_eval" in result
    assert "post_sft_eval" in result
    assert "final_eval" in result
    assert 0.0 <= result["final_eval"]["accuracy"] <= 1.0
