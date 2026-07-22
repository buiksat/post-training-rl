import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .ppo import ppo_clipped_loss
from .utils import compute_sequence_logps


class CharTokenizer:
    """Tiny character tokenizer for arithmetic prompts and numeric answers."""

    def __init__(self):
        self.tokens = ["<pad>", "<eos>"] + list("0123456789+=")
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.eos_token_id = self.token_to_id["<eos>"]

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode_text(self, text):
        return [self.token_to_id[ch] for ch in text]

    def decode_completion(self, token_ids):
        chars = []
        for token_id in token_ids:
            if token_id == self.eos_token_id:
                break
            if token_id == self.pad_token_id:
                continue
            chars.append(self.id_to_token[int(token_id)])
        return "".join(chars)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, hidden_size=96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, hidden=None):
        embedded = self.embedding(input_ids)
        output, hidden = self.rnn(embedded, hidden)
        return self.lm_head(output), hidden


@dataclass
class ExperimentConfig:
    seed: int = 0
    max_operand: int = 20
    sft_steps: int = 200
    rl_steps: int = 100
    batch_size: int = 64
    rollout_batch_size: int = 32
    ppo_epochs: int = 2
    eval_every: int = 25
    d_model: int = 64
    hidden_size: int = 96
    lr: float = 1e-3
    clip_eps: float = 0.2
    kl_coef: float = 0.2
    temperature: float = 1.0
    max_new_tokens: int = 4
    eval_prompts: int = 64
    pass_k: int = 4
    device: str = "cpu"


def format_prompt(a, b):
    return f"{a}+{b}="


def format_answer(a, b):
    return str(a + b)


def sample_addition_pairs(rng, batch_size, max_operand):
    return [
        (rng.randint(0, max_operand), rng.randint(0, max_operand))
        for _ in range(batch_size)
    ]


def _pad_rows(rows, pad_id, device, dtype=torch.long):
    max_len = max(len(row) for row in rows)
    tensor = torch.full((len(rows), max_len), pad_id, dtype=dtype, device=device)
    for row_idx, row in enumerate(rows):
        if row:
            tensor[row_idx, : len(row)] = torch.tensor(row, dtype=dtype, device=device)
    return tensor


def make_lm_batch(tokenizer, prompt_ids, completion_ids, device):
    inputs = []
    labels = []
    masks = []
    for prompt, completion in zip(prompt_ids, completion_ids):
        if not completion:
            raise ValueError("completion must contain at least one token")
        full = prompt + completion
        inputs.append(full[:-1])
        labels.append(full[1:])
        masks.append([1.0 if idx + 1 >= len(prompt) else 0.0 for idx in range(len(full) - 1)])

    input_ids = _pad_rows(inputs, tokenizer.pad_token_id, device)
    label_ids = _pad_rows(labels, tokenizer.pad_token_id, device)
    loss_mask = _pad_rows(masks, 0.0, device, dtype=torch.float32)
    return input_ids, label_ids, loss_mask


def make_sft_batch(tokenizer, pairs, device):
    prompts = []
    completions = []
    for a, b in pairs:
        prompts.append(tokenizer.encode_text(format_prompt(a, b)))
        answer = tokenizer.encode_text(format_answer(a, b)) + [tokenizer.eos_token_id]
        completions.append(answer)
    return make_lm_batch(tokenizer, prompts, completions, device)


def sequence_logps_for_completions(model, tokenizer, prompt_ids, completion_ids, device):
    input_ids, labels, loss_mask = make_lm_batch(tokenizer, prompt_ids, completion_ids, device)
    logits, _ = model(input_ids)
    return compute_sequence_logps(logits, labels, loss_mask)


def sft_loss(model, tokenizer, pairs, device):
    # Exercise 2:
    # 1. Build the supervised batch with make_sft_batch.
    input_ids, labels, loss_mask = make_sft_batch(tokenizer, pairs, device)
    # 2. Run the model to get logits.
    logits, _ = model(input_ids)
    # 3. Use compute_sequence_logps over answer tokens only.
    seq_logp_sum, seq_len = compute_sequence_logps(logits, labels, loss_mask)
    # 4. Return mean negative per-token log-prob.    
    return (- (seq_logp_sum / seq_len.clamp(min=1))).mean()
    


def generate_completion(model, tokenizer, prompt_ids, max_new_tokens, temperature, device):
    # Exercise 3:
    # 1. Run the prompt through the model.
    completion = []
    with torch.no_grad():
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )
        logits, hidden = model(input_ids)
        next_logits = logits[0, -1, :]
        # 2. Repeatedly choose the next token by greedy decode or temperature sampling.
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_token = int(torch.argmax(next_logits))
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1))
        # 3. Feed each sampled token back into the model.
            completion.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break 
        # 4. Stop on eos_token_id or max_new_tokens.
            step_input = torch.tensor(
                    [[next_token]],
                    dtype=torch.long,
                    device=device
                ) 
            logits, hidden = model(step_input, hidden)
            next_logits = logits[0, -1, :]
    return completion


def arithmetic_reward(answer_text, target):
    # Exercise 4:
    # Return a scalar reward for the generated answer.
    # Suggested first version:
    # - exact string match with target: 1.0
    # - numeric but wrong: small/partial reward or penalty
    # - non-numeric: negative reward
    raise NotImplementedError("Exercise 4: implement verifier reward.")


def is_correct_answer(answer_text, target):
    return answer_text == str(target)


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _clean_metrics(metrics):
    return {key: _to_float(value) for key, value in metrics.items()}


def evaluate_policy(
    model,
    tokenizer,
    max_operand,
    eval_prompts,
    pass_k,
    max_new_tokens,
    temperature,
    seed,
    device,
):
    rng = random.Random(seed)
    pairs = sample_addition_pairs(rng, eval_prompts, max_operand)
    greedy_hits = 0
    pass_hits = 0
    greedy_rewards = []

    for a, b in pairs:
        prompt = tokenizer.encode_text(format_prompt(a, b))
        target = a + b
        greedy_completion = generate_completion(
            model, tokenizer, prompt, max_new_tokens, temperature=0.0, device=device
        )
        greedy_text = tokenizer.decode_completion(greedy_completion)
        greedy_hits += int(is_correct_answer(greedy_text, target))
        greedy_rewards.append(arithmetic_reward(greedy_text, target))

        any_hit = False
        for _ in range(pass_k):
            sampled_completion = generate_completion(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                temperature=temperature,
                device=device,
            )
            sampled_text = tokenizer.decode_completion(sampled_completion)
            any_hit = any_hit or is_correct_answer(sampled_text, target)
        pass_hits += int(any_hit)

    return {
        "accuracy": greedy_hits / eval_prompts,
        f"pass_at_{pass_k}": pass_hits / eval_prompts,
        "mean_reward": sum(greedy_rewards) / eval_prompts,
    }


def collect_rollouts(model, reference_model, tokenizer, pairs, config, device):
    # Exercise 6:
    # 1. Encode prompts and generate one completion per prompt.
    # 2. Decode completions and score them with arithmetic_reward.
    # 3. Store old policy log-probs and frozen reference log-probs.
    # 4. Compute KL-penalized rewards: verifier_reward - kl_coef * sample_kl.
    # 5. Normalize rewards into advantages.
    # 6. Return the rollout dict expected by run_experiment.
    raise NotImplementedError("Exercise 6: implement rollout collection.")


def run_experiment(config=None, progress=True):
    config = config or ExperimentConfig()
    device = torch.device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    tokenizer = CharTokenizer()
    rng = random.Random(config.seed)
    model = TinyCausalLM(
        tokenizer.vocab_size,
        d_model=config.d_model,
        hidden_size=config.hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    def emit(event):
        if progress:
            print(json.dumps(event, sort_keys=True), flush=True)

    emit({"phase": "config", **asdict(config)})
    initial_eval = evaluate_policy(
        model,
        tokenizer,
        config.max_operand,
        config.eval_prompts,
        config.pass_k,
        config.max_new_tokens,
        config.temperature,
        seed=config.seed + 1,
        device=device,
    )
    emit({"phase": "initial_eval", **initial_eval})

    last_sft_loss = None
    for step in range(1, config.sft_steps + 1):
        pairs = sample_addition_pairs(rng, config.batch_size, config.max_operand)
        loss = sft_loss(model, tokenizer, pairs, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_sft_loss = _to_float(loss)
        if step == 1 or step == config.sft_steps or step % config.eval_every == 0:
            emit({"phase": "sft", "step": step, "loss": last_sft_loss})

    reference_model = copy.deepcopy(model).to(device)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    post_sft_eval = evaluate_policy(
        model,
        tokenizer,
        config.max_operand,
        config.eval_prompts,
        config.pass_k,
        config.max_new_tokens,
        config.temperature,
        seed=config.seed + 2,
        device=device,
    )
    emit({"phase": "post_sft_eval", **post_sft_eval})

    last_rl_metrics = {}
    for step in range(1, config.rl_steps + 1):
        pairs = sample_addition_pairs(rng, config.rollout_batch_size, config.max_operand)
        rollouts = collect_rollouts(model, reference_model, tokenizer, pairs, config, device)

        epoch_metrics = {}
        for _ in range(config.ppo_epochs):
            logps, _ = sequence_logps_for_completions(
                model,
                tokenizer,
                rollouts["prompt_ids"],
                rollouts["completion_ids"],
                device,
            )
            loss, ppo_metrics = ppo_clipped_loss(
                logps,
                rollouts["old_logps"],
                rollouts["advantages"],
                clip_eps=config.clip_eps,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_metrics = _clean_metrics(ppo_metrics)

        exact_rate = sum(
            is_correct_answer(answer, target)
            for answer, target in zip(rollouts["answers"], rollouts["targets"])
        ) / len(rollouts["targets"])
        last_rl_metrics = {
            **epoch_metrics,
            "rollout_exact": exact_rate,
            "verifier_reward": _to_float(rollouts["verifier_rewards"].mean()),
            "rl_reward": _to_float(rollouts["rewards"].mean()),
            "sample_kl": _to_float(
                (rollouts["old_logps"] - rollouts["reference_logps"]).mean()
            ),
        }
        if step == 1 or step == config.rl_steps or step % config.eval_every == 0:
            emit({"phase": "rl", "step": step, **last_rl_metrics})

    final_eval = evaluate_policy(
        model,
        tokenizer,
        config.max_operand,
        config.eval_prompts,
        config.pass_k,
        config.max_new_tokens,
        config.temperature,
        seed=config.seed + 3,
        device=device,
    )
    emit({"phase": "final_eval", **final_eval})

    return {
        "config": asdict(config),
        "initial_eval": initial_eval,
        "last_sft_loss": last_sft_loss,
        "post_sft_eval": post_sft_eval,
        "last_rl_metrics": last_rl_metrics,
        "final_eval": final_eval,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Toy RLVR post-training experiment.")
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-test run.")
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--max-operand", type=int, default=ExperimentConfig.max_operand)
    parser.add_argument("--sft-steps", type=int, default=ExperimentConfig.sft_steps)
    parser.add_argument("--rl-steps", type=int, default=ExperimentConfig.rl_steps)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument(
        "--rollout-batch-size", type=int, default=ExperimentConfig.rollout_batch_size
    )
    parser.add_argument("--ppo-epochs", type=int, default=ExperimentConfig.ppo_epochs)
    parser.add_argument("--eval-every", type=int, default=ExperimentConfig.eval_every)
    parser.add_argument("--d-model", type=int, default=ExperimentConfig.d_model)
    parser.add_argument("--hidden-size", type=int, default=ExperimentConfig.hidden_size)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    parser.add_argument("--clip-eps", type=float, default=ExperimentConfig.clip_eps)
    parser.add_argument("--kl-coef", type=float, default=ExperimentConfig.kl_coef)
    parser.add_argument("--temperature", type=float, default=ExperimentConfig.temperature)
    parser.add_argument(
        "--max-new-tokens", type=int, default=ExperimentConfig.max_new_tokens
    )
    parser.add_argument("--eval-prompts", type=int, default=ExperimentConfig.eval_prompts)
    parser.add_argument("--pass-k", type=int, default=ExperimentConfig.pass_k)
    parser.add_argument("--device", default=ExperimentConfig.device)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        max_operand=args.max_operand,
        sft_steps=args.sft_steps,
        rl_steps=args.rl_steps,
        batch_size=args.batch_size,
        rollout_batch_size=args.rollout_batch_size,
        ppo_epochs=args.ppo_epochs,
        eval_every=args.eval_every,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        lr=args.lr,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        eval_prompts=args.eval_prompts,
        pass_k=args.pass_k,
        device=args.device,
    )
    if args.quick:
        config.max_operand = 10
        config.sft_steps = 100
        config.rl_steps = 20
        config.batch_size = 32
        config.rollout_batch_size = 16
        config.eval_prompts = 16
        config.eval_every = 20
        config.d_model = 32
        config.hidden_size = 48
    run_experiment(config, progress=True)


if __name__ == "__main__":
    main()
