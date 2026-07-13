From-scratch post-training RL practice repo, built as interview prep.

The repo is intentionally an exercise scaffold. The boring wiring is present,
but the important post-training pieces are left as TODO placeholders for you to
implement step by step.

You will build the main post-training moving parts:

- Hugging Face causal-LM probing
- sequence log-prob aggregation over response tokens
- SFT warmup
- frozen reference policy
- sampled rollouts
- verifier rewards
- KL-penalized rewards
- PPO clipped updates
- accuracy and pass@K evaluation
- preference optimization with DPO
- GRPO-style grouped advantages
- reward-model training and diagnostics
- uncertainty estimates for evals
- failure analysis for reward hacking and KL drift

## Roadmap

- [x] Hugging Face model probe
- [x] Hugging Face completion log-probs
- [x] response sequence log-probs
- [x] SFT loss
- [x] autoregressive generation
- [x] verifier reward
- [x] PPO clipped loss + diagnostics
- [x] rollout collection with KL-penalized rewards
- [x] DPO loss (+ optional NLL)
- [x] pass@K and bootstrap CIs
- [x] GRPO grouped advantages
- [x] reward-model pairwise loss
- [x] reward-hacking and KL-drift diagnostics
- [x] failure-analysis examples

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/ -q
```

The exercise implementations and tests are now complete. Use the tests as
contracts: when extending an exercise, add an edge-case test before changing
its behavior.

## Exercise Order

### Step 0: Hugging Face Probe

You already have this in `experiments/hf_probe.py`.

```bash
PYTHONPATH=. python experiments/hf_probe.py
```

What to understand before moving on:

- how a tokenizer turns prompts into token IDs
- how `model.generate(...)` produces continuation tokens
- why rewards should score only the generated response, not the prompt

### Step HF-1: Hugging Face Completion Log-Probs

Implement:

```text
post_training_rl/hf_logprobs.py::compute_hf_completion_logps
```

This is a critical bridge for frontier-lab-style work. You should be able to
explain how to compute log-probs for only the completion tokens of a
prompt/completion pair using a causal LM.

### Step 1: Response Log-Probs

Implement:

```text
post_training_rl/utils.py::compute_sequence_logps
```

Run:

```bash
PYTHONPATH=. python -m pytest tests/test_dpo.py::test_compute_sequence_logps_shapes_finite_and_fully_masked_zero -q
```

### Step 2: SFT Loss

Implement:

```text
post_training_rl/toy_rlvr.py::sft_loss
```

This should use `make_sft_batch(...)`, model logits, and
`compute_sequence_logps(...)`.

### Step 3: Autoregressive Generation

Implement:

```text
post_training_rl/toy_rlvr.py::generate_completion
```

Run:

```bash
PYTHONPATH=. python -m pytest tests/test_toy_rlvr.py::test_generate_completion_returns_at_least_one_token -q
```

### Step 4: Verifier Reward

Implement:

```text
post_training_rl/toy_rlvr.py::arithmetic_reward
```

Run:

```bash
PYTHONPATH=. python -m pytest tests/test_toy_rlvr.py::test_arithmetic_reward_prefers_exact_answer -q
```

### Step 5: PPO Loss

Implement:

```text
post_training_rl/ppo.py::ppo_clipped_loss
```

Run:

```bash
PYTHONPATH=. python -m pytest tests/test_ppo.py -q
```

### Step 6: Rollout Collection

Implement:

```text
post_training_rl/toy_rlvr.py::collect_rollouts
```

This is where the RLVR loop comes together:

- sample completions
- compute verifier rewards
- compute policy/reference log-probs
- subtract KL penalty
- normalize advantages

### Step 7: DPO Loss

Implement:

```text
post_training_rl/dpo.py::dpo_loss
```

Run:

```bash
PYTHONPATH=. python -m pytest tests/test_dpo.py -q
```

### Step 8: Evaluation Metrics

Implement:

```text
post_training_rl/eval.py::pass_at_k
post_training_rl/eval.py::bootstrap_mean_ci
post_training_rl/eval.py::paired_win_rate
```

Run:

```bash
PYTHONPATH=. python -m pytest tests/test_eval.py -q
```

### Step 9: GRPO

Implement:

```text
post_training_rl/grpo.py::group_normalize_rewards
post_training_rl/grpo.py::grpo_clipped_loss
```

Run:

```bash
PYTHONPATH=. python -m pytest tests/test_grpo.py -q
```

### Step 10: Reward Modeling

Implement:

```text
post_training_rl/reward_modeling.py::pairwise_preference_loss
post_training_rl/reward_modeling.py::reward_accuracy
post_training_rl/reward_modeling.py::detect_reward_hacking
```

Run:

```bash
PYTHONPATH=. python -m pytest tests/test_reward_modeling.py -q
```

## Interview Coverage Map

Use this repo to practice explaining and implementing:

- **Tokenization and causal LM mechanics**: prompt vs completion tokens,
  causal shift, label masking, response-only loss.
- **SFT**: completion-only supervised fine-tuning, per-token NLL, why SFT is
  often the reference policy before RL.
- **RLVR/RLHF loop**: rollout sampling, scalar rewards, frozen reference model,
  KL penalty, advantage normalization, policy update.
- **PPO**: probability ratios, clipping, approximate KL, clip fraction,
  why on-policy data matters.
- **GRPO**: multiple completions per prompt, within-group reward normalization,
  no learned value model.
- **DPO**: chosen/rejected log-ratios, reference correction, beta, label
  smoothing, relation to preference optimization without online rollouts.
- **Reward modeling**: pairwise preference loss, reward accuracy, calibration
  mindset, reward hacking risks.
- **Evaluation**: exact match, pass@K, paired comparisons, bootstrap CIs,
  benchmark overfitting, contamination checks.
- **Failure analysis**: reward up but task metric down, KL spikes, mode collapse,
  verbosity hacks, format hacks, invalid reward proxies.
- **Systems/research engineering**: deterministic seeds, reproducible runs,
  metric logging, small smoke tests, scaling path from toy model to HF model.

## Run the Toy RL Pipeline

After Steps 1-6 are implemented, run the toy character-model pipeline:

```bash
PYTHONPATH=. python -m post_training_rl.toy_rlvr --quick
```

Longer CPU run:

```bash
PYTHONPATH=. python -m post_training_rl.toy_rlvr \
  --sft-steps 200 \
  --rl-steps 100 \
  --eval-prompts 64 \
  --pass-k 4
```

The script prints JSON lines:

- `initial_eval`: random policy before SFT
- `sft`: supervised warmup loss
- `post_sft_eval`: accuracy before RL
- `rl`: PPO diagnostics plus rollout reward and sample KL
- `final_eval`: post-RL accuracy and pass@K

For the interview framing, see `INTERVIEW_NOTES.md`. It contains the key
equations, metric interpretations, failure modes, and timed prompts to practice
after implementing each exercise.

For hands-on experiments, vary:

- `--kl-coef`: stronger/weaker pull toward the SFT reference
- `--temperature`: rollout exploration
- `--clip-eps`: PPO clipping strength
- `--max-operand`: task difficulty
- `--sft-steps` vs `--rl-steps`: amount of warmup vs RL
