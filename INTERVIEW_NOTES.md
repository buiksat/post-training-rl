# Frontier Post-Training Interview Notes

Use this file after implementing each exercise. The goal is to explain the
mechanics from memory, state the assumptions, and identify what can go wrong.

## Core equations

### Causal LM log-probabilities

For input tokens `x_0, ..., x_T`, the causal model produces logits at position
`t` for `x_{t+1}`. Completion log-probability is:

```text
log p(y | x) = sum_t log p(y_t | x, y_<t)
```

Only completion labels should be included in SFT and policy objectives. Prompt
labels are context, not supervised targets for completion quality.

### PPO

```text
r_t(theta) = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t))
L_clip = min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)
```

The implementation returns `-mean(L_clip)` because optimizers minimize losses.
`approx_kl`, `ratio_mean`, and `clip_fraction` are safety diagnostics, not
optimization targets.

### RLVR with a reference policy

```text
sample_kl = log pi_old(y|x) - log pi_ref(y|x)
rl_reward = verifier_reward - kl_coef * sample_kl
```

The reference is frozen at the post-SFT policy. A verifier can improve measured
task reward while still being exploitable, so always compare verifier reward to
an independently designed task metric.

### DPO

```text
z = beta * ((log pi(y+|x) - log pi(y-|x))
           - (log pi_ref(y+|x) - log pi_ref(y-|x)))
loss = -log sigmoid(z)
```

DPO uses offline preference pairs and does not require online rollout data.
Sequence log-prob normalization is a design choice: summed log-probs preserve
the likelihood-ratio interpretation, while length normalization changes the
preference being optimized.

### GRPO

For completions sampled from the same prompt:

```text
A_i = (r_i - mean_group(r)) / (std_group(r) + eps)
```

This supplies relative advantages without a learned value model. Grouping must
be correct; mixing completions from different prompts creates invalid baselines.

## Metric interpretation

- `accuracy`: greedy task success.
- `pass@K`: probability that at least one of K samples is correct; it measures
  sampled capability, not deterministic deployment quality.
- `approx_kl`: directional estimate of policy movement relative to the old
  policy in PPO diagnostics.
- `clip_fraction`: fraction of examples whose policy ratio is outside the PPO
  trust interval.
- `reward_accuracy`: fraction of preference pairs ranked correctly by a reward
  model; it does not establish calibration or robustness.
- bootstrap CI: uncertainty around an evaluation estimate, not uncertainty in
  the model parameters.

## Failure-analysis prompts

1. Reward increases but task accuracy decreases. What proxy is being exploited?
2. KL rises rapidly. Which knobs and metrics do you inspect before lowering the
   learning rate?
3. PPO clip fraction is near 100%. Is the issue the learning rate, stale data,
   advantage scale, or ratio computation?
4. Pass@4 improves while greedy accuracy falls. Is that useful for the product
   or only for a sampling-based benchmark?
5. A reward model reaches high pairwise accuracy but produces poor downstream
   behavior. Which distribution shift or shortcut could explain it?
6. GRPO groups have zero reward variance. What should the advantages be, and
   what does that imply about learning signal?

## Research-engineering design prompts

- How would you shard rollout generation and learner updates across workers?
- Which tensors must be detached, and which must retain gradients?
- How do you ensure policy and reference tokenization boundaries match?
- Which metrics are logged per-token, per-sequence, per-prompt, and globally?
- How would you make a post-training gain statistically credible?
- What ablations separate SFT quality, verifier quality, KL control, and PPO
  optimization quality?

## Timed practice format

- 30 minutes: implement one loss from its equation and write edge-case tests.
- 30 minutes: explain one algorithm without looking at the code.
- 45 minutes: design a scalable post-training system and its metrics.
- 15 minutes: critique an experiment where reward and task metrics diverge.
