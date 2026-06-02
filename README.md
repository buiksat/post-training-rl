from-scratch post-training RL primitives with tests, built as interview prep.

## Roadmap

Only DPO/PPO/utils are scaffolded so far.

- [ ] DPO loss (+NLL)
- [ ] PPO clipped + diagnostics
- [ ] GRPO grouped advantages
- [ ] toy RLVR verifier
- [ ] Pass@K eval
- [ ] bootstrap CIs
- [ ] failure-analysis examples

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/ -q
```
