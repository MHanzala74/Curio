# Curio

**An AI feed curator that learns your taste.**

A multi-agent news feed curator that learns **your** taste using real
reinforcement learning (PPO actor-critic), not just a bandit.

> Status: early development. Built in small steps, one commit at a time.

## Idea

Every day the system collects many candidate items (papers, blog posts,
repos), picks a small slate to show you, and learns from how you react.
Because what we show today shapes your interests tomorrow, this is a
sequential decision problem, which is exactly what RL is for.

## MDP formulation

| Element    | Definition                                                                 |
|------------|----------------------------------------------------------------------------|
| State      | User interest vector, recent history summary, candidate pool features      |
| Action     | Pick one candidate item at a time until the slate is full (slate building) |
| Reward     | Engagement (click, read time) + novelty bonus - redundancy penalty         |
| Transition | The user's interests drift toward what they were shown and liked           |
| Discount   | gamma = 0.9, so long-term satisfaction matters, not just instant clicks    |

## Planned architecture

- **Collector agent**: fetches candidate items from several sources
- **Simulated user**: used to pretrain the policy with thousands of episodes
- **Policy (PPO actor-critic)**: decides which items to show
- **Summarizer agent**: writes short summaries for the chosen items
- **Reflector agent**: writes a weekly natural-language profile of your taste

## Roadmap

- [x] Project scaffold
- [x] Gymnasium environment and simulated user
- [x] PPO implementation (PyTorch) and baselines (random, greedy, bandit)
- [x] Learning curves and multi-seed comparison (5 seeds, mean ± std)
- [ ] Expose click feedback in the observation so the policy can personalize
- [ ] Real feedback pipeline
- [ ] LLM agents (collector, summarizer, reflector)
- [ ] GitHub Actions daily run

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## License

MIT

## Results (simulated user, 5 seeds)

![Learning curve over 5 seeds](docs/multi_seed_curve.png)

Each seed trains a fresh PPO agent and evaluates every method on 300 episodes
with a **different simulated user** than the one used in training:

| Method | Mean episode return (mean ± std over 5 seeds) | CTR |
|---|---|---|
| random | 3.454 ± 0.038 | 0.443 ± 0.007 |
| linucb-bandit | 4.443 ± 0.049 | 0.701 ± 0.011 |
| oracle-greedy | 4.450 ± 0.030 | 0.706 ± 0.007 |
| ppo (ours) | 4.765 ± 0.057 | 0.706 ± 0.012 |

Paired difference (PPO - LinUCB): +0.323 ± 0.044; PPO wins on 5/5 seeds.

Reproduce with:

```bash
PYTHONPATH=src python multi_seed.py --seeds 5
```

### What PPO actually learned (important)

PPO beats the greedy baselines on all 5 seeds, but the reason is **not
personalization**. In the current environment the observation only contains the
items that were shown, not whether the user clicked them, so the policy cannot
infer a user's taste. A diagnostic run confirmed this: the trained policy keeps
a click-through rate of about 0.69 on a user whose interest vector has cosine
similarity -0.5 with the training user.

What it learned instead is a **diversity strategy**: showing items that are
dissimilar to each other lowers the simulated user's boredom and earns the
novelty bonus, something the one-step greedy baselines ignore. That is a real
sequential effect, but it is a user-agnostic one.

**Other caveats**

- The simulated user is simple, so these numbers say little about a real person.
- Oracle-greedy peeks at the true interest and ignores novelty; it is a
  reference point, not a fair competitor.
- Next fix: expose click feedback in the observation so the policy can
  personalize within an episode, then re-run this comparison.

## Running the tests

```bash
PYTHONPATH=src pytest -q
```
