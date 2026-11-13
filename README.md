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
- [x] Expose click feedback in the observation so the policy can personalize (partial success, see below)
- [ ] Stronger baselines and more seeds for the personalization task
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
- The follow-up experiment below adds click feedback to the observation.
- These results were produced with the code at tag `v0.1-no-click-feedback`.

## Personalization experiment (click feedback)

![Personalization results](docs/personalization.png)

To test real personalization, every episode now uses a **brand-new simulated
user** and a longer slate of 10 items, so a policy cannot memorize one user.
The observation includes summaries of the items clicked and skipped earlier in
the episode (`click_feedback=True`). The "no click feedback" ablation removes
exactly that information.

| Method | Episode return (mean ± std, 3 seeds) | CTR |
|---|---|---|
| random | 6.048 ± 0.029 | 0.401 |
| linucb (per-user) | 6.116 ± 0.080 | 0.445 |
| ppo (no click feedback) | 6.630 ± 0.138 | 0.429 |
| ppo (with click feedback) | 6.866 ± 0.051 | 0.460 |
| oracle-greedy | 7.554 ± 0.088 | 0.597 |
| oracle-diverse | 7.979 ± 0.093 | 0.614 |

Paired difference, PPO with vs. without click feedback: +0.236 ± 0.093 (with-feedback wins on 3/3 seeds).
Paired difference, PPO with click feedback vs. per-user LinUCB: +0.751 ± 0.054 (wins on 3/3 seeds).

Reproduce with:

```bash
PYTHONPATH=src python personalization.py --variant baselines --seeds 0 1 2
PYTHONPATH=src python personalization.py --variant noclick   --seeds 0 1 2
PYTHONPATH=src python personalization.py --variant click     --seeds 0 1 2
python aggregate.py
```

**Reading these results honestly**

- Click feedback helps: PPO improves by about +0.24 return on all 3 seeds, which
  supports the claim that it learns some personalization from in-episode clicks.
- It is only **partial**. PPO closes roughly 40% of the gap between random and
  the oracle-diverse reference, and its CTR (0.46) is far below the oracles (0.60).
- Without click feedback PPO still beats random, which matches the earlier
  finding that part of the gain is a user-agnostic diversity strategy.
- The per-user LinUCB baseline barely beats random. With only 10 interactions per
  user and default hyperparameters it is probably under-tuned, so the PPO vs.
  LinUCB gap should not be over-interpreted. Tuning it is on the roadmap.
- Only 3 seeds, so standard deviations are rough.

## Running the tests

```bash
PYTHONPATH=src pytest -q
```
