"""Long-horizon experiment: multi-day episodes with fatigue and churn.

    python long_horizon.py --variant baselines --seeds 0 1 2 3 4
    python long_horizon.py --variant ppo99 --seeds 0 1 2      # PPO, gamma = 0.99
    python long_horizon.py --variant ppo90 --seeds 0 1 2      # PPO, gamma = 0.90

Results are saved per seed under docs/runs_long/.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from curio.baselines import LinUCBPolicy, OracleFatiguePolicy, RandomPolicy
from curio.env_longterm import LongTermFeedEnv
from curio.evaluate import run_policy, run_ppo
from curio.training import train_ppo

VAL_SEEDS = [100, 101, 102]


def tune_linucb() -> dict:
    """Pick LinUCB hyperparameters on validation seeds (never the eval seeds)."""
    best = (-1.0, None)
    for alpha, reg in itertools.product([1.0, 2.0, 3.0, 5.0, 8.0, 12.0], [0.03, 0.1, 0.3]):
        score = np.mean([
            run_policy(LinUCBPolicy(8, alpha=alpha, reg=reg, signed_reward=True, reset_each_episode=True),
                       150, 2000 + s, env_cls=LongTermFeedEnv)["return"]
            for s in VAL_SEEDS
        ])
        if score > best[0]:
            best = (float(score), dict(alpha=alpha, reg=reg))
    print("tuned LinUCB:", best, flush=True)
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baselines", "ppo99", "ppo90"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes-per-update", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--out", type=str, default="docs/runs_long")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(env_cls=LongTermFeedEnv)
    lin_params = tune_linucb() if args.variant == "baselines" else None

    for seed in args.seeds:
        eval_seed = 1000 + seed
        if args.variant == "baselines":
            runs = {
                "random": run_policy(RandomPolicy(seed), args.eval_episodes, eval_seed, **kw),
                "linucb (tuned)": run_policy(
                    LinUCBPolicy(8, signed_reward=True, reset_each_episode=True, **lin_params),
                    args.eval_episodes, eval_seed, **kw),
                "oracle-myopic": run_policy(OracleFatiguePolicy(0.0), args.eval_episodes, eval_seed, **kw),
                "oracle-lookahead": run_policy(OracleFatiguePolicy(3.0), args.eval_episodes, eval_seed, **kw),
            }
            payload = {"results": runs, "linucb_params": lin_params}
        else:
            gamma = 0.99 if args.variant == "ppo99" else 0.9
            model, history = train_ppo(seed, args.updates, args.episodes_per_update, log_every=50,
                                       lr=args.lr, gamma=gamma, **kw)
            label = f"ppo (gamma={gamma})"
            payload = {"results": {label: run_ppo(model, args.eval_episodes, eval_seed, **kw)},
                       "train_curve": [h["train_return"] for h in history]}
        (out / f"{args.variant}_seed{seed}.json").write_text(json.dumps(payload, indent=2))
        print(f"{args.variant} seed {seed}: " + ", ".join(
            f"{k}={v['return']:.2f} ({v['days']:.1f}d)" for k, v in payload["results"].items()), flush=True)


if __name__ == "__main__":
    main()
