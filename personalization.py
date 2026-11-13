"""Personalization experiment.

Every episode uses a brand-new simulated user (``resample_user_every=1``) and a
longer slate, so a policy can only do well by inferring taste from in-episode
click feedback (or by using a user-agnostic diversity strategy).

Usage (one variant at a time; results are saved per seed under docs/runs/):

    python personalization.py --variant click   --seeds 0 1 2
    python personalization.py --variant noclick --seeds 0 1 2
    python personalization.py --variant baselines --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curio.baselines import LinUCBPolicy, OracleDiversePolicy, OracleGreedyPolicy, RandomPolicy
from curio.evaluate import run_policy, run_ppo
from curio.training import train_ppo

ENV = dict(slate_size=10, resample_user_every=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["click", "noclick", "belief", "baselines"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes-per-update", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--out", type=str, default="docs/runs")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        eval_seed = 1000 + seed
        if args.variant == "baselines":
            runs = {
                "random": run_policy(RandomPolicy(seed=seed), args.eval_episodes, eval_seed, **ENV),
                "linucb (untuned)": run_policy(
                    LinUCBPolicy(8, reset_each_episode=True), args.eval_episodes, eval_seed, **ENV
                ),
                "linucb (tuned)": run_policy(
                    LinUCBPolicy(8, alpha=2.0, reg=0.1, signed_reward=True, reset_each_episode=True),
                    args.eval_episodes, eval_seed, **ENV,
                ),
                "oracle-greedy": run_policy(OracleGreedyPolicy(), args.eval_episodes, eval_seed, **ENV),
                "oracle-diverse": run_policy(OracleDiversePolicy(), args.eval_episodes, eval_seed, **ENV),
            }
            payload = {"results": runs}
        else:
            click = args.variant in ("click", "belief")
            belief = args.variant == "belief"
            kw = dict(click_feedback=click, belief_state=belief, **ENV)
            model, history = train_ppo(
                seed, args.updates, args.episodes_per_update, log_every=50, lr=args.lr, **kw,
            )
            result = run_ppo(model, args.eval_episodes, eval_seed, **kw)
            label = {"noclick": "ppo (no click feedback)", "click": "ppo (with click feedback)",
                     "belief": "ppo (click feedback + belief state)"}[args.variant]
            payload = {
                "results": {label: result},
                "train_curve": [h["train_return"] for h in history],
            }
        (out / f"{args.variant}_seed{seed}.json").write_text(json.dumps(payload, indent=2))
        print(f"{args.variant} seed {seed}: " + ", ".join(
            f"{k}={v['return']:.3f}" for k, v in payload["results"].items()), flush=True)


if __name__ == "__main__":
    main()
