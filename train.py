"""Train PPO on the simulated feed environment and compare with baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from curio.baselines import LinUCBPolicy, OracleGreedyPolicy, RandomPolicy
from curio.env import FeedEnv
from curio.evaluate import run_policy, run_ppo
from curio.training import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--episodes-per-update", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="docs")
    args = parser.parse_args()

    env = FeedEnv(seed=args.seed)
    model, history = train_ppo(args.seed, args.updates, args.episodes_per_update, log_every=10)

    # Final evaluation on a fresh environment, same seed for every method.
    eval_seed, n_eval = 123, 300
    results = {
        "random": run_policy(RandomPolicy(seed=1), n_eval, eval_seed),
        "linucb-bandit": run_policy(LinUCBPolicy(env.dim), n_eval, eval_seed),
        "oracle-greedy": run_policy(OracleGreedyPolicy(), n_eval, eval_seed),
        "ppo (ours)": run_ppo(model, n_eval, eval_seed),
    }

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2))

    lines = ["| Method | Mean episode return | Click-through rate |", "|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['return']:.3f} | {r['ctr']:.3f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = [h["update"] for h in history]
    y = np.array([h["train_return"] for h in history])
    smooth = np.convolve(y, np.ones(5) / 5, mode="valid")
    ax[0].plot(x, y, alpha=0.3, label="per update")
    ax[0].plot(x[4:], smooth, label="5-update average")
    ax[0].axhline(results["random"]["return"], color="gray", ls="--", label="random")
    ax[0].set_xlabel("PPO update"); ax[0].set_ylabel("Mean episode return")
    ax[0].set_title("Learning curve"); ax[0].legend()
    ax[1].plot(x, [h["entropy"] for h in history], color="tab:orange")
    ax[1].set_xlabel("PPO update"); ax[1].set_ylabel("Policy entropy")
    ax[1].set_title("Exploration (entropy)")
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=140)


if __name__ == "__main__":
    main()
