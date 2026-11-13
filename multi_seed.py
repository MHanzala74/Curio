"""Train PPO on several seeds and compare against baselines with mean +/- std."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from curio.baselines import LinUCBPolicy, OracleGreedyPolicy, RandomPolicy
from curio.evaluate import run_policy, run_ppo
from curio.training import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--episodes-per-update", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--out", type=str, default="docs")
    args = parser.parse_args()

    methods = ["random", "linucb-bandit", "oracle-greedy", "ppo (ours)"]
    scores = {m: [] for m in methods}
    ctrs = {m: [] for m in methods}
    curves = []

    for seed in range(args.seeds):
        model, history = train_ppo(seed, args.updates, args.episodes_per_update, log_every=40)
        curves.append([h["train_return"] for h in history])
        eval_seed = 1000 + seed  # a different user and pools than training
        dim = 8
        runs = {
            "random": run_policy(RandomPolicy(seed=seed), args.eval_episodes, eval_seed),
            "linucb-bandit": run_policy(LinUCBPolicy(dim), args.eval_episodes, eval_seed),
            "oracle-greedy": run_policy(OracleGreedyPolicy(), args.eval_episodes, eval_seed),
            "ppo (ours)": run_ppo(model, args.eval_episodes, eval_seed),
        }
        for m, r in runs.items():
            scores[m].append(r["return"])
            ctrs[m].append(r["ctr"])
        print(f"seed {seed} done: " + ", ".join(f"{m}={r['return']:.3f}" for m, r in runs.items()), flush=True)

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    def ms(x):
        return float(np.mean(x)), float(np.std(x, ddof=1))

    lines = [f"| Method | Mean episode return (mean ± std over {args.seeds} seeds) | CTR |", "|---|---|---|"]
    summary = {}
    for m in methods:
        mr, sr = ms(scores[m])
        mc, sc = ms(ctrs[m])
        summary[m] = {"return_mean": mr, "return_std": sr, "ctr_mean": mc, "ctr_std": sc, "per_seed": scores[m]}
        lines.append(f"| {m} | {mr:.3f} ± {sr:.3f} | {mc:.3f} ± {sc:.3f} |")

    diff = np.array(scores["ppo (ours)"]) - np.array(scores["linucb-bandit"])
    dm, ds = ms(diff)
    wins = int((diff > 0).sum())
    lines.append("")
    lines.append(f"Paired difference (PPO - LinUCB): {dm:+.3f} ± {ds:.3f}; PPO wins on {wins}/{args.seeds} seeds.")
    (out / "multi_seed_results.md").write_text("\n".join(lines) + "\n")
    (out / "multi_seed_results.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))

    arr = np.array(curves)
    x = np.arange(1, arr.shape[1] + 1)
    mean, std = arr.mean(0), arr.std(0, ddof=1)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(x, mean, label="PPO (mean of seeds)")
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, label="± 1 std")
    ax.axhline(np.mean(scores["random"]), color="gray", ls="--", label="random")
    ax.axhline(np.mean(scores["linucb-bandit"]), color="tab:green", ls=":", label="LinUCB")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Mean training episode return")
    ax.set_title(f"Learning curve over {args.seeds} seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "multi_seed_curve.png", dpi=140)


if __name__ == "__main__":
    main()
