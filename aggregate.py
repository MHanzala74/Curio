"""Aggregate per-seed results from docs/runs into a table and a plot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("docs/runs")
SEEDS = [0, 1, 2]  # seeds available for every method


def load(variant: str):
    return [json.loads((RUNS / f"{variant}_seed{s}.json").read_text()) for s in SEEDS]


def main() -> None:
    scores: dict[str, list[float]] = {}
    ctrs: dict[str, list[float]] = {}
    for variant in ["baselines", "noclick", "click", "belief"]:
        for payload in load(variant):
            for name, r in payload["results"].items():
                scores.setdefault(name, []).append(r["return"])
                ctrs.setdefault(name, []).append(r["ctr"])

    order = ["random", "linucb (untuned)", "ppo (no click feedback)",
             "ppo (with click feedback)", "linucb (tuned)",
             "ppo (click feedback + belief state)", "oracle-greedy", "oracle-diverse"]
    lines = [f"| Method | Episode return (mean ± std, {len(SEEDS)} seeds) | CTR |", "|---|---|---|"]
    for name in order:
        lines.append(f"| {name} | {np.mean(scores[name]):.3f} ± {np.std(scores[name], ddof=1):.3f} "
                     f"| {np.mean(ctrs[name]):.3f} |")

    def paired(a_name: str, b_name: str) -> str:
        d = np.array(scores[a_name]) - np.array(scores[b_name])
        return (f"- {a_name} minus {b_name}: {d.mean():+.3f} ± {d.std(ddof=1):.3f} "
                f"({a_name} wins on {int((d > 0).sum())}/{len(SEEDS)} seeds)")

    lines += [
        "",
        "Paired differences (same evaluation users and pools per seed):",
        paired("ppo (with click feedback)", "ppo (no click feedback)"),
        paired("ppo (click feedback + belief state)", "ppo (with click feedback)"),
        paired("ppo (click feedback + belief state)", "linucb (tuned)"),
        paired("ppo (with click feedback)", "linucb (tuned)"),
    ]
    Path("docs/personalization_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    means = [np.mean(scores[n]) for n in order]
    errs = [np.std(scores[n], ddof=1) for n in order]
    colors = ["gray", "#9bd39b", "tab:orange", "tab:blue", "tab:green", "tab:cyan", "tab:purple", "tab:red"]
    ax[0].barh(order, means, xerr=errs, color=colors)
    ax[0].set_xlim(5.5, 8.4)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Mean episode return")
    ax[0].set_title("Personalization task (new user every episode)")
    for variant, label in [("noclick", "no click feedback"), ("click", "with click feedback"), ("belief", "click + belief state")]:
        curves = np.array([p["train_curve"] for p in load(variant)])
        k = 10
        sm = np.array([np.convolve(c, np.ones(k) / k, mode="valid") for c in curves])
        x = np.arange(k, curves.shape[1] + 1)
        ax[1].plot(x, sm.mean(0), label=f"PPO, {label}")
        ax[1].fill_between(x, sm.mean(0) - sm.std(0, ddof=1), sm.mean(0) + sm.std(0, ddof=1), alpha=0.2)
    ax[1].axhline(np.mean(scores["random"]), color="gray", ls="--", label="random")
    ax[1].set_xlabel("PPO update")
    ax[1].set_ylabel("Training return (10-update average)")
    ax[1].set_title("Learning curves")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("docs/personalization.png", dpi=140)


if __name__ == "__main__":
    main()
