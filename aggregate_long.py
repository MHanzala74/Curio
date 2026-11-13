"""Aggregate docs/runs_long into a table and a plot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("docs/runs_long")
SEEDS = [0, 1, 2]
ORDER = ["random", "linucb (tuned)", "ppo (gamma=0.9)", "ppo (gamma=0.99)",
         "linucb (fatigue-aware)", "oracle-myopic", "oracle-lookahead"]


def main() -> None:
    ret: dict[str, list[float]] = {}
    days: dict[str, list[float]] = {}
    for variant in ["baselines", "ppo90", "ppo99"]:
        for s in SEEDS:
            payload = json.loads((RUNS / f"{variant}_seed{s}.json").read_text())
            for name, r in payload["results"].items():
                ret.setdefault(name, []).append(r["return"])
                days.setdefault(name, []).append(r["days"])

    lines = [f"| Method | Return (clicks per episode, mean ± std, {len(SEEDS)} seeds) | Days before churn (max 7) |",
             "|---|---|---|"]
    for n in ORDER:
        lines.append(f"| {n} | {np.mean(ret[n]):.2f} ± {np.std(ret[n], ddof=1):.2f} | {np.mean(days[n]):.2f} |")

    def paired(a: str, b: str) -> str:
        d = np.array(ret[a]) - np.array(ret[b])
        return f"- {a} minus {b}: {d.mean():+.2f} ± {d.std(ddof=1):.2f} (first wins on {int((d > 0).sum())}/{len(SEEDS)} seeds)"

    lines += ["", "Paired differences (same evaluation users per seed):",
              paired("ppo (gamma=0.99)", "linucb (tuned)"),
              paired("linucb (fatigue-aware)", "ppo (gamma=0.99)"),
              paired("linucb (fatigue-aware)", "linucb (tuned)"),
              paired("ppo (gamma=0.99)", "ppo (gamma=0.9)"),
              paired("oracle-lookahead", "oracle-myopic")]
    lo, hi = np.mean(ret["random"]), np.mean(ret["oracle-lookahead"])
    frac = (np.mean(ret["ppo (gamma=0.99)"]) - lo) / (hi - lo)
    frac_b = (np.mean(ret["linucb (fatigue-aware)"]) - lo) / (hi - lo)
    lines += ["", f"Share of the random-to-oracle-lookahead gap closed: PPO {100 * frac:.0f}%, "
                  f"fatigue-aware LinUCB {100 * frac_b:.0f}%."]
    Path("docs/long_horizon_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    means = [np.mean(ret[n]) for n in ORDER]
    errs = [np.std(ret[n], ddof=1) for n in ORDER]
    ax.barh(ORDER, means, xerr=errs,
            color=["gray", "#9bd39b", "#8fb8e8", "tab:blue", "tab:green", "tab:purple", "tab:red"])
    ax.invert_yaxis()
    ax.set_xlabel("Clicks per episode (7-day horizon, churn ends the episode)")
    ax.set_title("Long-horizon task with fatigue and churn")
    fig.tight_layout()
    fig.savefig("docs/long_horizon.png", dpi=140)


if __name__ == "__main__":
    main()
