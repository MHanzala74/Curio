"""Tune the per-user LinUCB baseline on validation seeds.

Tuning uses seeds 100-102 (never the evaluation seeds 1000+), so the baseline
gets a fair chance without leaking test information.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from curio.baselines import LinUCBPolicy
from curio.evaluate import run_policy

ENV = dict(slate_size=10, resample_user_every=1)
VAL_SEEDS = [100, 101, 102]


def main() -> None:
    rows = []
    for alpha, reg, signed in itertools.product([1.0, 2.0, 3.0, 5.0, 8.0], [0.05, 0.1, 0.2], [True]):
        scores = [
            run_policy(
                LinUCBPolicy(8, alpha=alpha, reset_each_episode=True, reg=reg, signed_reward=signed),
                200, 2000 + s, **ENV,
            )["return"]
            for s in VAL_SEEDS
        ]
        rows.append((float(np.mean(scores)), alpha, reg, signed))
    rows.sort(reverse=True)
    lines = ["| alpha | reg | signed reward | validation return |", "|---|---|---|---|"]
    for score, alpha, reg, signed in rows:
        lines.append(f"| {alpha} | {reg} | {signed} | {score:.3f} |")
    Path("docs/linucb_tuning.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    print("best:", rows[0])


if __name__ == "__main__":
    main()
