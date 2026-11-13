"""Multi-day feed environment with fatigue and churn.

One episode is up to ``days`` days. Each day the agent picks ``slate_size``
items one at a time from a fresh pool. The reward is **only clicks**, so any
benefit of diversity must come from keeping the user engaged for longer.

Hidden user dynamics (the agent never sees these directly):

- ``exposure``: a decaying sum of recently shown items. Showing items similar
  to recent exposure lowers the click probability (fatigue), and the effect
  lasts across days.
- ``interest`` drifts toward clicked items (a mild filter-bubble effect).
- ``satisfaction`` is a running average of daily click-through rate. If it
  falls below ``churn_threshold`` at the end of a day, the user churns and the
  episode ends, so all future clicks are lost.

The agent can observe what it showed (so it can track exposure), the clicks and
skips, a running belief about the user's interest, and day-level CTR history.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LongTermUser:
    def __init__(self, dim: int, rng: np.random.Generator, fatigue: float,
                 exposure_decay: float, drift_rate: float) -> None:
        self.dim, self.rng = dim, rng
        self.fatigue, self.exposure_decay, self.drift_rate = fatigue, exposure_decay, drift_rate
        v = rng.normal(size=dim)
        self.interest = v / (np.linalg.norm(v) + 1e-8)
        self.exposure = np.zeros(dim)
        self.satisfaction = 0.6

    def logit(self, item: np.ndarray) -> float:
        return 3.0 * float(self.interest @ item) - self.fatigue * max(0.0, float(self.exposure @ item))

    def click_probability(self, item: np.ndarray) -> float:
        return 1.0 / (1.0 + np.exp(-self.logit(item)))

    def react(self, item: np.ndarray) -> bool:
        clicked = bool(self.rng.random() < self.click_probability(item))
        self.exposure = self.exposure_decay * self.exposure + item
        if clicked:
            self.interest = (1 - self.drift_rate) * self.interest + self.drift_rate * item
            self.interest /= np.linalg.norm(self.interest) + 1e-8
        return clicked


class LongTermFeedEnv(gym.Env):
    metadata = {"render_modes": []}
    N_SUMMARIES = 4  # exposure, clicked mean, skipped mean, belief
    N_SCALARS = 3    # day fraction, last day CTR, CTR so far

    def __init__(
        self,
        pool_size: int = 30,
        slate_size: int = 5,
        dim: int = 8,
        days: int = 7,
        fatigue: float = 2.5,
        exposure_decay: float = 0.97,
        drift_rate: float = 0.15,
        sat_decay: float = 0.6,
        churn_threshold: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.pool_size, self.slate_size, self.dim, self.days = pool_size, slate_size, dim, days
        self.fatigue, self.exposure_decay, self.drift_rate = fatigue, exposure_decay, drift_rate
        self.sat_decay, self.churn_threshold = sat_decay, churn_threshold

        obs_len = self.N_SUMMARIES * dim + self.N_SCALARS + pool_size * dim + pool_size
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.Discrete(pool_size)
        self.rng = np.random.default_rng(seed)
        self.user: LongTermUser | None = None
        self.pool = np.zeros((pool_size, dim))
        self.available = np.ones(pool_size, dtype=bool)

    # ---- helpers -------------------------------------------------------
    def _new_pool(self) -> np.ndarray:
        items = self.rng.normal(size=(self.pool_size, self.dim))
        return items / (np.linalg.norm(items, axis=1, keepdims=True) + 1e-8)

    def _mean_or_zero(self, items) -> np.ndarray:
        return np.mean(items, axis=0) if items else np.zeros(self.dim)

    def _obs(self) -> np.ndarray:
        belief = np.clip(np.linalg.solve(self._A, self._b), -1.0, 1.0)
        exposure = np.clip(self.user.exposure / 3.0, -1.0, 1.0)
        summaries = np.concatenate([exposure, self._mean_or_zero(self.clicked_items),
                                    self._mean_or_zero(self.skipped_items), belief])
        steps_done = max(self.total_steps, 1)
        scalars = np.array([
            self.day / self.days,
            self.last_day_ctr,
            self.total_clicks / steps_done,
        ])
        return np.concatenate(
            [summaries, scalars, self.pool.ravel(), self.available.astype(float)]
        ).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        return self.available.copy()

    # ---- gym API -------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.user = LongTermUser(self.dim, self.rng, self.fatigue, self.exposure_decay, self.drift_rate)
        self.day, self.step_in_day = 0, 0
        self.total_steps, self.total_clicks, self.day_clicks = 0, 0, 0
        self.last_day_ctr = 0.0
        self.clicked_items, self.skipped_items = [], []
        self._A, self._b = 0.1 * np.eye(self.dim), np.zeros(self.dim)
        self.pool = self._new_pool()
        self.available = np.ones(self.pool_size, dtype=bool)
        return self._obs(), {}

    def step(self, action: int):
        item = self.pool[action]
        clicked = self.user.react(item)
        (self.clicked_items if clicked else self.skipped_items).append(item.copy())
        self._A += np.outer(item, item)
        self._b += (1.0 if clicked else -1.0) * item
        self.available[action] = False
        self.step_in_day += 1
        self.total_steps += 1
        self.total_clicks += int(clicked)
        self.day_clicks += int(clicked)

        done, churned = False, False
        if self.step_in_day == self.slate_size:  # end of day
            self.last_day_ctr = self.day_clicks / self.slate_size
            u = self.user
            u.satisfaction = self.sat_decay * u.satisfaction + (1 - self.sat_decay) * self.last_day_ctr
            self.day += 1
            self.step_in_day, self.day_clicks = 0, 0
            self.pool = self._new_pool()
            self.available = np.ones(self.pool_size, dtype=bool)
            if u.satisfaction < self.churn_threshold:
                done, churned = True, True
            elif self.day >= self.days:
                done = True
        info = {"clicked": clicked, "churned": churned, "days": self.day}
        return self._obs(), float(clicked), done, False, info
