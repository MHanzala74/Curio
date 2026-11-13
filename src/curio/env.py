"""Gymnasium environment for slate building.

Each episode is one "day". The agent picks `slate_size` items one by one from
a pool of candidates. The user reacts to each item, and interests drift, so
actions affect future states.

Two options control how hard (and how meaningful) the personalization task is:

- ``click_feedback``: if True, the observation includes summaries of the items
  the user clicked and skipped earlier in the episode, so the policy can infer
  the user's taste. If False, the policy only sees what was shown.
- ``resample_user_every``: if set to N, a brand new simulated user is drawn
  every N episodes. With N=1 the policy cannot memorize one user and must
  personalize from in-episode feedback.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .simulated_user import SimulatedUser


class FeedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        pool_size: int = 30,
        slate_size: int = 5,
        dim: int = 8,
        novelty_weight: float = 0.3,
        click_feedback: bool = False,
        resample_user_every: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.slate_size = slate_size
        self.dim = dim
        self.novelty_weight = novelty_weight
        self.click_feedback = click_feedback
        self.resample_user_every = resample_user_every
        self._seed = seed

        # Observation layout:
        #   [shown summary | clicked summary | skipped summary | pool | availability]
        obs_len = 3 * dim + pool_size * dim + pool_size
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.Discrete(pool_size)

        self.rng = np.random.default_rng(seed)
        self.user: SimulatedUser | None = None
        self.pool = np.zeros((pool_size, dim))
        self.available = np.ones(pool_size, dtype=bool)
        self.step_count = 0
        self.episode_count = 0
        self.clicked_items: list[np.ndarray] = []
        self.skipped_items: list[np.ndarray] = []

    def _new_pool(self) -> np.ndarray:
        items = self.rng.normal(size=(self.pool_size, self.dim))
        return items / (np.linalg.norm(items, axis=1, keepdims=True) + 1e-8)

    def _mean_or_zero(self, items: list[np.ndarray]) -> np.ndarray:
        return np.mean(items, axis=0) if items else np.zeros(self.dim)

    def _obs(self) -> np.ndarray:
        assert self.user is not None
        shown = self._mean_or_zero(self.user.recent)
        if self.click_feedback:
            clicked = self._mean_or_zero(self.clicked_items)
            skipped = self._mean_or_zero(self.skipped_items)
        else:
            clicked = skipped = np.zeros(self.dim)
        obs = np.concatenate(
            [shown, clicked, skipped, self.pool.ravel(), self.available.astype(float)]
        )
        return obs.astype(np.float32)

    def action_masks(self) -> np.ndarray:
        """Boolean mask of items that can still be chosen."""
        return self.available.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        needs_new_user = self.user is None or (
            self.resample_user_every is not None
            and self.episode_count % self.resample_user_every == 0
        )
        if needs_new_user:
            self.user = SimulatedUser(self.dim, self.rng)
        self.episode_count += 1
        self.user.recent = []
        self.pool = self._new_pool()
        self.available = np.ones(self.pool_size, dtype=bool)
        self.step_count = 0
        self.clicked_items, self.skipped_items = [], []
        return self._obs(), {}

    def step(self, action: int):
        assert self.user is not None
        if not self.available[action]:
            # Invalid action: small penalty, no state change besides step count.
            self.step_count += 1
            done = self.step_count >= self.slate_size
            return self._obs(), -1.0, done, False, {"invalid": True}

        item = self.pool[action]
        clicked, novelty = self.user.react(item)
        (self.clicked_items if clicked else self.skipped_items).append(item.copy())
        self.available[action] = False
        self.step_count += 1

        reward = (1.0 if clicked else 0.0) + self.novelty_weight * novelty
        done = self.step_count >= self.slate_size
        return self._obs(), float(reward), done, False, {"clicked": clicked}
