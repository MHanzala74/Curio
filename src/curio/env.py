"""Gymnasium environment for slate building.

Each episode is one "day". The agent picks `slate_size` items one by one from
a pool of candidates. The user reacts to each item, and interests drift, so
actions affect future states.
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
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.slate_size = slate_size
        self.dim = dim
        self.novelty_weight = novelty_weight
        self._seed = seed

        # Observation: user history summary + flattened pool + availability mask.
        obs_len = dim + pool_size * dim + pool_size
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.Discrete(pool_size)

        self.rng = np.random.default_rng(seed)
        self.user: SimulatedUser | None = None
        self.pool = np.zeros((pool_size, dim))
        self.available = np.ones(pool_size, dtype=bool)
        self.step_count = 0

    def _new_pool(self) -> np.ndarray:
        items = self.rng.normal(size=(self.pool_size, self.dim))
        return items / (np.linalg.norm(items, axis=1, keepdims=True) + 1e-8)

    def _history_summary(self) -> np.ndarray:
        assert self.user is not None
        if not self.user.recent:
            return np.zeros(self.dim)
        return np.mean(self.user.recent, axis=0)

    def _obs(self) -> np.ndarray:
        obs = np.concatenate(
            [self._history_summary(), self.pool.ravel(), self.available.astype(float)]
        )
        return obs.astype(np.float32)

    def action_masks(self) -> np.ndarray:
        """Boolean mask of items that can still be chosen."""
        return self.available.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Keep the same user across episodes so long-term drift is learnable.
        if self.user is None:
            self.user = SimulatedUser(self.dim, self.rng)
        self.user.recent = []
        self.pool = self._new_pool()
        self.available = np.ones(self.pool_size, dtype=bool)
        self.step_count = 0
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
        self.available[action] = False
        self.step_count += 1

        reward = (1.0 if clicked else 0.0) + self.novelty_weight * novelty
        done = self.step_count >= self.slate_size
        return self._obs(), float(reward), done, False, {"clicked": clicked}
