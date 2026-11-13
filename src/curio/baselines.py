"""Baseline policies used for comparison against PPO.

- RandomPolicy: picks any available item uniformly.
- OracleGreedyPolicy: peeks at the true user interest (an upper-bound-style
  reference that ignores novelty and long-term drift).
- LinUCBPolicy: a contextual bandit that learns click probability per item
  feature vector. It optimizes immediate reward only.
"""

from __future__ import annotations

import numpy as np

from .env import FeedEnv


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, env: FeedEnv, obs: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(env.action_masks())))

    def observe(self, *args, **kwargs) -> None:
        pass


class OracleGreedyPolicy:
    name = "oracle-greedy"

    def act(self, env: FeedEnv, obs: np.ndarray) -> int:
        scores = env.pool @ env.user.interest
        scores[~env.action_masks()] = -np.inf
        return int(np.argmax(scores))

    def observe(self, *args, **kwargs) -> None:
        pass


class LinUCBPolicy:
    name = "linucb-bandit"

    def __init__(self, dim: int, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.A = np.eye(dim)
        self.b = np.zeros(dim)

    def act(self, env: FeedEnv, obs: np.ndarray) -> int:
        a_inv = np.linalg.inv(self.A)
        theta = a_inv @ self.b
        mean = env.pool @ theta
        bonus = self.alpha * np.sqrt(np.einsum("ij,jk,ik->i", env.pool, a_inv, env.pool))
        scores = mean + bonus
        scores[~env.action_masks()] = -np.inf
        return int(np.argmax(scores))

    def observe(self, item: np.ndarray, clicked: bool) -> None:
        self.A += np.outer(item, item)
        self.b += float(clicked) * item
