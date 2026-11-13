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


class OracleDiversePolicy:
    """Oracle that knows the true interest AND avoids redundancy.

    A strong reference: it combines personalization (true interest) with the
    diversity strategy, so it approximates what a good policy should reach.
    """

    name = "oracle-diverse"

    def __init__(self, diversity_weight: float = 0.5) -> None:
        self.diversity_weight = diversity_weight

    def act(self, env: FeedEnv, obs: np.ndarray) -> int:
        scores = env.pool @ env.user.interest
        if env.user.recent:
            recent = np.stack(env.user.recent)
            scores = scores - self.diversity_weight * (env.pool @ recent.T).max(axis=1)
        scores[~env.action_masks()] = -np.inf
        return int(np.argmax(scores))

    def observe(self, *args, **kwargs) -> None:
        pass


class LinUCBPolicy:
    """Contextual bandit. With ``reset_each_episode`` it is a per-user bandit
    that only learns from the current episode's clicks (the fair comparison
    when every episode has a new user)."""

    name = "linucb-bandit"

    def __init__(
        self,
        dim: int,
        alpha: float = 0.5,
        reset_each_episode: bool = False,
        reg: float = 1.0,
        signed_reward: bool = False,
    ) -> None:
        self.alpha = alpha
        self.dim = dim
        self.reset_each_episode = reset_each_episode
        self.reg = reg
        # If True, a skip counts as -1 evidence instead of 0 (uses negative feedback).
        self.signed_reward = signed_reward
        self._reset()

    def _reset(self) -> None:
        self.A = self.reg * np.eye(self.dim)
        self.b = np.zeros(self.dim)

    def start_episode(self) -> None:
        if self.reset_each_episode:
            self._reset()

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
        signal = (2.0 * float(clicked) - 1.0) if self.signed_reward else float(clicked)
        self.b += signal * item


class OracleFatiguePolicy:
    """Oracle for the long-horizon env: knows the true interest and fatigue.

    ``extra_weight = 0`` is the myopic optimum (maximize the immediate click
    probability). A positive ``extra_weight`` additionally avoids items similar
    to current exposure, a crude proxy for planning ahead.
    """

    name = "oracle-fatigue"

    def __init__(self, extra_weight: float = 0.0) -> None:
        self.extra_weight = extra_weight

    def act(self, env, obs) -> int:
        u = env.user
        scores = 3.0 * (env.pool @ u.interest) - (u.fatigue + self.extra_weight) * np.maximum(
            0.0, env.pool @ u.exposure
        )
        scores[~env.action_masks()] = -np.inf
        return int(np.argmax(scores))

    def observe(self, *args, **kwargs) -> None:
        pass


class FatigueAwareLinUCBPolicy(LinUCBPolicy):
    """Per-user LinUCB plus a fatigue penalty on recently shown content.

    The policy tracks its own exposure vector from the items it showed (it never
    reads the hidden user state) and subtracts ``beta * max(0, exposure . item)``
    from each score. ``beta`` and ``decay`` are hyperparameters tuned on
    validation seeds. This is a fair, non-RL counterpart to PPO that has access
    to the same fatigue information.
    """

    name = "linucb-fatigue-aware"

    def __init__(self, dim: int, alpha: float = 2.0, reg: float = 0.3, beta: float = 2.0,
                 decay: float = 0.97) -> None:
        super().__init__(dim, alpha=alpha, reg=reg, reset_each_episode=True, signed_reward=True)
        self.beta, self.decay = beta, decay
        self.exposure = np.zeros(dim)

    def start_episode(self) -> None:
        super().start_episode()
        self.exposure = np.zeros(self.dim)

    def act(self, env, obs) -> int:
        a_inv = np.linalg.inv(self.A)
        theta = a_inv @ self.b
        mean = env.pool @ theta
        bonus = self.alpha * np.sqrt(np.einsum("ij,jk,ik->i", env.pool, a_inv, env.pool))
        scores = mean + bonus - self.beta * np.maximum(0.0, env.pool @ self.exposure)
        scores[~env.action_masks()] = -np.inf
        return int(np.argmax(scores))

    def observe(self, item: np.ndarray, clicked: bool) -> None:
        super().observe(item, clicked)
        self.exposure = self.decay * self.exposure + item
