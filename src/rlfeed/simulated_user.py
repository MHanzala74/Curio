"""A simple simulated user whose interests drift over time.

The simulator is used to pretrain the policy before it sees real feedback.
It is intentionally small and fully deterministic given a seed.
"""

from __future__ import annotations

import numpy as np


class SimulatedUser:
    """Holds a latent interest vector and reacts to shown items.

    Interests drift toward items the user engaged with, which is what makes
    the problem sequential: today's slate changes tomorrow's state.
    """

    def __init__(
        self,
        dim: int,
        rng: np.random.Generator,
        drift_rate: float = 0.05,
        boredom_decay: float = 0.15,
    ) -> None:
        self.dim = dim
        self.rng = rng
        self.drift_rate = drift_rate
        self.boredom_decay = boredom_decay
        self.interest = self._random_unit_vector()
        self.recent: list[np.ndarray] = []

    def _random_unit_vector(self) -> np.ndarray:
        v = self.rng.normal(size=self.dim)
        return v / (np.linalg.norm(v) + 1e-8)

    def engagement_probability(self, item: np.ndarray) -> float:
        """Probability of a click, based on similarity minus boredom."""
        similarity = float(self.interest @ item)
        boredom = 0.0
        if self.recent:
            boredom = max(float(r @ item) for r in self.recent)
        logit = 3.0 * similarity - 2.0 * boredom * self.boredom_decay * 5
        return 1.0 / (1.0 + np.exp(-logit))

    def react(self, item: np.ndarray) -> tuple[bool, float]:
        """Return (clicked, novelty) for a shown item."""
        p = self.engagement_probability(item)
        clicked = bool(self.rng.random() < p)
        novelty = 1.0
        if self.recent:
            novelty = 1.0 - max(float(r @ item) for r in self.recent)
        self.recent.append(item)
        self.recent = self.recent[-10:]
        if clicked:
            self.interest = (1 - self.drift_rate) * self.interest + self.drift_rate * item
            self.interest /= np.linalg.norm(self.interest) + 1e-8
        return clicked, novelty
