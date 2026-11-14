"""Live (non-simulated) use of the trained policy.

``LiveState`` is a JSON-serializable record of one real user. ``SlateBuilder``
turns real item embeddings into the same observation layout the policy saw in
``LongTermFeedEnv`` and picks a daily slate.

Known simplifications compared with training:

- In training the policy saw click feedback after every item. A real daily digest is
  built **open loop**: clicks only arrive after the whole slate is shown.
- Unlike the 7-day simulated episodes, a real user persists indefinitely; the belief
  accumulates without forgetting and the day counter saturates at the horizon.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .collector import Item
from .ppo import ActorCritic

EXPOSURE_DECAY = 0.97
HORIZON_DAYS = 7
MAX_HISTORY = 50
BELIEF_PRIOR = 0.1


@dataclass
class LiveState:
    dim: int = 8
    exposure: list = field(default_factory=list)
    belief_A: list = field(default_factory=list)
    belief_b: list = field(default_factory=list)
    clicked: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    day: int = 0
    last_day_ctr: float = 0.0
    total_clicks: int = 0
    total_shown: int = 0
    pending: list = field(default_factory=list)       # [{"item": {...}, "vec": [...]}]
    feedback_log: list = field(default_factory=list)  # [{"date","id","title","source","clicked"}]

    def __post_init__(self) -> None:
        if not self.exposure:
            self.exposure = [0.0] * self.dim
        if not self.belief_A:
            self.belief_A = (BELIEF_PRIOR * np.eye(self.dim)).tolist()
        if not self.belief_b:
            self.belief_b = [0.0] * self.dim

    @classmethod
    def load(cls, path: str | Path, dim: int = 8) -> "LiveState":
        path = Path(path)
        return cls(**json.loads(path.read_text())) if path.exists() else cls(dim=dim)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))

    def belief(self) -> np.ndarray:
        return np.clip(np.linalg.solve(np.array(self.belief_A), np.array(self.belief_b)), -1.0, 1.0)

    def apply(self, vec: np.ndarray, clicked: bool) -> None:
        """Apply one shown item and the user's reaction (same update as the simulator)."""
        vec = np.asarray(vec, dtype=float)
        self.exposure = (EXPOSURE_DECAY * np.array(self.exposure) + vec).tolist()
        self.belief_A = (np.array(self.belief_A) + np.outer(vec, vec)).tolist()
        self.belief_b = (np.array(self.belief_b) + (1.0 if clicked else -1.0) * vec).tolist()
        (self.clicked if clicked else self.skipped).append(vec.tolist())
        self.clicked, self.skipped = self.clicked[-MAX_HISTORY:], self.skipped[-MAX_HISTORY:]
        self.total_shown += 1
        self.total_clicks += int(clicked)

    def end_day(self, day_ctr: float) -> None:
        self.last_day_ctr = float(day_ctr)
        self.day = min(self.day + 1, HORIZON_DAYS)


def load_policy(path: str | Path) -> ActorCritic:
    blob = torch.load(path, weights_only=False)
    model = ActorCritic(**blob["config"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def shortlist(items: list[Item], k: int) -> list[int]:
    """Round-robin over sources (each already newest-first) to pick up to ``k`` indices."""
    by_source: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        by_source.setdefault(item.source.split(":")[0], []).append(idx)
    queues, picked = list(by_source.values()), []
    while len(picked) < k and any(queues):
        for q in queues:
            if q and len(picked) < k:
                picked.append(q.pop(0))
    return picked


def _mean_or_zero(vectors: list, dim: int) -> np.ndarray:
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)


class SlateBuilder:
    def __init__(self, model: ActorCritic, slate_size: int = 5) -> None:
        self.model, self.slate_size = model, slate_size
        self.pool_size, self.dim = model.pool_size, model.dim

    def build_obs(self, state: LiveState, pool: np.ndarray, available: np.ndarray,
                  exposure: np.ndarray | None = None) -> np.ndarray:
        """Observation with the exact layout of ``LongTermFeedEnv._obs``."""
        exposure = np.array(state.exposure) if exposure is None else exposure
        summaries = np.concatenate([
            np.clip(exposure / 3.0, -1.0, 1.0),
            _mean_or_zero(state.clicked, self.dim),
            _mean_or_zero(state.skipped, self.dim),
            state.belief(),
        ])
        scalars = np.array([state.day / HORIZON_DAYS, state.last_day_ctr,
                            state.total_clicks / max(state.total_shown, 1)])
        padded = np.zeros((self.pool_size, self.dim))
        padded[: len(pool)] = pool
        avail = np.zeros(self.pool_size)
        avail[: len(pool)] = np.asarray(available, dtype=float)
        return np.concatenate([summaries, scalars, padded.ravel(), avail]).astype(np.float32)

    def select(self, state: LiveState, vectors: np.ndarray) -> list[int]:
        """Pick up to ``slate_size`` distinct candidate indices (greedy policy, open loop)."""
        n = min(len(vectors), self.pool_size)
        pool = np.asarray(vectors[:n])
        available = np.ones(n, dtype=bool)
        exposure = np.array(state.exposure)
        chosen: list[int] = []
        for _ in range(min(self.slate_size, n)):
            obs = self.build_obs(state, pool, available, exposure)
            action, _, _ = self.model.act(obs, greedy=True)
            if action >= n or not available[action]:  # defensive; masking should prevent this
                action = int(np.flatnonzero(available)[0])
            chosen.append(action)
            available[action] = False
            exposure = EXPOSURE_DECAY * exposure + pool[action]
        return chosen


def start_day(state: LiveState, items: list[Item], vectors: np.ndarray) -> None:
    state.pending = [{"item": it.to_dict(), "vec": np.asarray(v).tolist()} for it, v in zip(items, vectors)]


def record_feedback(state: LiveState, clicked_ids: set[str], date: str) -> float:
    """Apply feedback for the pending slate; items not in ``clicked_ids`` count as ignored."""
    pending, clicks = state.pending, 0
    for entry in pending:
        item = Item.from_dict(entry["item"])
        was_clicked = item.id in clicked_ids
        state.apply(np.array(entry["vec"]), was_clicked)
        state.feedback_log.append({"date": date, "id": item.id, "title": item.title,
                                   "source": item.source, "clicked": was_clicked})
        clicks += int(was_clicked)
    ctr = clicks / len(pending) if pending else 0.0
    if pending:
        state.end_day(ctr)
    state.pending = []
    state.feedback_log = state.feedback_log[-500:]
    return ctr
