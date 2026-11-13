"""A from-scratch PPO actor-critic with action masking.

The actor scores each candidate item with a shared MLP (so the policy is
permutation-invariant over the pool). The critic estimates the state value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, pool_size: int, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.pool_size, self.dim = pool_size, dim
        # Per-item features: summary, item, summary*item, availability flag.
        self.scorer = nn.Sequential(
            nn.Linear(3 * dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def _split(self, obs: torch.Tensor):
        b = obs.shape[0]
        summary = obs[:, : self.dim]
        pool = obs[:, self.dim : self.dim + self.pool_size * self.dim]
        pool = pool.reshape(b, self.pool_size, self.dim)
        avail = obs[:, -self.pool_size :]
        return summary, pool, avail

    def forward(self, obs: torch.Tensor):
        summary, pool, avail = self._split(obs)
        s = summary.unsqueeze(1).expand_as(pool)
        feats = torch.cat([s, pool, s * pool, avail.unsqueeze(-1)], dim=-1)
        logits = self.scorer(feats).squeeze(-1)
        logits = logits.masked_fill(avail < 0.5, -1e9)

        w = avail.unsqueeze(-1)
        pool_mean = (pool * w).sum(1) / w.sum(1).clamp(min=1.0)
        remaining = w.sum(1) / self.pool_size
        value = self.critic(torch.cat([summary, pool_mean, remaining], dim=-1)).squeeze(-1)
        return logits, value

    def act(self, obs: np.ndarray, greedy: bool = False):
        with torch.no_grad():
            logits, value = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if greedy else dist.sample()
            return int(action), float(dist.log_prob(action)), float(value)


@dataclass
class PPOConfig:
    gamma: float = 0.9
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-3
    epochs: int = 4
    minibatch: int = 128
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class PPO:
    def __init__(self, model: ActorCritic, config: PPOConfig | None = None) -> None:
        self.model = model
        self.cfg = config or PPOConfig()
        self.opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)

    def compute_gae(self, rewards, values, dones, last_value=0.0):
        """Generalized Advantage Estimation over a flat trajectory."""
        adv = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * gae
            adv[t] = gae
        returns = adv + np.asarray(values, dtype=np.float32)
        return adv, returns

    def update(self, obs, actions, old_logp, advantages, returns) -> dict:
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        old_logp = torch.as_tensor(old_logp, dtype=torch.float32)
        adv = torch.as_tensor(advantages, dtype=torch.float32)
        ret = torch.as_tensor(returns, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        steps = 0
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.cfg.minibatch):
                idx = perm[start : start + self.cfg.minibatch]
                logits, value = self.model(obs[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                unclipped = ratio * adv[idx]
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (value - ret[idx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                steps += 1
        return {k: v / max(steps, 1) for k, v in stats.items()}
