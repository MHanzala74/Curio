"""Shared evaluation helpers (work with any env that exposes pool/action_masks)."""

from __future__ import annotations

import numpy as np

from .env import FeedEnv


def _summarize(returns, clicks, steps, days) -> dict:
    return {
        "return": float(np.mean(returns)),
        "ctr": float(np.sum(clicks) / max(np.sum(steps), 1)),
        "days": float(np.mean(days)),
    }


def run_policy(policy, episodes: int, seed: int, env_cls=FeedEnv, **env_kwargs) -> dict:
    """Run a baseline policy on a fresh env and report mean episode return."""
    env = env_cls(seed=seed, **env_kwargs)
    returns, clicks, steps, days = [], [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        if hasattr(policy, "start_episode"):
            policy.start_episode()
        done, total, n_clicks, n_steps, info = False, 0.0, 0, 0, {}
        while not done:
            action = policy.act(env, obs)
            item = env.pool[action].copy()
            obs, reward, done, _, info = env.step(action)
            policy.observe(item, info.get("clicked", False))
            total += reward
            n_clicks += int(info.get("clicked", False))
            n_steps += 1
        returns.append(total)
        clicks.append(n_clicks)
        steps.append(n_steps)
        days.append(info.get("days", n_steps / env.slate_size))
    return _summarize(returns, clicks, steps, days)


def run_ppo(model, episodes: int, seed: int, env_cls=FeedEnv, **env_kwargs) -> dict:
    """Evaluate a trained actor-critic greedily on a fresh env."""
    env = env_cls(seed=seed, **env_kwargs)
    returns, clicks, steps, days = [], [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total, n_clicks, n_steps, info = False, 0.0, 0, 0, {}
        while not done:
            action, _, _ = model.act(obs, greedy=True)
            obs, reward, done, _, info = env.step(action)
            total += reward
            n_clicks += int(info.get("clicked", False))
            n_steps += 1
        returns.append(total)
        clicks.append(n_clicks)
        steps.append(n_steps)
        days.append(info.get("days", n_steps / env.slate_size))
    return _summarize(returns, clicks, steps, days)
