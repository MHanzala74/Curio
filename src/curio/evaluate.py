"""Shared evaluation helpers."""

from __future__ import annotations

import numpy as np

from .env import FeedEnv


def run_policy(policy, episodes: int, seed: int, **env_kwargs) -> dict:
    """Run a baseline policy on a fresh env and report mean episode return."""
    env = FeedEnv(seed=seed, **env_kwargs)
    returns, clicks = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        if hasattr(policy, "start_episode"):
            policy.start_episode()
        done, total, n_clicks = False, 0.0, 0
        while not done:
            action = policy.act(env, obs)
            item = env.pool[action].copy()
            obs, reward, done, _, info = env.step(action)
            policy.observe(item, info.get("clicked", False))
            total += reward
            n_clicks += int(info.get("clicked", False))
        returns.append(total)
        clicks.append(n_clicks)
    return {"return": float(np.mean(returns)), "ctr": float(np.mean(clicks)) / env.slate_size}


def run_ppo(model, episodes: int, seed: int, **env_kwargs) -> dict:
    """Evaluate a trained actor-critic greedily on a fresh env."""
    from .ppo import ActorCritic  # noqa: F401  (kept for type clarity)

    env = FeedEnv(seed=seed, **env_kwargs)
    returns, clicks = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total, n_clicks = False, 0.0, 0
        while not done:
            action, _, _ = model.act(obs, greedy=True)
            obs, reward, done, _, info = env.step(action)
            total += reward
            n_clicks += int(info.get("clicked", False))
        returns.append(total)
        clicks.append(n_clicks)
    return {"return": float(np.mean(returns)), "ctr": float(np.mean(clicks)) / env.slate_size}
