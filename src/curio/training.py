"""Reusable PPO training loop."""

from __future__ import annotations

import numpy as np
import torch

from .env import FeedEnv
from .ppo import PPO, ActorCritic, PPOConfig


def collect(env, model: ActorCritic, episodes: int):
    """Roll out `episodes` full episodes and return flat trajectory buffers."""
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    ep_returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action, logp, value = model.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(reward)
            val_buf.append(value)
            done_buf.append(float(done))
            obs, total = next_obs, total + reward
        ep_returns.append(total)
    return obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf, ep_returns


def train_ppo(seed: int, updates: int, episodes_per_update: int, log_every: int = 0, lr: float = 3e-3,
              gamma: float = 0.9, env_cls=FeedEnv, **env_kwargs):
    """Train a fresh actor-critic and return (model, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = env_cls(seed=seed, **env_kwargs)
    model = ActorCritic(
        env.pool_size, env.dim,
        n_summaries=getattr(env, "N_SUMMARIES", 4), n_scalars=getattr(env, "N_SCALARS", 0),
    )
    ppo = PPO(model, PPOConfig(lr=lr, gamma=gamma))
    history = []
    for update in range(1, updates + 1):
        obs, act, logp, rew, val, done, ep_returns = collect(env, model, episodes_per_update)
        adv, ret = ppo.compute_gae(rew, val, done)
        stats = ppo.update(obs, act, logp, adv, ret)
        history.append({"update": update, "train_return": float(np.mean(ep_returns)), **stats})
        if log_every and update % log_every == 0:
            print(f"seed {seed} | update {update:4d} | return {history[-1]['train_return']:.3f} "
                  f"| entropy {stats['entropy']:.3f}", flush=True)
    return model, history


def train_ppo_chunked(seed: int, total_updates: int, episodes_per_update: int, chunk: int,
                      ckpt_path: str, lr: float = 5e-4, gamma: float = 0.99, env_cls=FeedEnv,
                      model_kwargs: dict | None = None, log_every: int = 50, **env_kwargs):
    """Train for at most ``chunk`` updates, resuming from ``ckpt_path`` if it exists.

    Long runs can be split over several calls. Model, optimizer and history are
    restored exactly; the random number streams are re-seeded per chunk, so a
    chunked run is statistically equivalent to, but not bit-identical with, a
    single uninterrupted run. Returns ``(model, history, finished)``.
    """
    import os

    env = env_cls(seed=seed, **env_kwargs)
    model = ActorCritic(
        env.pool_size, env.dim,
        n_summaries=getattr(env, "N_SUMMARIES", 4), n_scalars=getattr(env, "N_SCALARS", 0),
        **(model_kwargs or {}),
    )
    ppo = PPO(model, PPOConfig(lr=lr, gamma=gamma))
    history, done_updates = [], 0
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(state["model"])
        ppo.opt.load_state_dict(state["opt"])
        history, done_updates = state["history"], state["updates"]
    torch.manual_seed(seed * 100003 + done_updates)
    np.random.seed(seed * 100003 + done_updates)
    env = env_cls(seed=seed * 100003 + done_updates, **env_kwargs)

    target = min(total_updates, done_updates + chunk)
    for update in range(done_updates + 1, target + 1):
        obs, act, logp, rew, val, done, ep_returns = collect(env, model, episodes_per_update)
        adv, ret = ppo.compute_gae(rew, val, done)
        stats = ppo.update(obs, act, logp, adv, ret)
        history.append({"update": update, "train_return": float(np.mean(ep_returns)), **stats})
        if log_every and update % log_every == 0:
            print(f"seed {seed} | update {update:4d} | return {history[-1]['train_return']:.3f} "
                  f"| entropy {stats['entropy']:.3f}", flush=True)
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": ppo.opt.state_dict(),
                "history": history, "updates": target}, ckpt_path)
    return model, history, target >= total_updates
