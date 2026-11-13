"""Reusable PPO training loop."""

from __future__ import annotations

import numpy as np
import torch

from .env import FeedEnv
from .ppo import PPO, ActorCritic, PPOConfig


def collect(env: FeedEnv, model: ActorCritic, episodes: int):
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


def train_ppo(seed: int, updates: int, episodes_per_update: int, log_every: int = 0, lr: float = 3e-3, **env_kwargs):
    """Train a fresh actor-critic and return (model, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = FeedEnv(seed=seed, **env_kwargs)
    model = ActorCritic(env.pool_size, env.dim)
    ppo = PPO(model, PPOConfig(lr=lr))
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
