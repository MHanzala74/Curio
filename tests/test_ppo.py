import numpy as np

from curio.env import FeedEnv
from curio.ppo import PPO, ActorCritic


def test_gae_matches_manual_computation():
    model = ActorCritic(pool_size=4, dim=2)
    ppo = PPO(model)
    ppo.cfg.gamma, ppo.cfg.gae_lambda = 0.5, 1.0
    rewards, values, dones = [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]
    adv, ret = ppo.compute_gae(rewards, values, dones)
    # Last step: delta = 1. First step: 1 + 0.5 * 0 - 0 + 0.5 * 1 = 1.5
    assert np.allclose(adv, [1.5, 1.0])
    assert np.allclose(ret, [1.5, 1.0])


def test_policy_never_picks_masked_items():
    env = FeedEnv(seed=0)
    model = ActorCritic(env.pool_size, env.dim)
    obs, _ = env.reset()
    env.step(0)
    obs = env._obs()
    for _ in range(50):
        action, _, _ = model.act(obs)
        assert action != 0


def test_update_changes_parameters():
    env = FeedEnv(seed=0)
    model = ActorCritic(env.pool_size, env.dim)
    ppo = PPO(model)
    before = [p.detach().clone() for p in model.parameters()]
    obs, acts, logps, advs, rets = [], [], [], [], []
    o, _ = env.reset()
    for _ in range(5):
        a, lp, _ = model.act(o)
        obs.append(o); acts.append(a); logps.append(lp); advs.append(1.0); rets.append(1.0)
        o, *_ = env.step(a)
    stats = ppo.update(obs, acts, logps, advs, rets)
    after = list(model.parameters())
    assert any(not np.allclose(b.numpy(), a.detach().numpy()) for b, a in zip(before, after))
    assert set(stats) == {"policy_loss", "value_loss", "entropy"}
