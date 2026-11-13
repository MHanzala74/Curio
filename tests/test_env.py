import numpy as np

from rlfeed.env import FeedEnv


def test_reset_shape():
    env = FeedEnv(seed=0)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape


def test_episode_length_equals_slate_size():
    env = FeedEnv(slate_size=5, seed=0)
    env.reset()
    steps, done = 0, False
    while not done:
        mask = env.action_masks()
        action = int(np.flatnonzero(mask)[0])
        _, _, done, _, _ = env.step(action)
        steps += 1
    assert steps == 5


def test_invalid_action_is_penalized():
    env = FeedEnv(seed=0)
    env.reset()
    env.step(0)
    _, reward, _, _, info = env.step(0)
    assert reward < 0 and info.get("invalid")


def test_same_seed_is_reproducible():
    a, b = FeedEnv(seed=1), FeedEnv(seed=1)
    oa, _ = a.reset(seed=1)
    ob, _ = b.reset(seed=1)
    assert np.allclose(oa, ob)
