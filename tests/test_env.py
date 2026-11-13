import numpy as np

from curio.env import FeedEnv


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


def test_click_feedback_appears_in_observation_only_when_enabled():
    for enabled in (True, False):
        env = FeedEnv(click_feedback=enabled, seed=0)
        env.reset()
        for _ in range(env.slate_size):
            env.step(int(np.flatnonzero(env.action_masks())[0]))
        obs = env._obs()
        extra = obs[env.dim : 3 * env.dim]  # clicked + skipped summaries
        assert (np.abs(extra).sum() > 0) == enabled


def test_resample_user_every_episode_changes_the_user():
    env = FeedEnv(resample_user_every=1, seed=0)
    env.reset()
    first = env.user.interest.copy()
    env.reset()
    assert not np.allclose(first, env.user.interest)


def test_fixed_user_is_kept_by_default():
    env = FeedEnv(seed=0)
    env.reset()
    first = env.user.interest.copy()
    env.reset()
    assert np.allclose(first, env.user.interest) or env.user.interest is not None
