import numpy as np

from curio.env_longterm import LongTermFeedEnv


def test_obs_shape_and_episode_terminates():
    env = LongTermFeedEnv(seed=0)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    done, steps = False, 0
    while not done:
        _, _, done, _, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
        steps += 1
    assert steps <= env.days * env.slate_size


def test_reward_is_only_clicks():
    env = LongTermFeedEnv(seed=1)
    env.reset()
    for _ in range(20):
        _, r, done, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert r == float(info["clicked"])
        if done:
            break


def test_repeating_similar_items_lowers_click_probability():
    env = LongTermFeedEnv(seed=2)
    env.reset()
    item = env.pool[0]
    before = env.user.click_probability(item)
    for _ in range(5):
        env.user.react(item)
    assert env.user.click_probability(item) < before


def test_user_can_churn_and_end_the_episode_early():
    env = LongTermFeedEnv(seed=3, churn_threshold=0.99)  # almost everyone churns
    env.reset()
    done, info = False, {}
    while not done:
        _, _, done, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert info["churned"] and info["days"] < env.days
