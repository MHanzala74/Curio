from curio.baselines import LinUCBPolicy, OracleGreedyPolicy, RandomPolicy
from curio.evaluate import run_policy


def test_random_is_worse_than_oracle():
    rand = run_policy(RandomPolicy(seed=0), 100, seed=3)
    oracle = run_policy(OracleGreedyPolicy(), 100, seed=3)
    assert oracle["return"] > rand["return"]


def test_linucb_beats_random():
    rand = run_policy(RandomPolicy(seed=0), 150, seed=3)
    bandit = run_policy(LinUCBPolicy(dim=8), 150, seed=3)
    assert bandit["return"] > rand["return"]


def test_oracle_diverse_beats_plain_oracle_when_novelty_matters():
    from curio.baselines import OracleDiversePolicy

    kw = dict(slate_size=10, resample_user_every=1)
    plain = run_policy(OracleGreedyPolicy(), 150, seed=4, **kw)
    diverse = run_policy(OracleDiversePolicy(), 150, seed=4, **kw)
    assert diverse["return"] > plain["return"]


def test_fatigue_aware_policy_tracks_exposure_and_avoids_it():
    import numpy as np

    from curio.baselines import FatigueAwareLinUCBPolicy
    from curio.env_longterm import LongTermFeedEnv

    env = LongTermFeedEnv(seed=0)
    env.reset()
    policy = FatigueAwareLinUCBPolicy(8, alpha=0.0, beta=10.0, decay=0.97)
    policy.start_episode()
    policy.observe(env.pool[0].copy(), True)
    assert np.linalg.norm(policy.exposure) > 0
    choice = policy.act(env, None)
    assert float(env.pool[choice] @ policy.exposure) <= 0.0
