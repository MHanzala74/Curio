from rlfeed.baselines import LinUCBPolicy, OracleGreedyPolicy, RandomPolicy
from rlfeed.evaluate import run_policy


def test_random_is_worse_than_oracle():
    rand = run_policy(RandomPolicy(seed=0), 100, seed=3)
    oracle = run_policy(OracleGreedyPolicy(), 100, seed=3)
    assert oracle["return"] > rand["return"]


def test_linucb_beats_random():
    rand = run_policy(RandomPolicy(seed=0), 150, seed=3)
    bandit = run_policy(LinUCBPolicy(dim=8), 150, seed=3)
    assert bandit["return"] > rand["return"]
