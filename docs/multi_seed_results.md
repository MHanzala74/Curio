| Method | Mean episode return (mean ± std over 5 seeds) | CTR |
|---|---|---|
| random | 3.454 ± 0.038 | 0.443 ± 0.007 |
| linucb-bandit | 4.443 ± 0.049 | 0.701 ± 0.011 |
| oracle-greedy | 4.450 ± 0.030 | 0.706 ± 0.007 |
| ppo (ours) | 4.765 ± 0.057 | 0.706 ± 0.012 |

Paired difference (PPO - LinUCB): +0.323 ± 0.044; PPO wins on 5/5 seeds.
