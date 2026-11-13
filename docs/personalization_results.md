| Method | Episode return (mean ± std, 3 seeds) | CTR |
|---|---|---|
| random | 6.048 ± 0.029 | 0.401 |
| linucb (per-user) | 6.116 ± 0.080 | 0.445 |
| ppo (no click feedback) | 6.630 ± 0.138 | 0.429 |
| ppo (with click feedback) | 6.866 ± 0.051 | 0.460 |
| oracle-greedy | 7.554 ± 0.088 | 0.597 |
| oracle-diverse | 7.979 ± 0.093 | 0.614 |

Paired difference, PPO with vs. without click feedback: +0.236 ± 0.093 (with-feedback wins on 3/3 seeds).
Paired difference, PPO with click feedback vs. per-user LinUCB: +0.751 ± 0.054 (wins on 3/3 seeds).
