| Method | Episode return (mean ± std, 3 seeds) | CTR |
|---|---|---|
| random | 6.048 ± 0.029 | 0.401 |
| linucb (untuned) | 6.116 ± 0.080 | 0.445 |
| ppo (no click feedback) | 6.642 ± 0.034 | 0.428 |
| ppo (with click feedback) | 6.894 ± 0.109 | 0.467 |
| linucb (tuned) | 6.915 ± 0.065 | 0.468 |
| ppo (click feedback + belief state) | 6.974 ± 0.069 | 0.468 |
| oracle-greedy | 7.554 ± 0.088 | 0.597 |
| oracle-diverse | 7.979 ± 0.093 | 0.614 |

Paired differences (same evaluation users and pools per seed):
- ppo (with click feedback) minus ppo (no click feedback): +0.252 ± 0.075 (ppo (with click feedback) wins on 3/3 seeds)
- ppo (click feedback + belief state) minus ppo (with click feedback): +0.080 ± 0.056 (ppo (click feedback + belief state) wins on 3/3 seeds)
- ppo (click feedback + belief state) minus linucb (tuned): +0.059 ± 0.069 (ppo (click feedback + belief state) wins on 2/3 seeds)
- ppo (with click feedback) minus linucb (tuned): -0.021 ± 0.067 (ppo (with click feedback) wins on 1/3 seeds)
