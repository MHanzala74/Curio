| Method | Return (clicks per episode, mean ± std, 3 seeds) | Days before churn (max 7) |
|---|---|---|
| random | 4.20 ± 0.04 | 2.20 |
| linucb (tuned) | 5.74 ± 0.11 | 2.67 |
| ppo (gamma=0.9) | 8.36 ± 0.10 | 3.44 |
| ppo (gamma=0.99) | 8.41 ± 0.08 | 3.46 |
| linucb (fatigue-aware) | 11.70 ± 0.34 | 4.31 |
| oracle-myopic | 14.59 ± 0.16 | 5.06 |
| oracle-lookahead | 14.85 ± 0.47 | 5.11 |

Paired differences (same evaluation users per seed):
- ppo (gamma=0.99) minus linucb (tuned): +2.68 ± 0.17 (first wins on 3/3 seeds)
- linucb (fatigue-aware) minus ppo (gamma=0.99): +3.28 ± 0.30 (first wins on 3/3 seeds)
- linucb (fatigue-aware) minus linucb (tuned): +5.96 ± 0.36 (first wins on 3/3 seeds)
- ppo (gamma=0.99) minus ppo (gamma=0.9): +0.05 ± 0.17 (first wins on 2/3 seeds)
- oracle-lookahead minus oracle-myopic: +0.27 ± 0.50 (first wins on 2/3 seeds)

Share of the random-to-oracle-lookahead gap closed: PPO 40%, fatigue-aware LinUCB 70%.
