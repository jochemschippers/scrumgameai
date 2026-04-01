# Assignment Track Model Comparison

## Scope
- This comparison belongs to `game/v1_assignment` and keeps the assignment-safe tabular work separate from the advanced DQN experiments.
- The compared policies are Baseline, Q-Learning, SARSA, and Monte Carlo.
- The advanced DQN work lives separately in `../v2_deep_rl` and is intentionally excluded from this table.

## Evaluation Protocol
- Training and evaluation were separated into different episode phases.
- During evaluation, epsilon was fixed at 0.0 for pure exploitation.
- During evaluation, no learning updates were allowed.
- The baseline agent used a fixed heuristic and did not learn.

## Results Table

| Model | Avg Reward | Std Dev | Best | Worst | Q-Table Size |
|---|---:|---:|---:|---:|---:|
| Q-Learning | 1500.00 | 0.00 | 1500.00 | 1500.00 | 151 |
| SARSA | 1500.00 | 0.00 | 1500.00 | 1500.00 | 96 |
| Monte Carlo | 1500.00 | 0.00 | 1500.00 | 1500.00 | 88 |
| Baseline | -65810.00 | 84221.37 | 69000.00 | -235000.00 | N/A |

## Hyperparameter Summary
- Q-Learning: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.
- SARSA: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes for this fair comparison run.
- Monte Carlo: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.
- Baseline: no learning, fixed always-continue heuristic.

## Model Choice Justification
- Q-Learning is the standard off-policy temporal-difference baseline.
- SARSA is included because on-policy updates can produce safer behavior in stochastic environments.
- Monte Carlo is included to contrast full-episode return learning with temporal-difference updates.
- The heuristic baseline shows whether learned behavior is better than a simple hand-written policy.

## Notes
- In this comparison run, the strongest assignment-track model was Q-Learning.
- If you also want to discuss the advanced DQN extension, treat it as a separate `v2_deep_rl` experiment rather than part of the assignment-safe table.