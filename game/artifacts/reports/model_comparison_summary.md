# Model Comparison Summary

## Evaluation Protocol
- Training and evaluation were separated into different episode phases.
- During evaluation, epsilon was fixed at 0.0 for pure exploitation.
- During evaluation, no learning updates were allowed.
- The baseline agent used a fixed heuristic and did not learn.

## Results Table

| Model | Avg Reward | Std Dev | Best | Worst | Q-Table Size |
|---|---:|---:|---:|---:|---:|
| SARSA | -15664.00 | 27673.73 | 20000.00 | -187000.00 | 98 |
| Monte Carlo | -16778.00 | 26615.23 | 10000.00 | -160000.00 | 86 |
| Q-Learning | -18732.00 | 23621.39 | -5000.00 | -212000.00 | 159 |
| Baseline | -73868.00 | 81760.47 | 57000.00 | -241000.00 | N/A |

## Hyperparameter Summary
- Q-Learning: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.
- SARSA: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.
- Monte Carlo: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.
- Baseline: no learning, fixed heuristic policy.

## Model Choice Justification
- Q-Learning was included as a standard off-policy temporal-difference baseline.
- SARSA was included because its on-policy updates can produce safer behavior in stochastic environments.
- Monte Carlo was included to compare full-episode return learning against temporal-difference methods.
- The heuristic baseline was included to show whether learned behavior outperformed a simple rule-based policy.

## Robustness And Limitations
- The Scrum Game environment is stochastic because sprint success depends on random outcomes, so reward variance is expected.
- Large reward standard deviations indicate that policy performance is sensitive to chance and should be discussed as a robustness limitation.
- If a training curve improves and then degrades late in training, that suggests instability or partial overfitting to recent stochastic experiences.
- A practical mitigation is repeated training over multiple random seeds and reporting the mean and spread across runs.

## Bias Discussion
- The environment encodes design assumptions such as loan penalties, sprint values, and failure penalties.
- These assumptions can bias which strategies appear optimal, so results should be interpreted as valid for this game design rather than all Scrum settings.
- If environment reward settings change, the ranking of agents may also change.

## Current Best Model
- Based on the latest comparison run, the best average evaluation reward was achieved by SARSA.