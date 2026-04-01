# Model Comparison Summary

## Evaluation Protocol
- Training and evaluation were separated into different episode phases.
- During evaluation, epsilon was fixed at 0.0 for pure exploitation.
- During evaluation, no learning updates were allowed.
- The baseline agent used a fixed heuristic and did not learn.

## Results Table

| Model | Avg Reward | Std Dev | Best | Worst | Q-Table Size |
|---|---:|---:|---:|---:|---:|
| Q-Learning | 1500.00 | 0.00 | 1500.00 | 1500.00 | 151 |
| SARSA | 1500.00 | 0.00 | 1500.00 | 1500.00 | 102 |
| Monte Carlo | 1500.00 | 0.00 | 1500.00 | 1500.00 | 88 |
| DQN | -13559.00 | 77325.77 | 79000.00 | -250000.00 | N/A |
| Baseline | -65810.00 | 84221.37 | 69000.00 | -235000.00 | N/A |

## Hyperparameter Summary
- Q-Learning: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.
- SARSA: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.
- Monte Carlo: alpha=0.05, gamma=0.95, epsilon decayed from 1.0 to 0.05 over 25,000 training episodes.
- DQN: learning_rate=0.0005, gamma=0.85, replay_buffer=100,000, epsilon decayed slowly over 400,000 of 500,000 training episodes.
- Baseline: no learning, fixed heuristic policy.

## Model Choice Justification
- Q-Learning was included as a standard off-policy temporal-difference baseline.
- SARSA was included because its on-policy updates can produce safer behavior in stochastic environments.
- Monte Carlo was included to compare full-episode return learning against temporal-difference methods.
- DQN was included to test whether a neural network with replay memory could learn a stronger policy than the tabular methods.
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
- Based on the latest comparison run, the best average evaluation reward was achieved by Q-Learning.

## Selected Deployment Model
- The final selected deployment model for the project is DQN.
- The saved deployment artifact is `artifacts/deep_rl/checkpoints/best_scrum_model.pth`.
- DQN was selected because it is the final deep-RL production model with dashboard support, checkpointing, and a dedicated demo runner.