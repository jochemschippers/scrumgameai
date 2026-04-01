# Deliverable B Report Notes

## Scope

These notes belong to the assignment-safe track in `game/v1_assignment`.

This version keeps the assignment deliverable separate from the advanced DQN work in `../v2_deep_rl`. The models discussed here are:
- Baseline heuristic
- Q-Learning
- SARSA
- Monte Carlo

## Baseline Performance

The baseline agent uses a simple fixed heuristic:
- Always continue the current sprint.

This is a valid baseline because it represents a non-learning rule-based policy that can be implemented without RL. It gives a minimum benchmark that the learned tabular agents should beat.

Use these outputs in the report:
- `artifacts/reports/model_comparison.csv`
- `artifacts/reports/model_comparison_summary.md`

## Models Compared

This assignment track compares four policies:
- Baseline heuristic agent
- Q-Learning agent
- SARSA agent
- Monte Carlo agent

That satisfies the requirement to compare at least three learning approaches.

## Model Choice Justification

### Q-Learning

Q-Learning was selected because it is the standard off-policy temporal-difference method.

Pros:
- Simple to implement
- Efficient tabular baseline
- Common reference algorithm for RL work

Cons:
- Can become unstable in stochastic environments
- Can overreact to noisy outcomes if the learning rate is too aggressive

### SARSA

SARSA was selected because it is on-policy and often behaves more conservatively in risky environments.

Pros:
- Can learn safer behavior under randomness
- Uses the next action actually chosen by the current policy

Cons:
- Can learn more slowly than Q-Learning
- Performance depends heavily on the exploration schedule

### Monte Carlo

Monte Carlo was selected to compare full-episode return learning against temporal-difference methods.

Pros:
- Easy to explain conceptually
- Learns from complete observed returns

Cons:
- Must wait until the full episode ends before updating
- Can have high variance

## Implementation Explanation

The environment follows a Gym-style interface with:
- `__init__`
- `reset()`
- `step(action)`

The state includes:
- current money
- current product
- current sprint
- features required
- sprint value
- loan status
- interest due
- win probability for the current sprint

Because tabular RL cannot handle large raw money values efficiently, the project uses `discretize_state(state)` to convert the raw environment state into a compact hashable tuple. That tuple is used as the key in each Q-table.

## Hyperparameters

For the main assignment comparisons:
- Q-Learning: `alpha = 0.05`, `gamma = 0.95`
- SARSA: `alpha = 0.05`, `gamma = 0.95` in fair comparison runs
- Monte Carlo: `alpha = 0.05`, `gamma = 0.95`
- Training episodes for comparison runs: `25,000`
- Evaluation episodes: `1,000`
- Epsilon decays from `1.0` to `0.05`

Separate long-haul SARSA production runs can still use the 100,000-episode script settings, but the comparison table should keep the learning budget consistent across the tabular agents.

## Strict Evaluation Isolation

This project follows the RL evaluation rules from the assignment:
- training episodes are used for learning
- evaluation episodes are used only for measurement
- during evaluation, `epsilon = 0`
- during evaluation, `learn()` is never called

That avoids leakage between training and evaluation.

## Robustness Discussion

The Scrum Game is stochastic because sprint outcomes depend on dice rolls. That means reward variance is expected.

Key points to document:
- Report mean reward and spread, not only one score
- A single run can be lucky or unlucky
- Multi-seed evaluation is a stronger robustness check

The tabular robustness script is:
- `evaluate_robustness.py`

## Overfitting And Underfitting

In RL, overfitting often appears as instability rather than classic train/test divergence.

Possible warning signs:
- Flat learning curves can indicate underfitting
- Curves that improve and then collapse can indicate instability or overreaction to noisy rewards

This happened earlier in Q-Learning and was improved by lowering the learning rate and extending training.

## Bias Discussion

The environment includes design assumptions that shape what the agent learns:
- loan size
- loan interest
- failure penalties
- sprint value distribution
- dice success probabilities

Because of that, the learned policy is best interpreted as optimal for this simulator configuration, not for every possible Scrum setting.

## Artifacts

Saved tabular model artifacts:
- `artifacts/models/q_learning_q_table.json`
- `artifacts/models/sarsa_q_table.json`
- `artifacts/models/mc_q_table.json`
- `artifacts/models/final_sarsa_model.json`

Saved plots:
- `artifacts/plots/q_learning_training_curve.png`
- `artifacts/plots/sarsa_training_curve.png`
- `artifacts/plots/mc_training_curve.png`
- `artifacts/plots/model_comparison.png`

Saved comparison outputs:
- `artifacts/reports/model_comparison.csv`
- `artifacts/reports/model_comparison.json`
- `artifacts/reports/model_comparison_summary.md`

## Final Assignment Track Model

Use `compare_models.py` and `evaluate_robustness.py` to decide which tabular model you want to present in the assignment discussion.

If you want to mention the advanced DQN extension, present it as separate follow-up work in `../v2_deep_rl`, not as part of this assignment-safe comparison table.
