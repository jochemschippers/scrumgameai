# Frozen v1 Reference Snapshot

This folder preserves the pre-refactor deep-RL benchmark so the advanced 8-action branch can be compared against a known-good reference.

Saved files:
- `best_scrum_model_v1.pth`
- `dqn_metrics_v1.json`
- `dqn_training_curve_v1.png`
- `logs_v1.csv`

This snapshot corresponds to the earlier binary-action DQN setup before:
- explicit product-switch actions
- richer state representation
- visible incidents/refinements
- Double DQN updates

Use it only as a comparison point. It is not compatible with the new 8-action model architecture.
