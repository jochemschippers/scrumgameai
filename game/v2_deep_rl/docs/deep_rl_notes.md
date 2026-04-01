# Deep RL Notes

This folder documents the advanced deep-RL branch separately from the assignment-safe tabular work.

## Frozen Reference Snapshot

Before the 8-action refactor, the earlier binary-action DQN branch was frozen into:
- `artifacts/reference_v1/best_scrum_model_v1.pth`
- `artifacts/reference_v1/dqn_metrics_v1.json`
- `artifacts/reference_v1/dqn_training_curve_v1.png`
- `artifacts/reference_v1/logs_v1.csv`

This preserves the old benchmark so the new branch can be compared against a known-good reference.

## Current Advanced Model Path

The advanced model checkpoint will be:
- `artifacts/checkpoints/best_scrum_model.pth`

Why it is kept separate:
- it uses a neural network instead of a tabular Q-table
- it depends on PyTorch and a replay buffer
- it includes Streamlit monitoring and checkpointing infrastructure

## Current Advanced Setup

- learning rate: `0.0005`
- gamma: `0.85`
- replay buffer size: `100000`
- training episodes: `500000`
- checkpoint interval: `10000`
- periodic evaluation log for checkpoint selection
- action space: `0 = Continue`, `1..7 = Switch to Product N`

## Supporting Features

- shaped rewards for debt pressure and prudent switching
- `win_probability`, `expected_value`, remaining-turn context, debt burden, and switch-cost context in the observation
- visible incidents and refinements in the environment transitions
- strategy heatmap, switch-target heatmap, and live dashboard diagnostics
- Beginner, Balanced, and Expert deployment profiles

## Demo

Run:

```powershell
py play_best_dqn_game.py
```

This loads the latest compatible checkpoint and plays one demo game. A fresh training run is required after the 8-action refactor.
