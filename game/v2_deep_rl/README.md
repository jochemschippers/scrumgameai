# v2 Deep RL Track

This folder contains the advanced deep-RL branch of the Scrum Game project.

It is intentionally separate from `../v1_assignment` so the deep-RL experiments do not pollute the assignment-safe code and artifacts.

## Main Files

- `scrum_game_env.py`
  Advanced environment with explicit product-switch actions, visible incidents/refinements, and richer observations.
- `dqn_agent.py`
  PyTorch Double DQN implementation.
- `train_dqn.py`
  Long training run with checkpointing, periodic evaluation, and CSV logging.
- `dashboard.py`
  Streamlit command center for training curves, action frequencies, switch-target heatmaps, and demo playback.
- `play_best_dqn_game.py`
  Demo runner for the saved DQN checkpoint.
- `deployment_profiles.py`
  Beginner, Balanced, and Expert inference profiles.
- `evaluate_ddqn_robustness.py`
  Multi-seed robustness evaluation for the advanced branch.

## Run Order

```powershell
py train_dqn.py
streamlit run dashboard.py
py play_best_dqn_game.py
```

After the 8-action refactor, the old binary-action checkpoint is only kept as a frozen benchmark in `artifacts/reference_v1/`. A fresh training run is required before the new demo and dashboard can load a compatible checkpoint.

## Artifacts

- `artifacts/checkpoints/`
  Saved DQN checkpoints such as `best_scrum_model.pth`
- `artifacts/plots/`
  Training curves
- `artifacts/reports/`
  `logs.csv`, `evaluation_history.csv`, and metrics JSON
- `artifacts/reference_v1/`
  Frozen snapshot of the pre-refactor benchmark
- `docs/deep_rl_notes.md`
  Notes about the selected deep-RL model
