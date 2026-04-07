# v2 Deep RL Track

This folder contains the advanced deep-RL branch of the Scrum Game project.

It is intentionally separate from `../v1_assignment` so the deep-RL experiments do not pollute the assignment-safe code and artifacts.

## Main Files

- `scrum_game_env.py`
  Advanced environment with explicit product-switch actions, a rule-backed incident/refinement flow, and richer observations.
- `cards.py`
  Incident deck definitions and draw/discard mechanics based on the documented physical cards.
- `refinements.py`
  Standard refinement model based on the documented D20 rules.
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
py -m streamlit run dashboard.py
py play_best_dqn_game.py
```

After the 8-action refactor, the old binary-action checkpoint is only kept as a frozen benchmark in `artifacts/reference_v1/`. A fresh training run is required before the new demo and dashboard can load a compatible checkpoint.

Important simplification:
- The advanced branch is still single-player for RL training, so "incident after each round" is modeled as "incident after each turn" because one environment episode tracks one player.

## What Is Implemented

- action space: `0 = Continue`, `1..7 = Switch to Product N`
- exact 5 Daily Scrum sprint resolution
- classical 7x4 board and real economy values
- incident deck module with the cards explicitly documented in the provided manuals
- Standard refinement model `301`
- richer observation including `win_probability`, `expected_value`, remaining-turn context, debt burden, incident state, and per-product target summaries
- invalid action logging in training and evaluation outputs
- Double DQN training with checkpoint selection from periodic evaluation

Important source caveat:
- the setup PDFs mention 8 incident card slots, but the provided manuals only show 5 concrete incident cards clearly enough to implement faithfully
- those 5 cards are implemented; the remaining 3 are intentionally not invented

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
