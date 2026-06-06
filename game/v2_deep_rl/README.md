# v2 Deep RL Track

This folder contains the advanced deep-RL branch of the Scrum Game project.

It is intentionally separate from `../v1_assignment` so the deep-RL experiments do not pollute the assignment-safe code and artifacts.

## Source Layout

The old top-level filenames still exist as thin compatibility wrappers, so existing commands and imports continue to work.

- `config/`
  Config dataclasses, JSON loading/saving, validation, signatures, and prototype mapping.
- `game_rules/`
  Incident cards, refinement rules, and rule-randomization helpers.
- `game_runtime/`
  The `ScrumGameEnv` environment and state discretization.
- `rl/`
  Double DQN agent, checkpoint utilities, and model metric helpers.
- `training/`
  Training loop, logging, evaluation during training, plotting, and CLI parsing.
- `evaluation/`
  Multi-seed checkpoint robustness evaluation.
- `play/`
  Deployment profiles, parallel/shared match runners, and the saved-model demo CLI.
- `dashboard_app/`
  Streamlit command center for training curves, action frequencies, switch-target heatmaps, and demo playback.
- `control_center/`
  FastAPI backend and browser UI for configs, jobs, campaigns, evaluation, and shared play.
- `config_editor/`
  Standalone browser editor for game configuration JSON.
- `docs/codebase_guide.md`
  Architecture, runtime flows, state ownership, and guidance for new contributors.

## Run Order

```powershell
py -m training.train_dqn
py -m streamlit run dashboard_app/dashboard.py
py -m play.play_best_dqn_game
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
