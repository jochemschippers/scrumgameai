# v1 Assignment Track

This folder is the clean assignment-safe version of the project.

It contains:
- the classical Scrum Game environment
- the heuristic baseline
- tabular Q-Learning
- tabular SARSA
- tabular Monte Carlo
- comparison, tuning, robustness, and demo scripts

## Main Files

- `scrum_game_env.py`
  Environment and `discretize_state(state)` helper.
- `compare_models.py`
  Fair comparison across Baseline, Q-Learning, SARSA, and Monte Carlo.
- `evaluate_robustness.py`
  Multi-seed tabular robustness evaluation.
- `train_sarsa.py`
  Long-haul SARSA training plus final model export.
- `play_final_game.py`
  Demo runner for the saved SARSA model.

## Run Order

```powershell
py compare_models.py
py evaluate_robustness.py
py train_sarsa.py
py play_final_game.py
```

## Artifacts

- `artifacts/models/`
  Saved Q-tables and `final_sarsa_model.json`
- `artifacts/plots/`
  Training curves and comparison plots
- `artifacts/reports/`
  CSV, JSON, and Markdown outputs for the report
- `docs/report_notes.md`
  Report-writing notes for the assignment track

The advanced DQN work is intentionally excluded from this folder and lives in `../v2_deep_rl`.
