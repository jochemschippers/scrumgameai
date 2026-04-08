# Unified Scrum Game Control Center

This folder is the clean application shell for the custom web app that will replace the split `config_editor` and Streamlit dashboard workflow.

## Structure

- `backend/`
  - Python API layer over the existing RL engine
  - job queue, config management, compatibility checks, play sessions
- `frontend/`
  - custom multi-page web UI
  - rules editor, models and runs, training, testing, play
- `shared/`
  - app-level shared assets and metadata contracts

## Boundaries

The current RL engine remains in the parent `game/v2_deep_rl/` directory:

- `train_dqn.py`
- `evaluate_ddqn_robustness.py`
- `checkpoint_utils.py`
- `config_manager.py`
- `match_runner.py`
- `scrum_game_env.py`

This `control_center` folder is the new product surface, not a replacement for the core RL modules.
