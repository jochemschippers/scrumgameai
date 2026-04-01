# Game Workspace

The `game/` folder is split into two separate tracks so the assignment work stays clean:

- `v1_assignment/`
  Assignment-safe tabular RL work. This is the version to study first if you need the deliverable code, comparison scripts, and report notes.
- `v2_deep_rl/`
  Advanced PyTorch DQN work, Streamlit dashboard, and deep-RL experiments.

## Why The Split Exists

The assignment track and the deep-RL track solve related problems, but they do not serve the same purpose:
- `v1_assignment` is organized for the course deliverable and keeps the model set simple and explainable.
- `v2_deep_rl` is the stronger experimental branch with extra engineering and a neural-network agent.

Both tracks keep their own copy of the environment and their own artifacts so changes in one do not silently break the other.

## Where To Start

For the assignment:

```powershell
cd game\v1_assignment
py compare_models.py
```

For the advanced DQN branch:

```powershell
cd game\v2_deep_rl
py play_best_dqn_game.py
```
