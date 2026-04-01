# Scrum Game AI

This repository contains two clearly separated versions of the Scrum Game AI project:

- `game/v1_assignment/`
  The clean assignment-safe version with tabular RL models and report outputs.
- `game/v2_deep_rl/`
  The advanced PyTorch DQN version with dashboard tooling and deep-RL experiments.

The split is intentional. The assignment work stays stable and easy to explain, while the deep-RL work can evolve without turning `game/` into one long flat list of unrelated scripts.

## Context From `gamedata/`

The environment logic is based on the project material in `gamedata/`, especially:
- the interactive prototype in `gamedata/Prototype - Game editor/Final Interactive prototype/`
- the simulation setup PDF
- the data analysis report
- the Big Data & AI notes

Those files define the classical board assumptions used in the simulator:
- fixed `7 x 4` board
- `25000` starting money
- `5000` ring value
- `5000` switch cost
- `50000` mandatory loan
- `5000` loan interest
- 5 Daily Scrums per sprint

## Repository Layout

- `gamedata/`
  Source material, PDFs, and the prototype app used to shape the simulator assumptions.
- `game/README.md`
  Short navigation guide for the code workspace.
- `game/v1_assignment/`
  Baseline, Q-Learning, SARSA, Monte Carlo, comparison scripts, report notes, and tabular artifacts.
- `game/v2_deep_rl/`
  DQN agent, DQN trainer, Streamlit dashboard, deep-RL notes, and neural-network artifacts.
- `lesson.md`
  Assignment and environment instructions.
- `agent.md`
  Coding standards and project guidance.

## Quick Start

Assignment track:

```powershell
cd game\v1_assignment
py compare_models.py
```

Deep-RL track:

```powershell
cd game\v2_deep_rl
py play_best_dqn_game.py
```

## v1 Assignment Track

Use `game/v1_assignment` if you need the deliverable-friendly version.

Main scripts:
- `compare_models.py`
  Compares Baseline, Q-Learning, SARSA, and Monte Carlo.
- `evaluate_robustness.py`
  Runs multi-seed tabular robustness checks.
- `train_sarsa.py`
  Long-haul SARSA training plus saved model export.
- `play_final_game.py`
  Demo runner for the saved SARSA model.

Artifacts are stored inside `game/v1_assignment/artifacts/`.

## v2 Deep RL Track

Use `game/v2_deep_rl` if you want the stronger experimental agent and dashboard tooling.

Main scripts:
- `train_dqn.py`
  Trains the DQN with replay memory and checkpointing.
- `dashboard.py`
  Streamlit dashboard for live training diagnostics.
- `play_best_dqn_game.py`
  Greedy DQN demo runner.

Artifacts are stored inside `game/v2_deep_rl/artifacts/`.

## Notes On Separation

The two version folders intentionally keep separate:
- environment copies
- model code
- reports
- saved artifacts

That makes it easier to present the assignment work without mixing it with the later deep-RL branch.
