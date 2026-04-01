# Scrum Game AI

This repository contains a headless Reinforcement Learning prototype for the digital Scrum Game NPC.

The project builds a Gym-style environment for the game, trains several tabular RL agents, compares them against a heuristic baseline, and exports a final SARSA model for demo play.

## What This Project Does

The code simulates the Scrum Game as a turn-based environment with:
- a fixed classical `7 x 4` board
- real money values from the game data
- 5 Daily Scrum dice resolution per sprint
- automatic mandatory loans
- two actions:
  - `0` = Continue Sprint
  - `1` = Switch Product

The RL agents learn a Q-table over a discretized version of the game state.

## Project Structure

- [game/scrum_game_env.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/scrum_game_env.py)
  Gym-style environment and `discretize_state(state)` helper.
- [game/baseline_agent.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/baseline_agent.py)
  Simple non-learning baseline agent.
- [game/q_learning_agent.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/q_learning_agent.py)
  Tabular Q-Learning agent.
- [game/sarsa_agent.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/sarsa_agent.py)
  Tabular SARSA agent.
- [game/mc_agent.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/mc_agent.py)
  Tabular Monte Carlo agent.
- [game/train_q_learning.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/train_q_learning.py)
  Train and evaluate Q-Learning.
- [game/train_sarsa.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/train_sarsa.py)
  Train and evaluate SARSA, and save the final production model.
- [game/train_mc.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/train_mc.py)
  Train and evaluate Monte Carlo.
- [game/compare_models.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/compare_models.py)
  Compare baseline, Q-Learning, SARSA, and Monte Carlo in one run.
- [game/evaluate_robustness.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/evaluate_robustness.py)
  Evaluate all models across multiple random seeds.
- [game/tune_sarsa.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/tune_sarsa.py)
  Grid search for the best SARSA discount factor.
- [game/play_final_game.py](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/play_final_game.py)
  Demo script that loads the final SARSA model and plays one full game.

## How The Environment Works

The environment follows the standard Gym structure:
- `__init__`
- `reset()`
- `step(action)`

The state contains:
- `current_money`
- `current_product`
- `current_sprint`
- `features_required`
- `sprint_value`
- `loan_active`
- `interest_due`

### Board

The current environment uses the classical fixed board:
- `7` products
- `4` sprints per product
- `6` turns per episode

Each board cell has:
- a fixed feature count
- a fixed sprint value in rings, converted to money with `ring_value = 5000`

### Dice Resolution

Each sprint is resolved with exactly `5` Daily Scrums:
- `1 feature` -> `1 x D20`
- `2 features` -> `2 x D10`
- `3+ features` -> `3 x D6`

For each Daily Scrum:
- roll the dice
- sum the result
- compare against target `12`
- accumulate the difference

Sprint outcome:
- `net <= 0` -> success
- `net > 0` -> failure

### Economy

Classical default values:
- starting money: `25000`
- switch cost: `5000`
- ring value: `5000`
- mandatory loan amount: `50000`
- loan interest: `5000`

Loans are automatic. The agent cannot actively choose them.

## How The Agents Work

All learning agents use:
- a dictionary-based Q-table
- a discretized state tuple as the table key
- epsilon-greedy action selection

Action space:
- `0` = Continue Sprint
- `1` = Switch Product

### Baseline Agent

The baseline is intentionally simple:
- always returns action `0`
- never learns

### Q-Learning

Q-Learning is an off-policy temporal-difference method.

It updates:
```text
Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max(Q(s')) - Q(s, a)]
```

### SARSA

SARSA is an on-policy temporal-difference method.

It updates:
```text
Q(s, a) <- Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]
```

### Monte Carlo

Monte Carlo updates after the full episode finishes.

It learns from the discounted return of complete state-action-reward histories.

## Training And Evaluation Rules

This project uses strict RL evaluation isolation:
- training episodes are used for learning
- evaluation episodes are used only for measurement
- during evaluation, `epsilon = 0`
- during evaluation, `learn()` is never called

This avoids data leakage between training and evaluation.

## How To Run

Run all commands from the `game` folder:

```powershell
cd game
```

### Baseline

```powershell
py baseline_agent.py
```

### Train Q-Learning

```powershell
py train_q_learning.py
```

### Train SARSA

```powershell
py train_sarsa.py
```

This script is the current production SARSA training run. It:
- trains for `100000` episodes
- uses `gamma = 0.85`
- evaluates the final policy
- saves `final_sarsa_model.json`

### Train Monte Carlo

```powershell
py train_mc.py
```

### Compare All Models

```powershell
py compare_models.py
```

This generates:
- `artifacts/reports/model_comparison.csv`
- `artifacts/reports/model_comparison.json`
- `artifacts/reports/model_comparison_summary.md`
- `artifacts/plots/model_comparison.png`

### Robustness Evaluation

```powershell
py evaluate_robustness.py
```

This trains and evaluates all models across multiple seeds and reports:
- mean average reward
- standard deviation across seeds

### Tune SARSA

```powershell
py tune_sarsa.py
```

This runs a grid search over:
- `gamma = [0.85, 0.90, 0.95, 0.99]`

### Demo The Final Model

```powershell
py play_final_game.py
```

This loads `final_sarsa_model.json` and plays one full greedy demo game with:
- `epsilon = 0`
- no random exploration
- turn-by-turn printed summaries

## Saved Outputs

Generated files are organized as follows:

- `game/artifacts/models/`
  Saved Q-tables from training and comparison runs.
- `game/artifacts/plots/`
  Training curves and model comparison plots.
- `game/artifacts/reports/`
  CSV, JSON, and Markdown summaries for the report.
- `game/docs/`
  Project notes and report support material.

## Current Final Model

The current final production demo model is:
- [final_sarsa_model.json](/c:/Users/joche/OneDrive/Documenten/GitClones/scrumgameai/game/final_sarsa_model.json)

It is trained from the SARSA agent and intended for project demonstration.

## Data Sources

The simulator and game constants were based on the files in:
- `gamedata/`

Important sources include:
- physical boardgame rule example
- simulation setup variables
- interactive game editor prototype
- project analysis and planning documents

## Notes

This is a tabular RL prototype, not a neural-network-based model.

The current environment is a classical-setup baseline. It captures the core Scrum Game mechanics, but future work can make it more configurable for:
- different board layouts
- incidents
- refinements
- difficulty profiles
- API integration
