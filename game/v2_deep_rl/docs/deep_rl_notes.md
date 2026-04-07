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
- current encoded state dimension after the rule-fidelity upgrade: `82`

## Supporting Features

- shaped rewards for debt pressure and prudent switching
- `win_probability`, `expected_value`, remaining-turn context, debt burden, and switch-cost context in the observation
- incident deck support in `cards.py`
- standard refinement rules in `refinements.py`
- visible incident and refinement effects in the environment transitions
- invalid action tracking in both training logs and evaluation logs
- strategy heatmap, switch-target heatmap, and live dashboard diagnostics
- Beginner, Balanced, and Expert deployment profiles

## Rule Fidelity Notes

The current advanced simulator is closer to the physical game than the earlier branch, but it still has one deliberate simplification:
- RL training is single-player, so the manual rule "draw one incident after each round" is approximated as "draw one incident after each turn"

What is now source-backed:
- 5 Daily Scrums per sprint
- real board values and starting economy
- refinement thresholds by product group
- incident cards that modify future sprint values

Current implemented incident set:
- `401` Demand Collapse Red
- `402` New Competitors Orange/Blue
- `403` Government Subsidy First Sprints
- `404` Yellow Demand Boost
- `405` Black Product Breakthrough

Source caveat:
- simulator setup documents mention `8` incident cards
- the provided rules/examples only expose `5` concrete cards clearly enough to implement without guessing
- the advanced branch therefore implements those `5` cards faithfully and leaves the missing `3` undefined on purpose

What still remains a simplification:
- only the incident cards explicitly visible in the provided manuals are implemented
- the project still trains one player in isolation, not a full multiplayer table

## Demo

Run:

```powershell
py play_best_dqn_game.py
```

This loads the latest compatible checkpoint and plays one demo game.

After the observation refactor, old checkpoints from earlier DQN versions are incompatible. A fresh training run is required whenever the environment changes the encoded state layout.
