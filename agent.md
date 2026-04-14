# v2 Deep RL Agent Context

This file is the fast-start context for `game/v2_deep_rl/`.
It intentionally skips `game/v1_assignment/`.
Use this before rereading code.

## Scope

- Main product surface today:
  - `game/v2_deep_rl/control_center/`
- Core RL engine it wraps:
  - `game/v2_deep_rl/config_manager.py`
  - `game/v2_deep_rl/scrum_game_env.py`
  - `game/v2_deep_rl/cards.py`
  - `game/v2_deep_rl/refinements.py`
  - `game/v2_deep_rl/dqn_agent.py`
  - `game/v2_deep_rl/checkpoint_utils.py`
  - `game/v2_deep_rl/train_dqn.py`
  - `game/v2_deep_rl/evaluate_ddqn_robustness.py`
  - `game/v2_deep_rl/play_best_dqn_game.py`
  - `game/v2_deep_rl/match_runner.py`
- Older but still present tooling:
  - `game/v2_deep_rl/dashboard.py`
  - `game/v2_deep_rl/config_editor/`

## What v2 Is

- Config-driven Double DQN branch for the Scrum board game.
- Default board is the classical 7-product x 4-sprint setup.
- Training is still single-player RL.
- The action space is no longer binary:
  - `0 = Continue`
  - `1..7 = Switch to Product N`
- Checkpoints are rule-aware through config signatures, not just tensor shapes.

## Core Architecture

### 1. Config Layer

- `config_manager.py` is the source of truth for rules and training settings.
- `GameConfig` holds board, economy, dice, refinement, and incident rules.
- `TrainingConfig` holds training hyperparameters and logging cadence.
- `compute_rule_signature(game_config)` hashes rule-defining fields.
- `compute_training_signature(training_config)` hashes training-defining fields.
- `map_prototype_to_config()` converts prototype JSON into canonical config JSON.
- Default bundled configs live in:
  - `game/v2_deep_rl/configs/default_game_config.json`
  - `game/v2_deep_rl/configs/default_training_config.json`

### 2. Environment Layer

- `scrum_game_env.py` is the real gameplay simulator.
- It is fully driven by `GameConfig`.
- Important mechanics:
  - automatic mandatory loans
  - per-turn loan interest
  - configurable switch costs
  - configurable dice regimes by feature count
  - refinement effects on future feature counts
  - incident effects on future sprint values
- Reward is shaped, not raw money only:
  - base reward = `new_money - old_money`
  - extra debt penalty while loans are active
  - bonus for successful debt-free turns
  - small bonus for switching while poor if the turn resolves
  - heavy penalty when a mandatory loan is triggered
  - penalty for invalid actions
  - small shaping around refinement increase/decrease

### 3. Observation / Encoding Layer

- Environment state is a dict, then normalized in `dqn_agent.encode_state()`.
- Encoded state dimension under the default board is `82`.
- Layout:
  - `19` global/current-product features
  - `9` per-product features x `7` products = `63`
  - total `82`
- Global/current-product features include:
  - money
  - current product and sprint
  - features required
  - sprint value
  - loan / interest / debt ratio
  - win probability
  - expected value
  - remaining turns
  - incident/refinement state
- Per-product features include:
  - next sprint
  - required features
  - sprint value
  - win probability
  - expected value
  - completed flag
  - incident delta
  - refinement delta
  - incident flag

### 4. Agent / Checkpoint Layer

- `dqn_agent.py` implements:
  - replay buffer
  - 4-layer MLP: `82 -> 256 -> 256 -> 128 -> 8`
  - Double DQN target update logic
  - Huber loss
  - gradient clipping
- `checkpoint_utils.py` wraps:
  - agent shape creation from config
  - checkpoint save/load
  - legacy checkpoint normalization
  - strict signature validation vs fine-tune shape validation

### 5. Training Layer

- `train_dqn.py` is the training entrypoint.
- It creates a timestamped run directory under:
  - `game/v2_deep_rl/artifacts/runs/run_YYYY-MM-DD_HHMM[_name]`
- It persists:
  - `game_config.json`
  - `training_config.json`
  - `run_metadata.json`
  - `reports/logs.csv`
  - `reports/evaluation_history.csv`
  - `reports/dqn_metrics.json`
  - `checkpoints/checkpoint_episode_XXXXXX.pth`
  - `checkpoints/best_scrum_model.pth`
  - `plots/dqn_training_curve.png`
- Logging behavior:
  - training CSV appends every `100` episodes
  - periodic greedy evaluation runs every `evaluation_interval`
  - best checkpoint is chosen by evaluation average reward
- Resume behavior:
  - `strict` requires matching rule signature
  - `fine-tune` allows rule mismatch if tensor shape still matches

### 6. Evaluation / Play Layer

- `evaluate_ddqn_robustness.py` runs greedy fixed-seed evaluation over 5 seeds.
- `play_best_dqn_game.py` loads a checkpoint and plays a terminal demo with deployment profiles:
  - `beginner`
  - `balanced`
  - `expert`
- `match_runner.py` is the shared play/session engine:
  - human controller
  - random controller
  - heuristic controller
  - checkpoint-backed model controller
- Important: the "parallel match" is still one independent env per seat, not one shared multiplayer board.

## Default Ruleset

- Products: `Yellow, Blue, Red, Orange, Green, Purple, Black`
- Products count: `7`
- Sprints per product: `4`
- Max turns: `6`
- Starting money: `25000`
- Ring value: `5000`
- Switch-mid cost: `5000`
- Mandatory loan: `50000`
- Loan interest: `5000`
- Daily scrums per sprint: `5`
- Daily scrum target: `12`
- Dice rules:
  - `1 feature -> 1d20`
  - `2 features -> 2d10`
  - `3+ features -> 3d6`
- Incident deck currently implements only 5 source-backed cards.

## Control Center

### Backend

- Stack: FastAPI + SQLite + local subprocess jobs.
- Entry points:
  - `control_center/backend/app.py`
  - `control_center/backend/run_api.py`
- Main backend areas:
  - `services/catalog_service.py`
    - managed config assets
    - run catalog
    - run progress summaries
  - `services/checkpoint_service.py`
    - checkpoint library
    - compatibility checks
  - `services/play_service.py`
    - in-memory play sessions
  - `services/testing_service.py`
    - direct evaluation and checkpoint comparison
  - `services/training_autopilot.py`
    - post-run logic and optional AI advisory loop
  - `jobs/queue_manager.py`
    - one-worker queue
  - `jobs/job_runner.py`
    - subprocess launcher for train / robustness jobs
  - `storage/jobs_db.py`
    - SQLite job persistence

### Frontend

- `control_center/frontend/app.js` is a large vanilla-JS SPA.
- Pages:
  - `Design`
  - `Train`
  - `Inspect`
  - `Evaluate`
  - `Play`
- Main frontend responsibilities:
  - sidebar selection of active blueprint / training profile / brain
  - visual game-config editor plus raw JSON editor
  - compatibility checks before resume or fine-tune
  - live job and run inspection
  - direct checkpoint evaluation and comparison
  - play-session control
  - autopilot controls and history display
- The control center is meant to replace the split `dashboard.py` + `config_editor/` workflow.

## Legacy Surfaces

- `dashboard.py`
  - Streamlit command center
  - still useful for charts, heatmaps, and match demos
  - overlaps with the control center and should be treated as legacy tooling
- `config_editor/`
  - standalone browser editor that outputs canonical `GameConfig` JSON
  - largely duplicated inside the control-center `Design` page

## Autopilot

- File: `control_center/backend/services/training_autopilot.py`
- Logic behavior:
  - continue when epsilon is still high
  - lower learning rate when reward improves but variance is high
  - extend epsilon decay when reward plateaus and invalid actions stay high
  - stop on plateau otherwise
- AI behavior:
  - optional NVIDIA-hosted LLM advisor only after logic decides stop
  - bounded to learning rate / epsilon decay / episode-count suggestions
- State files:
  - `game/v2_deep_rl/artifacts/autopilot_settings.json`
  - `game/v2_deep_rl/artifacts/autopilot_stop_requested.flag`
  - per-run history:
    - `reports/autopilot_decisions.jsonl`

## Current Experiment Snapshot

Observed from local artifacts on `2026-04-14`:

- `run_2026-04-14_1603` is the clearest completed reference run.
- Its saved metrics:
  - `60000` training episodes
  - `average_reward_per_episode = -25783.5`
  - `average_ending_money = 34088.0`
  - `bankruptcy_rate = 0.407`
  - `invalid_action_rate = 0.001366...`
  - `best_intermediate_evaluation_reward = -17220.0`
- Evaluation history for that run is noisy:
  - `10000 -> -23560`
  - `20000 -> -37542.5`
  - `30000 -> -49705`
  - `40000 -> -37870`
  - `50000 -> -17220`
  - `60000 -> -40305`
- Autopilot later classified that run as "improving but noisy" and proposed `lower_lr`, then later switched to plain `continue` because epsilon was still high (`0.873`).
- Later autopilot-continue runs exist on `2026-04-14`, but the newest ones are incomplete and do not yet have final metrics.
- The flat `artifacts/reports/` snapshot is stale and misleading:
  - it shows a `1`-episode run
  - invalid-action rate is extremely high there
  - some paths still point at older `GitHub` clones instead of the current `GitClones` workspace

## Known Traps

- `training_autopilot.py` currently contains a hardcoded fallback `NVIDIA_API_KEY` in source.
  - This should be removed and replaced with env-only loading.
- Resume-mode naming is inconsistent across history and UI:
  - `fine_tune`
  - `fine-tune`
  - current trainer only accepts `fine-tune`
- Latest fine-tune stdout logs include:
  - `forrtl: error (200): program aborting due to window-CLOSE event`
  - this likely means some Windows process-management issue in detached runs
- Many stored artifact JSON files contain absolute paths from older machines or older clone roots.
  - Do not trust persisted absolute paths blindly.
- Play sessions are in-memory only.
  - restarting the backend drops them
- Job queue is single-worker by design.
  - parallel training jobs are not supported

## Best Next Ideas

- Fix the autopilot naming drift so every path uses one spelling for fine-tune mode.
- Remove the hardcoded NVIDIA API key and make AI advisory opt-in by environment variable.
- Investigate the Windows `forrtl` close-event crashes in late autopilot fine-tune runs.
- Add artifact-path normalization or relative-path storage so old clone roots do not poison current metadata.
- Split reward shaping analysis from policy analysis.
  - right now invalid actions improved a lot, but bankruptcy and reward are still poor
- Add richer periodic evaluation outputs:
  - per-seed results
  - action histograms
  - switch-target frequencies
  - success/failure by product and sprint
- Consider action masking or invalid-action-aware loss shaping.
  - the env penalizes invalid actions, but the policy still learns in a fully unconstrained action space
- Revisit exploration schedule.
  - autopilot is continuing because epsilon stays high for a long time
  - reward quality may be getting obscured by prolonged exploration
- Consolidate UI surfaces.
  - long-term keep the control center and retire duplicated logic in `dashboard.py` and `config_editor/`
- If multiplayer fidelity matters, redesign beyond one-env-per-seat demos.
  - current training and play abstractions are still fundamentally single-player

## Quick Commands

```powershell
cd game\v2_deep_rl
py train_dqn.py
py evaluate_ddqn_robustness.py
py play_best_dqn_game.py --profile expert
py -m streamlit run dashboard.py
python control_center\backend\run_api.py
```
