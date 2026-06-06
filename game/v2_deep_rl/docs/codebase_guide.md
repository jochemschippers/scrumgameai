# Deep RL Codebase Guide

This guide explains where behavior belongs and how the main runtime paths fit
together. Source comments focus on constraints and non-obvious choices; this
document provides the broader map.

## Package Responsibilities

| Package | Responsibility |
| --- | --- |
| `config/` | Configuration dataclasses, JSON conversion, validation, and signatures |
| `game_rules/` | Incident cards, refinement rules, and randomized rule sets |
| `game_runtime/` | Stateful Scrum Game environment used by agents and matches |
| `rl/` | DQN network, replay memory, checkpoint serialization, and model helpers |
| `training/` | Training CLI, episode loop, evaluation, logging, and artifacts |
| `evaluation/` | Multi-seed robustness evaluation |
| `play/` | Controllers and private or shared-board match orchestration |
| `dashboard_app/` | Read-only Streamlit views over training artifacts |
| `control_center/backend/` | API routes, jobs, catalogs, campaigns, and autopilot |
| `control_center/frontend/` | ES-module browser client for the Control Center API |
| `config_editor/` | Standalone visual editor for game configuration JSON |

`v1_assignment/` is intentionally independent. Do not import v2 code into it;
the assignment track must remain reproducible without the deep-RL stack.

## Training Flow

1. `training.train_dqn` loads and validates game and training configurations.
2. `ScrumGameEnv` owns mutable episode state and calculates observations/rewards.
3. `DQNAgent` selects actions, stores transitions, and updates the online network.
4. Periodic evaluation selects the best checkpoint.
5. Each run writes configs, metadata, CSV reports, plots, and checkpoints beneath
   its run directory.

Checkpoint sidecar JSON files are important. The web API reads sidecars to avoid
loading large PyTorch replay buffers merely to display catalog metadata.

## Control Center Flow

`backend/app.py` assembles FastAPI routes and serves the frontend. Route modules
validate HTTP inputs and delegate work to services. Services own domain behavior;
`storage/jobs_db.py` owns job persistence.

Training and evaluation execute in subprocesses:

1. `jobs/queue_manager.py` persists a queued job and starts one worker at a time.
2. `jobs/job_runner.py` translates the job payload into a training/evaluation call.
3. Progress endpoints read generated CSV files instead of sharing process memory.
4. Completion may trigger autopilot, campaign progression, or robustness evaluation.

This process boundary keeps PyTorch out of the lightweight web-server environment.

## Autopilot Flow

Autopilot is split so each layer has one kind of side effect:

- `autopilot/analysis.py` is deterministic and read-only.
- `autopilot/advisor.py` optionally asks an LLM for a bounded adjustment.
- `autopilot/runner.py` applies overrides, writes history, and queues continuation.
- `autopilot/state.py` stores settings and the user stop-after-cycle request.

The continuation context travels in job payloads because each cycle is a new
process and run directory. Regression stops bypass the LLM guardrail.

## Shared Match State

Each seat has private finances, progress, and an environment instance. The board,
refinement changes, incident effects, and incident deck are synchronized through
the match-level `board_state`. `play/shared_match_runner.py` temporarily injects
that shared state into the acting environment and captures it again after a turn.

Incidents resolve once after all seats act. Their effects update shared future
board cells, while each seat's private sprint progress is restored afterward.

## Frontend State

The Control Center uses native browser modules:

- `main.js` owns startup and cross-feature event wiring.
- `state/store.js` exports the single mutable state object.
- `api/client.js` is the only module that should perform authenticated fetches.
- `components/` modules render and operate one feature area.
- `utils/` modules should stay independent of feature-specific behavior.

Avoid imports from low-level utilities back into components. They create circular
dependencies and can make browser module startup fail.

## Commenting Standard

Add comments for:

- business rules that are not obvious from code;
- safety bounds and compatibility behavior;
- process, cache, or shared-state constraints;
- deliberate fallbacks and non-fatal failures.

Do not comment simple assignments, loops, or function calls. Prefer a descriptive
name or a short docstring stating the function's contract.

## Verification

From `game/v2_deep_rl`:

```powershell
py -m pytest tests -q
```

For manual Control Center checks:

```powershell
cd control_center\backend
py run_api.py
```

Then open `http://127.0.0.1:8000/` and check the browser console for module or API
errors.
