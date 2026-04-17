# Graph Report - .  (2026-04-17)

## Corpus Check
- 134 files · ~170,927 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1458 nodes · 3386 edges · 31 communities detected
- Extraction: 70% EXTRACTED · 30% INFERRED · 0% AMBIGUOUS · INFERRED: 1005 edges (avg confidence: 0.67)
- Token cost: 19,700 input · 5,150 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Game Environment and RL Agents|Game Environment and RL Agents]]
- [[_COMMUNITY_Training Autopilot Service|Training Autopilot Service]]
- [[_COMMUNITY_Config and Catalog Management|Config and Catalog Management]]
- [[_COMMUNITY_Frontend App Core|Frontend App Core]]
- [[_COMMUNITY_Dashboard and Visualization|Dashboard and Visualization]]
- [[_COMMUNITY_Test Infrastructure|Test Infrastructure]]
- [[_COMMUNITY_Frontend UI Interactions|Frontend UI Interactions]]
- [[_COMMUNITY_Backend Service Layer|Backend Service Layer]]
- [[_COMMUNITY_Project Research Documents|Project Research Documents]]
- [[_COMMUNITY_Architecture Documentation|Architecture Documentation]]
- [[_COMMUNITY_Job Queue System|Job Queue System]]
- [[_COMMUNITY_Checkpoint Management|Checkpoint Management]]
- [[_COMMUNITY_Model Comparison Tools|Model Comparison Tools]]
- [[_COMMUNITY_Training Test Helpers|Training Test Helpers]]
- [[_COMMUNITY_Run Naming Utilities|Run Naming Utilities]]
- [[_COMMUNITY_Scrum Game Card Data|Scrum Game Card Data]]
- [[_COMMUNITY_RL Algorithm Comparison|RL Algorithm Comparison]]
- [[_COMMUNITY_DQN Training Patterns|DQN Training Patterns]]
- [[_COMMUNITY_Overnight Autopilot Analysis|Overnight Autopilot Analysis]]
- [[_COMMUNITY_Staying Active Fix Runs|Staying Active Fix Runs]]
- [[_COMMUNITY_Apr 14 Training Runs|Apr 14 Training Runs]]
- [[_COMMUNITY_Robustness Evaluation|Robustness Evaluation]]
- [[_COMMUNITY_SARSA Hyperparameter Tuning|SARSA Hyperparameter Tuning]]
- [[_COMMUNITY_Backend Module Init Files|Backend Module Init Files]]
- [[_COMMUNITY_Early DQN Spike Phenomena|Early DQN Spike Phenomena]]
- [[_COMMUNITY_FastAPI Health Endpoint|FastAPI Health Endpoint]]
- [[_COMMUNITY_Autopilot Decisions Tests|Autopilot Decisions Tests]]
- [[_COMMUNITY_v1 Assignment README|v1 Assignment README]]
- [[_COMMUNITY_v2 Deep RL README|v2 Deep RL README]]
- [[_COMMUNITY_Frontend README|Frontend README]]
- [[_COMMUNITY_Shared README|Shared README]]

## God Nodes (most connected - your core abstractions)
1. `ScrumGameEnv` - 172 edges
2. `$()` - 92 edges
3. `GameConfig` - 64 edges
4. `HumanController` - 44 edges
5. `RandomController` - 44 edges
6. `HeuristicController` - 44 edges
7. `ModelController` - 44 edges
8. `from_dict()` - 42 edges
9. `analyze_run()` - 42 edges
10. `IncidentDeck` - 36 edges

## Surprising Connections (you probably didn't know these)
- `Incident Deck (cards.py)` --references--> `2025 ScrumGame Manual (PDF)`  [INFERRED]
  game/v2_deep_rl/README.md → gamedata/2025 ScrumGame Manual.pdf
- `Return the currently valid action ids for one environment state.` --uses--> `ScrumGameEnv`  [INFERRED]
  game\v2_deep_rl\match_runner.py → game\v2_deep_rl\scrum_game_env.py
- `Create one config-consistent parallel match state.` --uses--> `ScrumGameEnv`  [INFERRED]
  game\v2_deep_rl\match_runner.py → game\v2_deep_rl\scrum_game_env.py
- `Advance every active seat by one turn. The human seat consumes the supplied acti` --uses--> `ScrumGameEnv`  [INFERRED]
  game\v2_deep_rl\match_runner.py → game\v2_deep_rl\scrum_game_env.py
- `Return whether every seat has finished its run.` --uses--> `ScrumGameEnv`  [INFERRED]
  game\v2_deep_rl\match_runner.py → game\v2_deep_rl\scrum_game_env.py

## Hyperedges (group relationships)
- **v1 Tabular RL Models (Q-Learning, SARSA, Monte Carlo, Baseline)** — concept_q_learning, concept_sarsa, concept_monte_carlo, concept_heuristic_baseline [EXTRACTED 1.00]
- **v2 Core RL Engine Modules** — concept_config_manager, concept_scrum_game_env, concept_dqn_agent, concept_checkpoint_utils, concept_train_dqn, concept_match_runner, concept_incident_deck, concept_refinements [EXTRACTED 1.00]
- **Control Center Subsystems (Backend + Frontend + Shared)** — concept_fastapi_backend, concept_vanilla_js_spa, control_center_shared_readme, concept_job_queue, concept_sqlite_storage [EXTRACTED 1.00]
- **Autopilot Decision Signals** — concept_training_autopilot, concept_train_dqn, concept_nvidia_ai_advisor, concept_job_queue [EXTRACTED 1.00]

## Communities

### Community 0 - "Game Environment and RL Agents"
Cohesion: 0.02
Nodes (209): BaselineAgent, evaluate_baseline_agent(), Select an action using a fixed heuristic policy.          Policy:         - A, Run a fixed-policy evaluation loop and return episode rewards.      This is a, Simple heuristic baseline agent with no learning logic., The baseline agent uses a fixed non-learning policy., build_incident_cards(), IncidentCard (+201 more)

### Community 1 - "Training Autopilot Service"
Cohesion: 0.02
Nodes (100): BaseModel, analyze_run_endpoint(), autopilot_history_endpoint(), autopilot_status_endpoint(), AutopilotRunRequest, AutopilotSettingsPayload, clear_stop_endpoint(), get_autopilot_settings_endpoint() (+92 more)

### Community 2 - "Config and Catalog Management"
Cohesion: 0.03
Nodes (57): validate_game_config_asset(), validate_training_config_asset(), build_default_incident_cards(), build_default_refinement_rules(), compute_rule_signature(), compute_training_signature(), DiceRuleConfig, from_dict() (+49 more)

### Community 3 - "Frontend App Core"
Cohesion: 0.06
Nodes (109): $(), actionTag(), advancePlayRound(), apiRequest(), attachEvents(), buildOptions(), buildPolyline(), checkpointByPath() (+101 more)

### Community 4 - "Dashboard and Visualization"
Cohesion: 0.06
Nodes (94): action_label(), action_short_label(), build_empty_evaluation_log(), build_empty_training_log(), choose_python_command(), clear_match_state(), controller_from_label(), ensure_runtime_dirs() (+86 more)

### Community 5 - "Test Infrastructure"
Cohesion: 0.04
Nodes (31): _FakeDQNAgent, _FakeModule, Configure sys.path so tests can import the backend services and engine modules., QNetwork, Fixed-size replay memory for DDQN transitions., Add one transition to replay memory., Sample a batch from replay memory and update the policy network.          This, Return a torch-serializable snapshot of replay memory. (+23 more)

### Community 6 - "Frontend UI Interactions"
Cohesion: 0.06
Nodes (68): applyDefaults(), applySelectedCellEdit(), attachListeners(), bindInputNumber(), bindInputText(), bindSelect(), buildLayoutMatrix(), buildPlayersFromCount() (+60 more)

### Community 7 - "Backend Service Layer"
Cohesion: 0.04
Nodes (69): ensure_engine_import_path(), Make the deep-RL engine modules importable from the backend package., delete_game_config_asset(), delete_training_config_asset(), get_game_config(), get_run(), get_run_progress(), get_training_config() (+61 more)

### Community 8 - "Project Research Documents"
Cohesion: 0.05
Nodes (60): AlphaGo (DeepMind Reference), AI Difficulty Profiles and Play Styles, Docker Container with API, Headless Simulator, Imperfect Information Constraint for AI, Karin Plekkenpol (Research Consultant, Inholland), Monte Carlo Tree Search (MCTS), Micha van der Meer (Product Owner) (+52 more)

### Community 9 - "Architecture Documentation"
Cohesion: 0.06
Nodes (58): v2 Deep RL Agent Context, Backend Python Requirements, 7-Product x 4-Sprint Board, 8-Action Refactor (Continue + Switch x7), Binary-Action DQN (pre-refactor v1 reference), Checkpoint Utilities, Standalone Config Editor, Config Manager (GameConfig / TrainingConfig) (+50 more)

### Community 10 - "Job Queue System"
Cohesion: 0.08
Nodes (50): build_command(), _choose_python_command(), Return a Python executable that can import torch.      job_runner may itself b, run_job(), create_job(), delete_job(), get_connection(), get_job() (+42 more)

### Community 11 - "Checkpoint Management"
Cohesion: 0.07
Nodes (34): _checkpoint_catalog_paths(), _checkpoint_id(), _checkpoint_type(), _engine_imports(), get_checkpoint_by_id(), get_checkpoint_compatibility(), _infer_shape_from_state_dict(), list_checkpoints() (+26 more)

### Community 12 - "Model Comparison Tools"
Cohesion: 0.08
Nodes (32): build_report_summary(), main(), print_results_table(), Print a compact terminal table sorted by average reward., Train and compare the assignment-safe baseline and tabular RL models., Create one compact metrics row for the assignment comparison table., Save a bar chart comparing the assignment-track models., Create a report-ready Markdown summary for the assignment deliverable. (+24 more)

### Community 13 - "Training Test Helpers"
Cohesion: 0.1
Nodes (14): Tests for pure helper functions in train_dqn.py:   - epsilon_by_episode   - _s, Epsilon must be strictly decreasing from episode 0 to decay-1., Episode just before the cliff should be just above epsilon_min., Default call must return 1.0 at episode 0 and 0.05 at 450000., resolve_training_config loads a base config then applies overrides., Return a TrainingConfig with sensible defaults, optionally overridden., run_notes='' (the default) must not overwrite a base config's notes., All override params defaulting to None must leave the base values intact. (+6 more)

### Community 14 - "Run Naming Utilities"
Cohesion: 0.2
Nodes (3): _slugify_run_name(), Multiple consecutive non-alnum chars should collapse to a single underscore., TestSlugifyRunName

### Community 15 - "Scrum Game Card Data"
Cohesion: 0.19
Nodes (17): Declining Value Trend Across Sprints, Number of Features per Sprint, Scrum Simulation Game, Sprint 1, Sprint 2, Sprint 3, Sprint 4, Sprint Card (+9 more)

### Community 16 - "RL Algorithm Comparison"
Cohesion: 0.25
Nodes (14): Reward Improvement via Training, DQN v1 Reference Training Curve (500k episodes), Double DQN Training Curve (v2 artifacts - empty/degenerate run), Monte Carlo Algorithm, Monte Carlo Training Curve (v1 Assignment), Baseline Agent (avg reward -65810), Assignment Track Model Comparison Chart, Monte Carlo Agent (avg reward ~1500) (+6 more)

### Community 17 - "DQN Training Patterns"
Cohesion: 0.49
Nodes (14): Double DQN Algorithm, Recurring Upward Reward Spike at ~Episode 24000, Persistent Negative Reward Plateau (~-43000 to -45000), DQN Training Curve - run_2026-04-14_1603 (60k episodes, no improvement), DQN Training Curve - run_2026-04-14_1631 autopilot_continue (50k episodes), DQN Training Curve - run_2026-04-14_1633 autopilot_continue (50k episodes), DQN Training Curve - run_2026-04-14_1715 test (500 episodes, short diagnostic), DQN Training Curve - run_2026-04-14_2001 autopilot_continue (50k episodes) (+6 more)

### Community 18 - "Overnight Autopilot Analysis"
Cohesion: 0.45
Nodes (14): Overnight Autopilot Training Session - April 15 2026, Just Restarting Bug - Episode Reset Issue, Stagnant DQN Training - No Upward Trend, Warm Start Initial Reward Spike, DQN Training Curve - Autopilot Continue Run 01:32, DQN Training Curve - Autopilot Continue Run 02:23, DQN Training Curve - Autopilot Continue Run 03:16, DQN Training Curve - Autopilot Continue Run 04:11 (+6 more)

### Community 19 - "Staying Active Fix Runs"
Cohesion: 0.24
Nodes (13): DQN Reward Stagnation Pattern (Apr 15-16 Runs), Initial Reward Spike Sign Change Between Run Groups, Staying Active Fix - No Visible Reward Improvement, DQN Training Curve - autopilot_continue (2026-04-15 21:05), DQN Training Curve - autopilot_continue (2026-04-15 21:25), DQN Training Curve - autopilot_continue_v2 (2026-04-15 21:46), DQN Training Curve - autopilot_continue_v3 (2026-04-15 22:08), DQN Training Curve - fixing_staying_active (2026-04-16 10:49) (+5 more)

### Community 20 - "Apr 14 Training Runs"
Cohesion: 0.22
Nodes (10): DQN Training Curve - Run 1155 (2026-04-14), DQN Training Curve - Run 1210_01 (2026-04-14), DQN Training Curve - Run 1210 (2026-04-14), DQN Training Curve - Run 1507 Long Run (2026-04-14), DQN Training Curve - First Try New Update2 (2026-04-14), DQN Training Curve - Run 1541 (2026-04-14), DQN Training Curve - Run 1542 (2026-04-14), DQN Training Curve - Run 1549 Extended (2026-04-14) (+2 more)

### Community 21 - "Robustness Evaluation"
Cohesion: 0.38
Nodes (6): evaluate_model_across_seeds(), main(), print_results_table(), Train and evaluate one model across multiple random seeds.      For learning a, Print a clean table of robustness results., Run the robustness evaluation across five fixed random seeds.

### Community 22 - "SARSA Hyperparameter Tuning"
Cohesion: 0.38
Nodes (6): main(), print_results(), Evaluate several SARSA discount factors under a fixed training setup.      A f, Print the tuning results in a clean table., Run a small grid search over SARSA gamma values., tune_sarsa_discount_factor()

### Community 23 - "Backend Module Init Files"
Cohesion: 0.29
Nodes (1): SQLite-backed persistence for the Unified Scrum Game Control Center backend.

### Community 24 - "Early DQN Spike Phenomena"
Cohesion: 0.67
Nodes (6): Early Episode Reward Spike Followed by Drop, High Variance in RL Training (persistent oscillation), Double DQN Training Curve run 2026-04-13 09:17 (20k episodes), Double DQN Training Curve run 2026-04-08 15:00 (5k episodes), Double DQN Training Curve run 2026-04-08 15:27 (1k episodes), Double DQN Training Curve run 2026-04-11 16:34 (10k episodes)

### Community 25 - "FastAPI Health Endpoint"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Autopilot Decisions Tests"
Cohesion: 1.0
Nodes (1): Any flat-reward window must resolve to stop when invalid_action_rate is low.

### Community 27 - "v1 Assignment README"
Cohesion: 1.0
Nodes (1): v1 Assignment Track

### Community 28 - "v2 Deep RL README"
Cohesion: 1.0
Nodes (1): v2 Deep RL Track

### Community 29 - "Frontend README"
Cohesion: 1.0
Nodes (1): Control Center Frontend

### Community 30 - "Shared README"
Cohesion: 1.0
Nodes (1): Control Center Shared

## Knowledge Gaps
- **247 isolated node(s):** `Create one compact metrics row for the assignment comparison table.`, `Save a bar chart comparing the assignment-track models.`, `Create a report-ready Markdown summary for the assignment deliverable.`, `Print a compact terminal table sorted by average reward.`, `Train and compare the assignment-safe baseline and tabular RL models.` (+242 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `FastAPI Health Endpoint`** (2 nodes): `health()`, `app.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Autopilot Decisions Tests`** (1 nodes): `Any flat-reward window must resolve to stop when invalid_action_rate is low.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `v1 Assignment README`** (1 nodes): `v1 Assignment Track`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `v2 Deep RL README`** (1 nodes): `v2 Deep RL Track`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Frontend README`** (1 nodes): `Control Center Frontend`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Shared README`** (1 nodes): `Control Center Shared`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ScrumGameEnv` connect `Game Environment and RL Agents` to `Model Comparison Tools`, `Dashboard and Visualization`?**
  _High betweenness centrality (0.260) - this node is a cross-community bridge._
- **Why does `train_dqn_agent()` connect `Game Environment and RL Agents` to `Config and Catalog Management`, `Test Infrastructure`, `Model Comparison Tools`, `Training Test Helpers`?**
  _High betweenness centrality (0.217) - this node is a cross-community bridge._
- **Why does `_evaluate_one_seed()` connect `Game Environment and RL Agents` to `Backend Service Layer`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 135 inferred relationships involving `ScrumGameEnv` (e.g. with `BaselineAgent` and `Simple heuristic baseline agent with no learning logic.`) actually correct?**
  _`ScrumGameEnv` has 135 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `GameConfig` (e.g. with `IncidentCard` and `IncidentDeck`) actually correct?**
  _`GameConfig` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `HumanController` (e.g. with `Ensure the runtime directories used by the dashboard exist.` and `Load JSON from disk and return None if the file is missing or malformed.`) actually correct?**
  _`HumanController` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `RandomController` (e.g. with `Ensure the runtime directories used by the dashboard exist.` and `Load JSON from disk and return None if the file is missing or malformed.`) actually correct?**
  _`RandomController` has 41 INFERRED edges - model-reasoned connections that need verification._