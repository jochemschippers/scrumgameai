# Training Campaign System — Design Spec
Date: 2026-04-20

## Goal

After a training run plateaus and the autopilot decides to stop, automatically continue training
from the same checkpoint with a varied game config (dice rules, incident cards, reward penalties,
starting money). Repeat for N variations per campaign, producing a model that generalises across
rule environments.

---

## User Flow

1. User starts a training run and optionally checks **"Start as campaign"** (max variations: 5)
2. Autopilot fine-tunes the run normally (lower_lr, extend_epsilon_decay, etc.)
3. Autopilot reaches a **"stop"** decision → triggers `CampaignService.on_run_stopped()`
4. AI generates a subtle/moderate game config variation targeting the run's weaknesses
5. New job queued: resumes from `latest_scrum_model.pth`, applies varied config
6. Steps 2–5 repeat until `variations_completed == max_variations`
7. Campaign status becomes **"completed"** — no further queuing
8. User can click **"Escalate Rules"** to start a new campaign from the final checkpoint
   with major-bounds variations

---

## Architecture

### New files

| File | Responsibility |
|------|---------------|
| `services/campaign_service.py` | Campaign lifecycle: create, progress, complete, stop |
| `services/campaign_variation_generator.py` | AI-driven config mutation within safe bounds |
| `routes/routes_campaigns.py` | REST API for campaign CRUD and status |
| `artifacts/campaigns/<campaign_id>.json` | Persisted campaign state |

### Modified files

| File | Change |
|------|--------|
| `services/training_autopilot.py` | On "stop" decision, check if run belongs to campaign → call `on_run_stopped()` |
| `frontend/index.html` | Campaign toggle on launch, status panel, Escalate button |

---

## Campaign State Schema

Stored at `artifacts/campaigns/<campaign_id>.json`:

```json
{
  "campaign_id": "campaign_2026-04-20_1400",
  "status": "running | completed | paused | stopped",
  "base_run_id": "run_2026-04-20_1400",
  "current_run_id": "run_2026-04-20_1430_v2",
  "variations_completed": 2,
  "max_variations": 5,
  "escalate_mode": false,
  "variation_history": [
    {
      "index": 1,
      "from_run": "run_2026-04-20_1400",
      "to_run": "run_2026-04-20_1415_v1",
      "changes": { "incident_cards[0].severity_multiplier": 1.4, "dice_rules[2].dice_sides": 8 },
      "reason": "High bankruptcy rate on seeds 123 and 2026 — increased incident severity and sprint-3 dice variance"
    }
  ]
}
```

---

## CampaignService

```python
create_campaign(base_run_id: str, max_variations: int = 5) -> str  # returns campaign_id
on_run_stopped(run_id: str) -> None       # called by autopilot on "stop"
get_campaign(campaign_id: str) -> dict
get_campaign_for_run(run_id: str) -> dict | None
list_campaigns() -> list[dict]
stop_campaign(campaign_id: str) -> None   # manual stop
escalate(campaign_id: str) -> None        # queue one escalate-bounds variation run
```

**`on_run_stopped` flow:**
1. Look up campaign by `current_run_id`
2. If not found or `status != "running"` → no-op
3. If `variations_completed >= max_variations` → set status `"completed"`, return
4. Call `CampaignVariationGenerator.generate(current_game_config, run_metrics, variation_index, escalate=False)`
5. Write varied `game_config.json` to new run dir
6. Queue job: `resume_from=latest_scrum_model.pth`, `game_config=<varied>`, `resume_mode="strict"`
7. Update campaign JSON: increment `variations_completed`, append history entry, update `current_run_id`

---

## CampaignVariationGenerator

Calls the existing AI advisor (NVIDIA/LLaMA endpoint, `NVIDIA_API_KEY`) with:
- Current `game_config` (full JSON)
- Run metrics: `bankruptcy_rate`, `invalid_action_rate`, `reward_variance`, `avg_ending_money`, `avg_turns`
- `variation_index` (1–N) — AI uses this to modulate how much it changes
- `escalate` flag — relaxes bounds when True

Returns: `(new_game_config: GameConfig, changes: dict, reason: str)`

AI response must be a JSON diff of only changed fields + a `reason` string. Generator validates
all values against bounds before applying, then passes through `validate_game_config()`.

### Safe bounds — automated (subtle/moderate)

| Parameter | Allowed range |
|-----------|--------------|
| `starting_money` | ±25% of current |
| `max_turns` | current ±2 |
| Dice sides per rule | current ±2, min 4, max 20 |
| Dice count per rule | current ±1, min 1, max 3 |
| Incident `severity_multiplier` | 0.5× – 2.0× |
| Incident `draw_probability` | 0.5 – 1.0 |
| Reward penalties | ±30% |
| Loan amounts / interest | ±20% |

### Escalate bounds — manual trigger

| Parameter | Allowed range |
|-----------|--------------|
| `starting_money` | ±60% |
| `max_turns` | 3 – 15 |
| Dice sides | 4 – 20 freely |
| Dice count | 1 – 4 |
| Incident severity | 0.2× – 3.0× |
| Incident draw probability | 0.2 – 1.0 |
| Reward penalties | ±60% |
| Loan amounts / interest | ±40% |
| Incident card weights | fully reshuffled |

---

## REST API

Base path: `/api/campaigns`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/campaigns` | Create campaign for a run |
| `GET` | `/api/campaigns` | List all campaigns |
| `GET` | `/api/campaigns/{id}` | Status + full variation history |
| `POST` | `/api/campaigns/{id}/stop` | Manual stop |
| `POST` | `/api/campaigns/{id}/escalate` | Trigger escalate-mode variation |

---

## Autopilot Integration

In `training_autopilot.py`, `run_autopilot()`, after writing the `"stop"` decision record:

```python
campaign = campaign_service.get_campaign_for_run(run_id)
if campaign:
    campaign_service.on_run_stopped(run_id)
    return  # campaign owns next job queuing
# existing stop logic (no campaign) continues as before
```

No other changes to autopilot logic.

---

## Frontend Changes

### 1. Campaign toggle on training launch
Checkbox: **"Start as campaign"** + number input for max variations (default 5).
Included in job payload as `campaign_max_variations: int | null`.
Backend creates campaign on job start if value is set.

### 2. Campaign status panel
Shown when a campaign is active for the current run:
```
Campaign: campaign_2026-04-20_1400         [Stop Campaign]
Variation 2 / 5  ████████░░░░░░░
  v1: "Increased incident severity ×1.4, sprint-3 dice +2 (high bankruptcy)"
  v2: currently running...
```

### 3. Escalate Rules button
Shown when `campaign.status == "completed"`:
```
[Escalate Rules →]
```
Calls `POST /api/campaigns/{id}/escalate`. Starts a new campaign from the final checkpoint
with escalate-bounds variation.

---

## Out of Scope

- Structural changes (number of products, sprint count) — reserved for a future major version
- Campaign branching (running multiple variation paths in parallel)
- Automated escalation — escalate is always a manual trigger
