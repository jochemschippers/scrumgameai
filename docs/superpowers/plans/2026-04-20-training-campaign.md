# Training Campaign System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the autopilot stops a training run due to plateau, automatically continue training from the same checkpoint with AI-varied game configs (dice rules, incidents, reward penalties) for up to N variations, producing a model that generalises across rule environments.

**Architecture:** A new `CampaignService` tracks multi-run training campaigns persisted as JSON files in `artifacts/campaigns/`. When `training_autopilot.run_autopilot()` writes a "stop" decision, it checks if the run belongs to a campaign and delegates to `campaign_service.on_run_stopped()`, which calls `CampaignVariationGenerator` (AI-driven bounded config mutation) and queues the next training job resuming from the latest checkpoint. Frontend gains a campaign toggle on the launch form, a status panel, and an escalate button.

**Tech Stack:** Python 3.10+, FastAPI, OpenAI SDK (NVIDIA endpoint, same as autopilot), frozen dataclasses (GameConfig), pytest, vanilla JS (existing frontend pattern)

---

## File Map

| Action | Path (relative to `game/v2_deep_rl/`) | Responsibility |
|--------|---------------------------------------|---------------|
| Create | `control_center/backend/services/campaign_service.py` | Campaign CRUD + lifecycle (on_run_stopped, escalate) |
| Create | `control_center/backend/services/campaign_variation_generator.py` | AI-driven bounded game config mutation |
| Create | `control_center/backend/api/routes_campaigns.py` | REST endpoints for campaign management |
| Create | `tests/test_campaign_service.py` | Unit tests for campaign lifecycle |
| Create | `tests/test_campaign_variation_generator.py` | Unit tests for variation generation |
| Modify | `control_center/backend/services/training_autopilot.py` | Call campaign_service on stop decision (lines 578–581) |
| Modify | `control_center/backend/app.py` | Register campaigns router (after line 13, line 41) |
| Modify | `control_center/frontend/index.html` | Campaign toggle (line ~320), status panel, escalate button |

---

## Task 1: Campaign persistence + CRUD

**Files:**
- Create: `game/v2_deep_rl/control_center/backend/services/campaign_service.py`
- Create: `game/v2_deep_rl/tests/test_campaign_service.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_campaign_service.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _patch_dir(campaigns_dir):
    return patch("services.campaign_service.CAMPAIGNS_DIR", campaigns_dir)


def test_create_campaign_writes_json(tmp_path):
    from services.campaign_service import create_campaign
    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_2026-04-20_1400", max_variations=5)
    data = json.loads((d / f"{cid}.json").read_text())
    assert data["campaign_id"] == cid
    assert data["base_run_id"] == "run_2026-04-20_1400"
    assert data["current_run_id"] == "run_2026-04-20_1400"
    assert data["max_variations"] == 5
    assert data["variations_completed"] == 0
    assert data["status"] == "running"
    assert data["variation_history"] == []
    assert data["escalate_mode"] is False


def test_get_campaign_returns_data(tmp_path):
    from services.campaign_service import create_campaign, get_campaign
    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_abc", max_variations=3)
        result = get_campaign(cid)
    assert result["campaign_id"] == cid
    assert result["max_variations"] == 3


def test_get_campaign_unknown_raises(tmp_path):
    from services.campaign_service import get_campaign
    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        with pytest.raises(FileNotFoundError):
            get_campaign("nonexistent")


def test_list_campaigns_returns_all(tmp_path):
    from services.campaign_service import create_campaign, list_campaigns
    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid1 = create_campaign("run_a", max_variations=3)
        cid2 = create_campaign("run_b", max_variations=5)
        results = list_campaigns()
    ids = [c["campaign_id"] for c in results]
    assert cid1 in ids and cid2 in ids


def test_stop_campaign_sets_status(tmp_path):
    from services.campaign_service import create_campaign, stop_campaign, get_campaign
    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_x", max_variations=5)
        stop_campaign(cid)
        result = get_campaign(cid)
    assert result["status"] == "stopped"


def test_get_campaign_for_run_finds_by_base_run_id(tmp_path):
    from services.campaign_service import create_campaign, get_campaign_for_run
    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_base", max_variations=5)
        result = get_campaign_for_run("run_base")
    assert result is not None
    assert result["campaign_id"] == cid


def test_get_campaign_for_run_returns_none_when_stopped(tmp_path):
    from services.campaign_service import create_campaign, stop_campaign, get_campaign_for_run
    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_stopped", max_variations=5)
        stop_campaign(cid)
        result = get_campaign_for_run("run_stopped")
    assert result is None


def test_get_campaign_for_run_finds_by_current_run_id(tmp_path):
    from services.campaign_service import create_campaign, get_campaign_for_run, _write_campaign, get_campaign
    d = tmp_path / "campaigns"
    d.mkdir()
    with _patch_dir(d):
        cid = create_campaign("run_base", max_variations=5)
        data = get_campaign(cid)
        data = {**data, "current_run_id": "run_base_v1"}
        _write_campaign(data)
        result = get_campaign_for_run("run_base_v1")
    assert result is not None
    assert result["campaign_id"] == cid
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_service.py -v
```
Expected: `ImportError` — `campaign_service` doesn't exist yet

- [ ] **Step 3: Implement campaign_service.py persistence + CRUD**

```python
# control_center/backend/services/campaign_service.py
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from services.app_paths import ARTIFACTS_DIR, RUNS_DIR

CAMPAIGNS_DIR = ARTIFACTS_DIR / "campaigns"


def _ensure_dir() -> None:
    CAMPAIGNS_DIR.mkdir(parents=True, exist_ok=True)


def _campaign_path(campaign_id: str) -> Path:
    return CAMPAIGNS_DIR / f"{campaign_id}.json"


def _read_campaign(campaign_id: str) -> dict:
    path = _campaign_path(campaign_id)
    if not path.exists():
        raise FileNotFoundError(f"Campaign not found: {campaign_id}")
    return json.loads(path.read_text())


def _write_campaign(data: dict) -> None:
    _ensure_dir()
    _campaign_path(data["campaign_id"]).write_text(json.dumps(data, indent=2))


def _make_campaign_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    uid = uuid.uuid4().hex[:6]
    return f"campaign_{ts}_{uid}"


def create_campaign(base_run_id: str, max_variations: int = 5) -> str:
    _ensure_dir()
    campaign_id = _make_campaign_id()
    _write_campaign({
        "campaign_id": campaign_id,
        "status": "running",
        "base_run_id": base_run_id,
        "current_run_id": base_run_id,
        "variations_completed": 0,
        "max_variations": max_variations,
        "escalate_mode": False,
        "variation_history": [],
    })
    return campaign_id


def get_campaign(campaign_id: str) -> dict:
    return _read_campaign(campaign_id)


def list_campaigns() -> list[dict]:
    _ensure_dir()
    campaigns = []
    for path in sorted(CAMPAIGNS_DIR.glob("campaign_*.json")):
        try:
            campaigns.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return campaigns


def stop_campaign(campaign_id: str) -> None:
    data = _read_campaign(campaign_id)
    data["status"] = "stopped"
    _write_campaign(data)


def get_campaign_for_run(run_id: str) -> dict | None:
    for campaign in list_campaigns():
        if campaign["status"] != "running":
            continue
        if campaign["base_run_id"] == run_id or campaign["current_run_id"] == run_id:
            return campaign
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_service.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add game/v2_deep_rl/control_center/backend/services/campaign_service.py \
        game/v2_deep_rl/tests/test_campaign_service.py
git commit -m "feat: add campaign service persistence and CRUD"
```

---

## Task 2: CampaignVariationGenerator — bounds clamping + config mutation

**Files:**
- Create: `game/v2_deep_rl/control_center/backend/services/campaign_variation_generator.py`
- Create: `game/v2_deep_rl/tests/test_campaign_variation_generator.py`

- [ ] **Step 1: Write failing tests for bounds clamping**

```python
# tests/test_campaign_variation_generator.py
from __future__ import annotations

import pytest


def _base_config_dict():
    return {
        "schema_version": "2.0",
        "config_name": "Test",
        "config_description": "",
        "players_count": 1,
        "product_names": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
        "max_turns": 6,
        "starting_money": 25000,
        "ring_value": 10000,
        "cost_continue": 0,
        "cost_switch_mid": 5000,
        "cost_switch_after": 2500,
        "mandatory_loan_amount": 50000,
        "loan_interest": 5000,
        "penalty_negative": -100,
        "penalty_positive": 0,
        "daily_scrums_per_sprint": 3,
        "daily_scrum_target": 3,
        "board_ring_values": [[0] * 7 for _ in range(4)],
        "board_features": [[3] * 7 for _ in range(4)],
        "dice_rules": [
            {"feature_min": 1, "feature_max": 4, "dice_count": 1, "dice_sides": 6},
            {"feature_min": 5, "feature_max": 8, "dice_count": 1, "dice_sides": 8},
        ],
        "refinement": {"active": False, "product_rules": []},
        "incident": {
            "active": True,
            "allow_player_specific_incidents": False,
            "draw_probability": 1.0,
            "severity_multiplier": 1.0,
            "cards": [],
        },
        "reserved_fields": {},
    }


def test_clamp_starting_money_within_safe_bounds():
    from services.campaign_variation_generator import clamp_diff
    base = _base_config_dict()
    # 25000 * 1.25 = 31250, so 40000 should be clamped to 31250
    diff = {"starting_money": 40000}
    result = clamp_diff(diff, base, escalate=False)
    assert result["starting_money"] == 31250


def test_clamp_max_turns_within_safe_bounds():
    from services.campaign_variation_generator import clamp_diff
    base = _base_config_dict()
    # max_turns=6, delta=2 → max allowed 8, min allowed 4
    diff = {"max_turns": 20}
    result = clamp_diff(diff, base, escalate=False)
    assert result["max_turns"] == 8


def test_clamp_incident_draw_probability():
    from services.campaign_variation_generator import clamp_diff
    base = _base_config_dict()
    diff = {"incident_draw_probability": 0.1}  # below safe min of 0.5
    result = clamp_diff(diff, base, escalate=False)
    assert result["incident_draw_probability"] == 0.5


def test_escalate_allows_wider_starting_money():
    from services.campaign_variation_generator import clamp_diff
    base = _base_config_dict()
    # 25000 * 1.60 = 40000, so 39000 should pass in escalate mode
    diff = {"starting_money": 39000}
    result = clamp_diff(diff, base, escalate=True)
    assert result["starting_money"] == 39000


def test_clamp_dice_sides():
    from services.campaign_variation_generator import clamp_diff
    base = _base_config_dict()
    # dice_rule_0 dice_sides=6, delta=2 → max 8
    diff = {"dice_rule_0_dice_sides": 15}
    result = clamp_diff(diff, base, escalate=False)
    assert result["dice_rule_0_dice_sides"] == 8


def test_apply_diff_to_config_dict():
    from services.campaign_variation_generator import apply_diff_to_config
    base = _base_config_dict()
    diff = {
        "starting_money": 28000,
        "incident_draw_probability": 0.8,
        "dice_rule_0_dice_sides": 8,
    }
    result = apply_diff_to_config(base, diff)
    assert result["starting_money"] == 28000
    assert result["incident"]["draw_probability"] == 0.8
    assert result["dice_rules"][0]["dice_sides"] == 8
    # unchanged fields stay the same
    assert result["max_turns"] == 6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_variation_generator.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement bounds clamping + config mutation**

```python
# control_center/backend/services/campaign_variation_generator.py
from __future__ import annotations

import copy
import json
import os
from typing import Any

NVIDIA_API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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",
)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "abacusai/dracarys-llama-3.1-70b-instruct"

# Bounds spec: {key: (type, *args)}
#   ("pct", pct)        → clamp to [current*(1-pct), current*(1+pct)]
#   ("delta", d, mn, mx)→ clamp to [current-d, current+d] then clip to [mn, mx]
#   ("abs", mn, mx)     → clamp directly to [mn, mx]

_SAFE_BOUNDS: dict[str, tuple] = {
    "starting_money":               ("pct", 0.25),
    "max_turns":                    ("delta", 2, 3, 15),
    "penalty_negative":             ("pct", 0.30),
    "penalty_positive":             ("pct", 0.30),
    "mandatory_loan_amount":        ("pct", 0.20),
    "loan_interest":                ("pct", 0.20),
    "incident_draw_probability":    ("abs", 0.5, 1.0),
    "incident_severity_multiplier": ("abs", 0.5, 2.0),
    **{f"dice_rule_{i}_dice_sides": ("delta", 2, 4, 20) for i in range(6)},
    **{f"dice_rule_{i}_dice_count": ("delta", 1, 1, 3) for i in range(6)},
}

_ESCALATE_BOUNDS: dict[str, tuple] = {
    "starting_money":               ("pct", 0.60),
    "max_turns":                    ("abs", 3, 15),
    "penalty_negative":             ("pct", 0.60),
    "penalty_positive":             ("pct", 0.60),
    "mandatory_loan_amount":        ("pct", 0.40),
    "loan_interest":                ("pct", 0.40),
    "incident_draw_probability":    ("abs", 0.2, 1.0),
    "incident_severity_multiplier": ("abs", 0.2, 3.0),
    **{f"dice_rule_{i}_dice_sides": ("abs", 4, 20) for i in range(6)},
    **{f"dice_rule_{i}_dice_count": ("abs", 1, 4) for i in range(6)},
}


def _get_current(key: str, config_dict: dict) -> float | None:
    """Read the current value for a flat diff key from a nested config dict."""
    if key == "incident_draw_probability":
        return config_dict.get("incident", {}).get("draw_probability")
    if key == "incident_severity_multiplier":
        return config_dict.get("incident", {}).get("severity_multiplier")
    if key.startswith("dice_rule_"):
        parts = key.split("_")  # ["dice", "rule", "0", "dice", "sides/count"]
        idx = int(parts[2])
        field = "_".join(parts[3:])  # "dice_sides" or "dice_count"
        rules = config_dict.get("dice_rules", [])
        if idx < len(rules):
            return rules[idx].get(field)
        return None
    return config_dict.get(key)


def _clamp_one(key: str, value: float, current: float, bounds: tuple) -> float:
    btype = bounds[0]
    if btype == "pct":
        pct = bounds[1]
        lo, hi = current * (1 - pct), current * (1 + pct)
        # handle negative values (penalty_negative is negative)
        if current < 0:
            lo, hi = hi, lo
        return max(lo, min(hi, value))
    if btype == "delta":
        _, delta, mn, mx = bounds
        lo = max(mn, current - delta)
        hi = min(mx, current + delta)
        return max(lo, min(hi, value))
    if btype == "abs":
        _, mn, mx = bounds
        return max(mn, min(mx, value))
    return value


def clamp_diff(diff: dict, config_dict: dict, escalate: bool = False) -> dict:
    """Return a copy of diff with all values clamped within safe/escalate bounds."""
    bounds_map = _ESCALATE_BOUNDS if escalate else _SAFE_BOUNDS
    result = {}
    for key, value in diff.items():
        if key not in bounds_map:
            continue  # ignore unknown keys
        current = _get_current(key, config_dict)
        if current is None:
            continue
        b = bounds_map[key]
        # preserve int type for integer fields
        clamped = _clamp_one(key, float(value), float(current), b)
        if isinstance(current, int):
            clamped = int(round(clamped))
        result[key] = clamped
    return result


def apply_diff_to_config(config_dict: dict, diff: dict) -> dict:
    """Apply a flat diff dict to a nested GameConfig dict, returning a modified copy."""
    result = copy.deepcopy(config_dict)
    for key, value in diff.items():
        if key == "incident_draw_probability":
            result.setdefault("incident", {})["draw_probability"] = value
        elif key == "incident_severity_multiplier":
            result.setdefault("incident", {})["severity_multiplier"] = value
        elif key.startswith("dice_rule_"):
            parts = key.split("_")
            idx = int(parts[2])
            field = "_".join(parts[3:])
            while len(result.get("dice_rules", [])) <= idx:
                result.setdefault("dice_rules", []).append({})
            result["dice_rules"][idx][field] = value
        else:
            result[key] = value
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_variation_generator.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add game/v2_deep_rl/control_center/backend/services/campaign_variation_generator.py \
        game/v2_deep_rl/tests/test_campaign_variation_generator.py
git commit -m "feat: add campaign variation generator bounds clamping"
```

---

## Task 3: CampaignVariationGenerator — AI call

**Files:**
- Modify: `game/v2_deep_rl/control_center/backend/services/campaign_variation_generator.py`
- Modify: `game/v2_deep_rl/tests/test_campaign_variation_generator.py`

- [ ] **Step 1: Add tests for the AI generate function**

Add these tests to `tests/test_campaign_variation_generator.py`:

```python
from unittest.mock import MagicMock, patch


def test_generate_with_mocked_ai_returns_new_config():
    from services.campaign_variation_generator import generate_variation
    base = _base_config_dict()
    metrics = {
        "latest_reward": 50000.0,
        "bankruptcy_rate": 0.40,
        "average_ending_money": 30000.0,
        "invalid_action_rate": 0.05,
        "reward_cv": 0.15,
    }
    ai_response = json.dumps({
        "changes": {"starting_money": 30000, "incident_draw_probability": 0.8},
        "reason": "High bankruptcy rate — more starting capital may help",
    })
    mock_choice = MagicMock()
    mock_choice.message.content = ai_response
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("services.campaign_variation_generator.OpenAI", return_value=mock_client):
        new_config, changes, reason = generate_variation(base, metrics, variation_index=1, escalate=False)

    assert new_config["starting_money"] == 30000
    assert new_config["incident"]["draw_probability"] == 0.8
    assert "bankruptcy" in reason.lower()
    assert "starting_money" in changes


def test_generate_falls_back_on_ai_failure():
    from services.campaign_variation_generator import generate_variation
    base = _base_config_dict()
    metrics = {"latest_reward": 0, "bankruptcy_rate": 0.5,
               "average_ending_money": 0, "invalid_action_rate": 0.1, "reward_cv": 0.3}

    with patch("services.campaign_variation_generator.OpenAI", side_effect=Exception("network error")):
        new_config, changes, reason = generate_variation(base, metrics, variation_index=1, escalate=False)

    # fallback: config is unchanged, reason explains the failure
    assert new_config == base
    assert "fallback" in reason.lower() or "failed" in reason.lower()
```

Import `json` at the top of the test file.

- [ ] **Step 2: Run to verify they fail**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_variation_generator.py::test_generate_with_mocked_ai_returns_new_config \
                 tests/test_campaign_variation_generator.py::test_generate_falls_back_on_ai_failure -v
```
Expected: `ImportError` (generate_variation not defined yet)

- [ ] **Step 3: Implement generate_variation in campaign_variation_generator.py**

Add to the bottom of `campaign_variation_generator.py`:

```python
def generate_variation(
    config_dict: dict,
    metrics: dict,
    variation_index: int,
    escalate: bool = False,
) -> tuple[dict, dict, str]:
    """
    Ask the AI to suggest game config changes targeting the run's weaknesses.

    Returns (new_config_dict, changes_applied, reason_string).
    On any failure returns (config_dict, {}, fallback_reason).
    """
    try:
        from openai import OpenAI
    except ImportError:
        return config_dict, {}, "fallback: openai package not installed"

    bounds_summary = (
        "starting_money: ±25% | max_turns: ±2 | penalty_negative: ±30% | "
        "loan_interest: ±20% | incident_draw_probability: 0.5–1.0 | "
        "incident_severity_multiplier: 0.5–2.0 | dice_rule_N_dice_sides: ±2 (4–20) | "
        "dice_rule_N_dice_count: ±1 (1–3)"
        if not escalate else
        "starting_money: ±60% | max_turns: 3–15 | penalty_negative: ±60% | "
        "loan_interest: ±40% | incident_draw_probability: 0.2–1.0 | "
        "incident_severity_multiplier: 0.2–3.0 | dice_rule_N_dice_sides: 4–20 | "
        "dice_rule_N_dice_count: 1–4"
    )

    prompt = f"""You are a game-rules advisor for a Scrum board game RL training campaign.
A DQN agent is being trained across multiple rule environments to improve adaptability.
This is variation {variation_index}.{"  Use wider parameter changes (escalate mode)." if escalate else ""}

=== Current game config (relevant fields) ===
starting_money            : {config_dict.get("starting_money")}
max_turns                 : {config_dict.get("max_turns")}
penalty_negative          : {config_dict.get("penalty_negative")}
penalty_positive          : {config_dict.get("penalty_positive")}
mandatory_loan_amount     : {config_dict.get("mandatory_loan_amount")}
loan_interest             : {config_dict.get("loan_interest")}
incident_draw_probability : {config_dict.get("incident", {}).get("draw_probability")}
incident_severity_multiplier: {config_dict.get("incident", {}).get("severity_multiplier")}
dice_rules                : {json.dumps(config_dict.get("dice_rules", []))}

=== Run metrics ===
latest_average_reward    : {metrics.get("latest_reward")}
bankruptcy_rate          : {metrics.get("bankruptcy_rate")} (target <0.20)
average_ending_money     : {metrics.get("average_ending_money")}
invalid_action_rate      : {metrics.get("invalid_action_rate")} (target <0.10)
reward_cv                : {metrics.get("reward_cv")} (>0.20 = high variance)

=== Your task ===
Suggest 1–3 game config changes that will create a meaningfully different but learnable
environment. Target the agent's observed weaknesses. For dice rules use index 0, 1, … matching
the dice_rules list above.

Allowed parameter keys and bounds:
{bounds_summary}

Respond with JSON only:
{{
  "changes": {{
    "<key>": <value>,
    ...
  }},
  "reason": "<one sentence explaining what weakness you are targeting>"
}}"""

    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        return config_dict, {}, f"fallback: AI call failed ({exc})"

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        suggestion = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return config_dict, {}, f"fallback: AI returned unparseable response"

    raw_changes = suggestion.get("changes", {})
    reason = str(suggestion.get("reason", "AI variation"))

    clamped = clamp_diff(raw_changes, config_dict, escalate=escalate)
    if not clamped:
        return config_dict, {}, f"fallback: no valid changes after clamping"

    new_config = apply_diff_to_config(config_dict, clamped)
    return new_config, clamped, reason
```

Also add `import json` to the top of `campaign_variation_generator.py`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_variation_generator.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add game/v2_deep_rl/control_center/backend/services/campaign_variation_generator.py \
        game/v2_deep_rl/tests/test_campaign_variation_generator.py
git commit -m "feat: add AI-driven campaign variation generator"
```

---

## Task 4: CampaignService.on_run_stopped + _generate_and_queue_next

**Files:**
- Modify: `game/v2_deep_rl/control_center/backend/services/campaign_service.py`
- Modify: `game/v2_deep_rl/tests/test_campaign_service.py`

- [ ] **Step 1: Add tests for on_run_stopped**

Add to `tests/test_campaign_service.py`:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_run_dir(runs_dir: Path, run_id: str, game_config: dict | None = None) -> Path:
    run_dir = runs_dir / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "latest_scrum_model.pth").touch()
    if game_config:
        (run_dir / "game_config.json").write_text(json.dumps(game_config))
    else:
        (run_dir / "game_config.json").write_text(json.dumps({
            "schema_version": "2.0", "config_name": "Test",
            "config_description": "", "players_count": 1,
            "product_names": ["P1","P2","P3","P4","P5","P6","P7"],
            "max_turns": 6, "starting_money": 25000, "ring_value": 10000,
            "cost_continue": 0, "cost_switch_mid": 5000, "cost_switch_after": 2500,
            "mandatory_loan_amount": 50000, "loan_interest": 5000,
            "penalty_negative": -100, "penalty_positive": 0,
            "daily_scrums_per_sprint": 3, "daily_scrum_target": 3,
            "board_ring_values": [[0]*7 for _ in range(4)],
            "board_features": [[3]*7 for _ in range(4)],
            "dice_rules": [{"feature_min":1,"feature_max":4,"dice_count":1,"dice_sides":6}],
            "refinement": {"active": False, "product_rules": []},
            "incident": {"active": True, "allow_player_specific_incidents": False,
                         "draw_probability": 1.0, "severity_multiplier": 1.0, "cards": []},
            "reserved_fields": {},
        }))
    return run_dir


_METRICS = {"latest_reward": 50000.0, "bankruptcy_rate": 0.3,
            "average_ending_money": 30000.0, "invalid_action_rate": 0.05, "reward_cv": 0.2}


def test_on_run_stopped_queues_next_job(tmp_path):
    from services.campaign_service import create_campaign, on_run_stopped, get_campaign
    d = tmp_path / "campaigns"
    d.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dir(runs, "run_base")
    fake_job = {"id": 1, "run_dir": str(runs / "run_base_v1")}

    with _patch_dir(d), \
         patch("services.campaign_service.RUNS_DIR", runs), \
         patch("services.campaign_service.generate_variation",
               return_value=({}, {"starting_money": 28000}, "test reason")), \
         patch("services.campaign_service.enqueue_train_job", return_value=fake_job) as mock_enqueue:
        cid = create_campaign("run_base", max_variations=5)
        on_run_stopped("run_base", _METRICS)
        result = get_campaign(cid)

    mock_enqueue.assert_called_once()
    assert result["variations_completed"] == 1
    assert result["status"] == "running"
    assert len(result["variation_history"]) == 1
    assert result["variation_history"][0]["reason"] == "test reason"


def test_on_run_stopped_completes_at_max_variations(tmp_path):
    from services.campaign_service import create_campaign, on_run_stopped, get_campaign, _write_campaign
    d = tmp_path / "campaigns"
    d.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dir(runs, "run_base")

    with _patch_dir(d), patch("services.campaign_service.RUNS_DIR", runs):
        cid = create_campaign("run_base", max_variations=1)
        # Pre-set variations_completed to max - 1 (already at limit after this call)
        data = json.loads((d / f"{cid}.json").read_text())
        # max_variations=1 so this call should complete the campaign
        with patch("services.campaign_service.generate_variation",
                   return_value=({}, {}, "reason")), \
             patch("services.campaign_service.enqueue_train_job",
                   return_value={"id": 1, "run_dir": str(runs / "run_v1")}):
            on_run_stopped("run_base", _METRICS)
        result = get_campaign(cid)

    assert result["variations_completed"] == 1
    assert result["status"] == "completed"


def test_on_run_stopped_no_op_when_no_campaign(tmp_path):
    from services.campaign_service import on_run_stopped
    d = tmp_path / "campaigns"
    d.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dir(runs, "run_orphan")

    with _patch_dir(d), patch("services.campaign_service.RUNS_DIR", runs), \
         patch("services.campaign_service.enqueue_train_job") as mock_enqueue:
        on_run_stopped("run_orphan", _METRICS)
    mock_enqueue.assert_not_called()
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_service.py::test_on_run_stopped_queues_next_job \
                 tests/test_campaign_service.py::test_on_run_stopped_completes_at_max_variations \
                 tests/test_campaign_service.py::test_on_run_stopped_no_op_when_no_campaign -v
```
Expected: `AttributeError` — `on_run_stopped` not defined yet

- [ ] **Step 3: Implement on_run_stopped + _generate_and_queue_next in campaign_service.py**

Add these imports to the top of `campaign_service.py`:

```python
import sys
from pathlib import Path
```

Add `RUNS_DIR` to the import line:
```python
from services.app_paths import ARTIFACTS_DIR, RUNS_DIR
```

Add these functions to `campaign_service.py`:

```python
def _next_run_id(campaign: dict) -> str:
    n = campaign["variations_completed"] + 1
    base = campaign["base_run_id"]
    return f"{base}_cv{n}"


def _save_varied_config(campaign_id: str, variation_index: int, config_dict: dict) -> Path:
    path = CAMPAIGNS_DIR / f"{campaign_id}_game_config_v{variation_index}.json"
    path.write_text(json.dumps(config_dict, indent=2))
    return path


def on_run_stopped(run_id: str, metrics: dict) -> None:
    campaign = get_campaign_for_run(run_id)
    if campaign is None:
        return
    _generate_and_queue_next(campaign, run_id, metrics, escalate=False)


def _generate_and_queue_next(campaign: dict, from_run_id: str, metrics: dict, escalate: bool) -> None:
    from services.campaign_variation_generator import generate_variation
    from jobs.queue_manager import enqueue_train_job

    cid = campaign["campaign_id"]
    variation_index = campaign["variations_completed"] + 1

    run_dir = RUNS_DIR / from_run_id
    try:
        import json as _json
        config_dict = _json.loads((run_dir / "game_config.json").read_text())
    except (OSError, ValueError):
        config_dict = {}

    checkpoint_path = run_dir / "checkpoints" / "latest_scrum_model.pth"

    new_config_dict, changes, reason = generate_variation(
        config_dict, metrics, variation_index=variation_index, escalate=escalate
    )

    config_path = _save_varied_config(cid, variation_index, new_config_dict)

    new_run_id = _next_run_id(campaign)
    payload = {
        "run_name": new_run_id,
        "game_config_path": str(config_path),
        "resume_from": str(checkpoint_path),
        "resume_mode": "strict",
        "resume_episodes_mode": "incremental",
        "autopilot_after_completion": True,
        "autopilot_context": {},
    }

    job = enqueue_train_job(payload)
    new_run_dir = job.get("run_dir", "")
    new_run_id_actual = Path(new_run_dir).name if new_run_dir else new_run_id

    campaign = dict(campaign)
    campaign["current_run_id"] = new_run_id_actual
    campaign["variations_completed"] = variation_index
    campaign["variation_history"] = list(campaign["variation_history"]) + [{
        "index": variation_index,
        "from_run": from_run_id,
        "to_run": new_run_id_actual,
        "changes": changes,
        "reason": reason,
        "escalate": escalate,
    }]

    if variation_index >= campaign["max_variations"]:
        campaign["status"] = "completed"

    _write_campaign(campaign)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_service.py -v
```
Expected: all 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add game/v2_deep_rl/control_center/backend/services/campaign_service.py \
        game/v2_deep_rl/tests/test_campaign_service.py
git commit -m "feat: implement on_run_stopped and variation queuing in campaign service"
```

---

## Task 5: CampaignService.escalate

**Files:**
- Modify: `game/v2_deep_rl/control_center/backend/services/campaign_service.py`
- Modify: `game/v2_deep_rl/tests/test_campaign_service.py`

- [ ] **Step 1: Add test for escalate**

Add to `tests/test_campaign_service.py`:

```python
def test_escalate_queues_job_with_escalate_mode(tmp_path):
    from services.campaign_service import create_campaign, escalate_campaign, get_campaign, _write_campaign
    d = tmp_path / "campaigns"
    d.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_run_dir(runs, "run_final_v5")

    with _patch_dir(d), patch("services.campaign_service.RUNS_DIR", runs):
        cid = create_campaign("run_base", max_variations=5)
        # Simulate completed campaign with last run = run_final_v5
        data = json.loads((d / f"{cid}.json").read_text())
        data["status"] = "completed"
        data["current_run_id"] = "run_final_v5"
        (d / f"{cid}.json").write_text(json.dumps(data))

        with patch("services.campaign_service.generate_variation",
                   return_value=({}, {"max_turns": 8}, "escalate reason")) as mock_gen, \
             patch("services.campaign_service.enqueue_train_job",
                   return_value={"id": 2, "run_dir": str(runs / "run_final_v5_esc1")}):
            escalate_campaign(cid)
            result = get_campaign(cid)

    # escalate should set status back to running and set escalate_mode
    assert result["status"] == "running"
    assert result["escalate_mode"] is True
    # generate_variation should have been called with escalate=True
    _, kwargs = mock_gen.call_args
    assert kwargs.get("escalate") is True or mock_gen.call_args[0][3] is True
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_service.py::test_escalate_queues_job_with_escalate_mode -v
```
Expected: `AttributeError` — `escalate_campaign` not defined

- [ ] **Step 3: Implement escalate_campaign**

Add to `campaign_service.py`:

```python
def escalate_campaign(campaign_id: str) -> None:
    campaign = _read_campaign(campaign_id)
    last_run_id = campaign["current_run_id"]

    run_dir = RUNS_DIR / last_run_id
    try:
        import json as _json
        config_dict = _json.loads((run_dir / "game_config.json").read_text())
    except (OSError, ValueError):
        config_dict = {}

    # Use neutral metrics for escalate (no specific weakness targeting)
    metrics = {"latest_reward": 0, "bankruptcy_rate": 0,
               "average_ending_money": 0, "invalid_action_rate": 0, "reward_cv": 0}

    campaign["status"] = "running"
    campaign["escalate_mode"] = True
    campaign["max_variations"] = campaign["variations_completed"] + 1
    _write_campaign(campaign)

    _generate_and_queue_next(campaign, last_run_id, metrics, escalate=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_service.py -v
```
Expected: all 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add game/v2_deep_rl/control_center/backend/services/campaign_service.py \
        game/v2_deep_rl/tests/test_campaign_service.py
git commit -m "feat: implement escalate_campaign for major-bounds rule variation"
```

---

## Task 6: Autopilot integration

**Files:**
- Modify: `game/v2_deep_rl/control_center/backend/services/training_autopilot.py`

The stop branch is at lines 555–581. After `_write_decision_record(run_dir, decision)` (line 580) and before `return decision` (line 581), add the campaign hook.

- [ ] **Step 1: Write the failing test**

Add a new test file:

```python
# tests/test_campaign_autopilot_integration.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest


LOG_HEADER = [
    "episode", "epsilon", "episode_reward", "rolling_average_reward",
    "mean_recent_loss", "replay_buffer_size", "average_loan_duration",
    "bankruptcy_count", "average_ending_money", "invalid_action_count",
    "action_0_count", "action_1_count", "action_2_count", "action_3_count",
    "action_4_count", "action_5_count", "action_6_count", "action_7_count",
]
EVAL_HEADER = ["episode", "average_reward", "bankruptcy_rate",
               "average_ending_money", "invalid_action_rate"]


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _make_plateau_run(runs_dir: Path, run_id: str) -> None:
    run_dir = runs_dir / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "latest_scrum_model.pth").touch()
    (run_dir / "checkpoints" / "best_scrum_model.pth").touch()
    (run_dir / "game_config.json").write_text(json.dumps({
        "schema_version": "2.0", "config_name": "T", "config_description": "",
        "players_count": 1,
        "product_names": ["P1","P2","P3","P4","P5","P6","P7"],
        "max_turns": 6, "starting_money": 25000, "ring_value": 10000,
        "cost_continue": 0, "cost_switch_mid": 5000, "cost_switch_after": 2500,
        "mandatory_loan_amount": 50000, "loan_interest": 5000,
        "penalty_negative": -100, "penalty_positive": 0,
        "daily_scrums_per_sprint": 3, "daily_scrum_target": 3,
        "board_ring_values": [[0]*7]*4, "board_features": [[3]*7]*4,
        "dice_rules": [{"feature_min":1,"feature_max":4,"dice_count":1,"dice_sides":6}],
        "refinement": {"active": False, "product_rules": []},
        "incident": {"active": True, "allow_player_specific_incidents": False,
                     "draw_probability": 1.0, "severity_multiplier": 1.0, "cards": []},
        "reserved_fields": {},
    }))
    (run_dir / "training_config.json").write_text(json.dumps({
        "learning_rate": 0.0005,
        "epsilon_decay_episodes": 450000,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
    }))
    flat_rewards = [50000.0] * 8
    eval_rows = [[i * 10000, r, 0.2, 40000.0, 0.05]
                 for i, r in enumerate(flat_rewards)]
    _write_csv(run_dir / "reports" / "evaluation_history.csv", EVAL_HEADER, eval_rows)
    log_rows = [[100000, 0.04, 50000, 50000, 0.01, 1000, 0, 0, 40000, 5] + [10]*8]
    _write_csv(run_dir / "reports" / "logs.csv", LOG_HEADER, log_rows)


def test_run_autopilot_stop_notifies_campaign(tmp_path):
    from services.training_autopilot import run_autopilot
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_plateau_run(runs_dir, "run_plateau")

    with patch("services.training_autopilot.RUNS_DIR", runs_dir), \
         patch("services.training_autopilot.ARTIFACTS_DIR", tmp_path), \
         patch("services.training_autopilot.get_settings",
               return_value={"logic_enabled": True, "ai_enabled": False}), \
         patch("services.campaign_service.CAMPAIGNS_DIR", tmp_path / "campaigns"), \
         patch("services.campaign_service.RUNS_DIR", runs_dir), \
         patch("services.campaign_service.generate_variation",
               return_value=({}, {}, "test")) as mock_gen, \
         patch("services.campaign_service.enqueue_train_job",
               return_value={"id": 1, "run_dir": str(runs_dir / "run_plateau_cv1")}):
        from services.campaign_service import create_campaign
        (tmp_path / "campaigns").mkdir(exist_ok=True)
        cid = create_campaign("run_plateau", max_variations=3)

        decision = run_autopilot("run_plateau", dry_run=False)

    if decision["action"] in {"stop", "stop_regression"}:
        mock_gen.assert_called_once()
```

- [ ] **Step 2: Run to verify current state (test may skip if run doesn't plateau)**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_autopilot_integration.py -v
```
Expected: test runs but `mock_gen.assert_called_once()` fails (campaign hook not in autopilot yet)

- [ ] **Step 3: Add campaign hook to training_autopilot.py**

In `training_autopilot.py`, find the stop branch. It looks like this around lines 555–581:

```python
    if dry_run or decision["action"] in {"stop", "stop_regression"} or not decision["next_payload"]:
        decision["job_enqueued"] = False
        # ... robustness eval lines ...
        _write_decision_record(run_dir, decision)
        return decision
```

Change the final two lines of the stop branch from:
```python
        _write_decision_record(run_dir, decision)
        return decision
```
to:
```python
        _write_decision_record(run_dir, decision)
        if not dry_run and decision["action"] in {"stop", "stop_regression"}:
            try:
                from services.campaign_service import get_campaign_for_run, on_run_stopped
                campaign = get_campaign_for_run(run_id)
                if campaign:
                    on_run_stopped(run_id, decision.get("metrics", {}))
            except Exception:
                pass  # never let campaign errors break the autopilot return
        return decision
```

- [ ] **Step 4: Run integration test**

```bash
cd game/v2_deep_rl
python -m pytest tests/test_campaign_autopilot_integration.py -v
```
Expected: PASS (or skip if plateau not triggered — verify by checking decision action in test output)

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
cd game/v2_deep_rl
python -m pytest tests/ -v
```
Expected: all existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add game/v2_deep_rl/control_center/backend/services/training_autopilot.py \
        game/v2_deep_rl/tests/test_campaign_autopilot_integration.py
git commit -m "feat: hook campaign service into autopilot stop decision"
```

---

## Task 7: REST API routes + app.py registration

**Files:**
- Create: `game/v2_deep_rl/control_center/backend/api/routes_campaigns.py`
- Modify: `game/v2_deep_rl/control_center/backend/app.py`

- [ ] **Step 1: Implement routes_campaigns.py**

```python
# control_center/backend/api/routes_campaigns.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.campaign_service import (
    create_campaign,
    escalate_campaign,
    get_campaign,
    list_campaigns,
    stop_campaign,
)

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


class CreateCampaignRequest(BaseModel):
    run_id: str
    max_variations: int = 5


@router.post("")
def post_create_campaign(body: CreateCampaignRequest) -> dict:
    campaign_id = create_campaign(body.run_id, max_variations=body.max_variations)
    return get_campaign(campaign_id)


@router.get("")
def get_list_campaigns() -> list:
    return list_campaigns()


@router.get("/{campaign_id}")
def get_one_campaign(campaign_id: str) -> dict:
    try:
        return get_campaign(campaign_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id!r} not found")


@router.post("/{campaign_id}/stop")
def post_stop_campaign(campaign_id: str) -> dict:
    try:
        stop_campaign(campaign_id)
        return get_campaign(campaign_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id!r} not found")


@router.post("/{campaign_id}/escalate")
def post_escalate_campaign(campaign_id: str) -> dict:
    try:
        escalate_campaign(campaign_id)
        return get_campaign(campaign_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id!r} not found")
```

- [ ] **Step 2: Register router in app.py**

In `control_center/backend/app.py`:

Add import after line 13 (after the last existing router import):
```python
from api.routes_campaigns import router as campaigns_router
```

Add registration after line 41 (after the last `app.include_router` call):
```python
app.include_router(campaigns_router)
```

- [ ] **Step 3: Verify the API starts without errors**

```bash
cd game/v2_deep_rl/control_center/backend
python -c "from app import app; print('OK')"
```
Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add game/v2_deep_rl/control_center/backend/api/routes_campaigns.py \
        game/v2_deep_rl/control_center/backend/app.py
git commit -m "feat: add campaigns REST API routes"
```

---

## Task 8: Frontend — campaign toggle on launch form

**Files:**
- Modify: `game/v2_deep_rl/control_center/frontend/index.html`

The `trainJobForm` is at line ~286. It ends with a `<button>Queue Training Job</button>` submit button around line ~320.

- [ ] **Step 1: Add campaign fields to the launch form**

In `index.html`, find:
```html
              <button class="button primary" type="submit">Queue Training Job</button>
```

Replace with:
```html
              <div class="field-group" style="border-top: 1px solid var(--border); padding-top: 0.75rem; margin-top: 0.25rem;">
                <label class="field" style="flex-direction: row; align-items: center; gap: 0.5rem;">
                  <input id="campaignEnabledToggle" type="checkbox" />
                  <span>Start as Campaign</span>
                </label>
                <label class="field" id="campaignMaxVarField" style="display:none;">
                  <span>Max Variations</span>
                  <input id="campaignMaxVariationsInput" type="number" min="1" max="20" value="5" />
                </label>
              </div>
              <button class="button primary" type="submit">Queue Training Job</button>
```

- [ ] **Step 2: Wire the campaign toggle visibility**

Find the `<script>` section in index.html. Locate the `trainJobForm` submit handler (search for `trainJobForm`). Just before that handler, add:

```javascript
document.getElementById('campaignEnabledToggle').addEventListener('change', function() {
  document.getElementById('campaignMaxVarField').style.display = this.checked ? '' : 'none';
});
```

- [ ] **Step 3: Add campaign creation to the form submit handler**

Find the `trainJobForm` submit handler. It calls something like `fetch('/api/jobs/train', ...)`. After a successful job enqueue, add:

```javascript
// Inside the .then() success callback of the train job fetch, after job is enqueued:
const campaignEnabled = document.getElementById('campaignEnabledToggle').checked;
if (campaignEnabled) {
  const maxVariations = parseInt(document.getElementById('campaignMaxVariationsInput').value, 10) || 5;
  const runDir = job.run_dir || '';
  const runId = runDir.split('/').pop().split('\\').pop();
  if (runId) {
    fetch(`${apiBase}/campaigns`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_id: runId, max_variations: maxVariations }),
    }).then(() => refreshCampaigns());
  }
}
```

Note: You will need to read the actual submit handler to find the exact insertion point. Search for `trainJobForm` in index.html and follow the fetch chain.

- [ ] **Step 4: Commit**

```bash
git add game/v2_deep_rl/control_center/frontend/index.html
git commit -m "feat: add campaign toggle to training launch form"
```

---

## Task 9: Frontend — campaign status panel + escalate button

**Files:**
- Modify: `game/v2_deep_rl/control_center/frontend/index.html`

- [ ] **Step 1: Add the campaign panel to the training page**

In `index.html`, find the training page's autopilot article (around line 338):
```html
          <article class="panel">
            <div class="panel-head">
              <h3>Autopilot</h3>
```

Add a new article after the closing `</article>` of the autopilot panel (around line 345):

```html
          <article class="panel" id="campaignPanelWrapper">
            <div class="panel-head">
              <h3>Campaign</h3>
              <span id="campaignStatusLabel" class="meta-count">-</span>
              <button id="stopCampaignButton" class="button secondary" type="button" style="display:none;">Stop Campaign</button>
            </div>
            <div id="campaignCard" class="empty-state">No active campaign.</div>
            <button id="escalateCampaignButton" class="button primary" type="button" style="display:none; margin-top: 0.5rem;">Escalate Rules →</button>
          </article>
```

- [ ] **Step 2: Add refreshCampaigns() JS function**

In the `<script>` section, add:

```javascript
let _activeCampaignId = null;

async function refreshCampaigns() {
  try {
    const campaigns = await apiFetch('/campaigns');
    const active = (campaigns || []).find(c => c.status === 'running');
    const completed = (campaigns || []).find(c => c.status === 'completed');
    const display = active || completed;

    if (!display) {
      document.getElementById('campaignCard').innerHTML = '<p class="muted">No active campaign.</p>';
      document.getElementById('campaignStatusLabel').textContent = '-';
      document.getElementById('stopCampaignButton').style.display = 'none';
      document.getElementById('escalateCampaignButton').style.display = 'none';
      _activeCampaignId = null;
      return;
    }

    _activeCampaignId = display.campaign_id;
    const done = display.variations_completed;
    const total = display.max_variations;
    const pct = total > 0 ? Math.round((done / total) * 100) : 0;
    const barFilled = Math.round(pct / 5);
    const bar = '█'.repeat(barFilled) + '░'.repeat(20 - barFilled);

    let historyHtml = '';
    (display.variation_history || []).forEach(h => {
      historyHtml += `<p class="muted" style="font-size:0.8em; margin: 0.2rem 0;">
        v${h.index}: ${h.reason || '—'}</p>`;
    });
    if (display.status === 'running') {
      historyHtml += `<p class="muted" style="font-size:0.8em;">v${done + 1}: currently running…</p>`;
    }

    document.getElementById('campaignCard').innerHTML = `
      <p><strong>${display.campaign_id}</strong></p>
      <p style="font-family: monospace; font-size: 0.85em;">${bar} ${done} / ${total}</p>
      ${historyHtml}
    `;
    document.getElementById('campaignStatusLabel').textContent = display.status;
    document.getElementById('stopCampaignButton').style.display = display.status === 'running' ? '' : 'none';
    document.getElementById('escalateCampaignButton').style.display = display.status === 'completed' ? '' : 'none';
  } catch (_) {}
}
```

- [ ] **Step 3: Wire stop and escalate buttons**

In the `<script>` section, add:

```javascript
document.getElementById('stopCampaignButton').addEventListener('click', async () => {
  if (!_activeCampaignId) return;
  await apiFetch(`/campaigns/${_activeCampaignId}/stop`, { method: 'POST' });
  refreshCampaigns();
});

document.getElementById('escalateCampaignButton').addEventListener('click', async () => {
  if (!_activeCampaignId) return;
  await apiFetch(`/campaigns/${_activeCampaignId}/escalate`, { method: 'POST' });
  refreshCampaigns();
});
```

- [ ] **Step 4: Call refreshCampaigns() in the existing refreshAll() function**

Find the `refreshAll()` or page-load fetch calls and add:
```javascript
refreshCampaigns();
```

- [ ] **Step 5: Commit**

```bash
git add game/v2_deep_rl/control_center/frontend/index.html
git commit -m "feat: add campaign status panel and escalate button to frontend"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Campaign created after training job enqueued (Task 8)
- ✅ Campaign persisted as JSON in `artifacts/campaigns/` (Task 1)
- ✅ Autopilot stop triggers on_run_stopped (Task 6)
- ✅ AI-driven config variation with bounded clamping (Tasks 2–3)
- ✅ Transfer learning: resume from latest checkpoint (Task 4)
- ✅ Max variations limit → status "completed" (Task 4)
- ✅ Escalate mode with wider bounds (Tasks 5, 9)
- ✅ Frontend campaign toggle on launch (Task 8)
- ✅ Frontend status panel showing variation history (Task 9)
- ✅ Stop campaign button (Task 9)
- ✅ REST API for all campaign operations (Task 7)

**Type consistency:**
- `on_run_stopped(run_id: str, metrics: dict)` — consistent across service, test, and autopilot hook
- `generate_variation(config_dict, metrics, variation_index, escalate)` — used identically in Task 3 and mocked in Tasks 4–5
- `_write_campaign(data: dict)` — exposed for test fixture setup in Task 4 tests
- `escalate_campaign(campaign_id)` — name consistent across service, route, and frontend call
