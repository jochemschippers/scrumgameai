"""
Campaign Config Mutation and Clamping Service.

This service generates mutated game configuration variations for robustness campaigns.
It queries an LLM to recommend changes targeting policy weaknesses (e.g. increasing/decreasing turn bounds, loans,
starting cash, or incident rates) based on previous run metrics. It then clamps recommendations strictly
against safety envelopes (Safe bounds vs. Escalation bounds) to ensure mutated configurations remain learnable.

Key Features:
  - Safety Clamping: Compares suggested mutations with percentage/delta/absolute bounds to prevent chaotic drift.
  - Config Patching: Maps flat recommended keys (like `dice_rule_0_dice_sides`) to nested dictionary fields.
  - Fallback Handling: Safely returns original configurations if LLM parsing or validation fails.

Connections:
  - Used by: `services/campaign_service.py` to produce config files for successive campaign iterations.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised through tests by patching OpenAI
    OpenAI = None


NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "abacusai/dracarys-llama-3.1-70b-instruct")

_SAFE_BOUNDS: dict[str, tuple] = {
    "starting_money": ("pct", 0.25),
    "max_turns": ("delta", 2, 3, 15),
    "penalty_negative": ("pct", 0.30),
    "penalty_positive": ("pct", 0.30),
    "mandatory_loan_amount": ("pct", 0.20),
    "loan_interest": ("pct", 0.20),
    "incident_draw_probability": ("abs", 0.5, 1.0),
    "incident_severity_multiplier": ("abs", 0.5, 2.0),
    **{f"dice_rule_{i}_dice_sides": ("delta", 2, 4, 20) for i in range(6)},
    **{f"dice_rule_{i}_dice_count": ("delta", 1, 1, 3) for i in range(6)},
}

_ESCALATE_BOUNDS: dict[str, tuple] = {
    "starting_money": ("pct", 0.60),
    "max_turns": ("abs", 3, 15),
    "penalty_negative": ("pct", 0.60),
    "penalty_positive": ("pct", 0.60),
    "mandatory_loan_amount": ("pct", 0.40),
    "loan_interest": ("pct", 0.40),
    "incident_draw_probability": ("abs", 0.2, 1.0),
    "incident_severity_multiplier": ("abs", 0.2, 3.0),
    **{f"dice_rule_{i}_dice_sides": ("abs", 4, 20) for i in range(6)},
    **{f"dice_rule_{i}_dice_count": ("abs", 1, 4) for i in range(6)},
}


def _get_current(key: str, config_dict: dict) -> float | None:
    """Retrieve the current value of a configuration key, handling nested keys and lists."""
    if key == "incident_draw_probability":
        return config_dict.get("incident", {}).get("draw_probability")
    if key == "incident_severity_multiplier":
        return config_dict.get("incident", {}).get("severity_multiplier")
    if key.startswith("dice_rule_"):
        parts = key.split("_")
        if len(parts) < 5:
            return None
        idx = int(parts[2])
        field = "_".join(parts[3:])
        rules = config_dict.get("dice_rules", [])
        if idx < len(rules):
            return rules[idx].get(field)
        return None
    return config_dict.get(key)


def _clamp_one(value: float, current: float, bounds: tuple) -> float:
    """Clamp a mutated value according to its scaling/boundary rules (percentage, delta, or absolute)."""
    btype = bounds[0]
    if btype == "pct":
        pct = bounds[1]
        lo, hi = current * (1 - pct), current * (1 + pct)
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
    """Return supported changes clamped relative to the current configuration."""
    bounds_map = _ESCALATE_BOUNDS if escalate else _SAFE_BOUNDS
    result = {}
    for key, value in diff.items():
        if key not in bounds_map:
            continue
        current = _get_current(key, config_dict)
        if current is None:
            continue
        clamped = _clamp_one(float(value), float(current), bounds_map[key])
        if isinstance(current, int):
            clamped = int(round(clamped))
        result[key] = clamped
    return result


def apply_diff_to_config(config_dict: dict, diff: dict) -> dict:
    """Apply the campaign API's flat change format to a copied nested config."""
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
            rules = result.setdefault("dice_rules", [])
            while len(rules) <= idx:
                rules.append({})
            rules[idx][field] = value
        else:
            result[key] = value
    return result


def _bounds_summary(escalate: bool) -> str:
    """Return a human-readable summary description of the active safety bounds envelope."""
    if escalate:
        return (
            "starting_money +/-60%; max_turns 3-15; penalties +/-60%; "
            "loan values +/-40%; incident_draw_probability 0.2-1.0; "
            "incident_severity_multiplier 0.2-3.0; dice sides 4-20; dice count 1-4."
        )
    return (
        "starting_money +/-25%; max_turns +/-2 within 3-15; penalties +/-30%; "
        "loan values +/-20%; incident_draw_probability 0.5-1.0; "
        "incident_severity_multiplier 0.5-2.0; dice sides +/-2 within 4-20; "
        "dice count +/-1 within 1-3."
    )


def _build_prompt(config_dict: dict, metrics: dict, variation_index: int, escalate: bool) -> str:
    """Construct the hyperparameter/rule tuning prompt detailing situation, bounds, and requirements."""
    return f"""You are a game-rules advisor for a Scrum board game RL training campaign.
Suggest 1-3 bounded game config changes that create a meaningfully different but learnable
environment for variation {variation_index}. Target observed weaknesses in the metrics.

Current fields:
starting_money: {config_dict.get("starting_money")}
max_turns: {config_dict.get("max_turns")}
penalty_negative: {config_dict.get("penalty_negative")}
penalty_positive: {config_dict.get("penalty_positive")}
mandatory_loan_amount: {config_dict.get("mandatory_loan_amount")}
loan_interest: {config_dict.get("loan_interest")}
incident_draw_probability: {config_dict.get("incident", {}).get("draw_probability")}
incident_severity_multiplier: {config_dict.get("incident", {}).get("severity_multiplier")}
dice_rules: {json.dumps(config_dict.get("dice_rules", []))}

Run metrics:
latest_average_reward: {metrics.get("latest_reward")}
bankruptcy_rate: {metrics.get("bankruptcy_rate")}
average_ending_money: {metrics.get("average_ending_money")}
invalid_action_rate: {metrics.get("invalid_action_rate")}
reward_cv: {metrics.get("reward_cv")}

Allowed keys:
starting_money, max_turns, penalty_negative, penalty_positive, mandatory_loan_amount,
loan_interest, incident_draw_probability, incident_severity_multiplier,
dice_rule_N_dice_sides, dice_rule_N_dice_count.

Bounds: {_bounds_summary(escalate)}

Respond with JSON only:
{{"changes": {{"<key>": <value>}}, "reason": "<one sentence>"}}"""


def generate_variation(
    config_dict: dict,
    metrics: dict,
    variation_index: int,
    escalate: bool = False,
) -> tuple[dict, dict, str]:
    """
    Ask the AI for a bounded game config variation.

    Returns (new_config_dict, changes_applied, reason). Failures are conservative:
    the original config is returned unchanged with an explanatory fallback reason.
    """
    if OpenAI is None:
        return config_dict, {}, "fallback: openai package not installed"

    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": _build_prompt(config_dict, metrics, variation_index, escalate)}],
            temperature=0.5,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        return config_dict, {}, f"fallback: AI call failed ({exc})"

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        suggestion: dict[str, Any] = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return config_dict, {}, "fallback: AI returned unparseable response"

    clamped = clamp_diff(suggestion.get("changes", {}), config_dict, escalate=escalate)
    if not clamped:
        return config_dict, {}, "fallback: no valid changes after clamping"

    return apply_diff_to_config(config_dict, clamped), clamped, str(suggestion.get("reason", "AI variation"))
