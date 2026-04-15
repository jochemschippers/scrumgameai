from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .app_paths import ARTIFACTS_DIR, RUNS_DIR

# Path for the user-requested stop flag (a plain file; existence = stop requested)
_STOP_FLAG_PATH = ARTIFACTS_DIR / "autopilot_stop_requested.flag"
_SETTINGS_PATH = ARTIFACTS_DIR / "autopilot_settings.json"

# --- Logic decision thresholds ---
IMPROVEMENT_MIN_RATIO = 0.02    # below this over the window = plateau
VARIANCE_THRESHOLD = 0.20       # coefficient of variation above this = unstable
PLATEAU_WINDOW = 4              # number of evaluation rows to assess
INVALID_ACTION_HIGH = 0.10      # invalid action rate above this = under-explored

# While epsilon is above this the agent is still heavily exploring.
# Plateau detection is meaningless during exploration, so autopilot
# always returns "continue" until the policy starts to stabilise.
EPSILON_EXPLORE_THRESHOLD = 0.50

CONTINUE_EPISODES = 50_000      # episodes per auto-continuation block
LR_REDUCTION_FACTOR = 0.5
EPSILON_EXTENSION_FACTOR = 1.25
MAX_LR_REDUCTIONS = 3           # stop oscillating after this many consecutive lower_lr decisions

# --- AI advisor settings ---
MAX_AI_INTERVENTIONS = 3        # AI gets this many chances before the run truly stops
NVIDIA_API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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",
)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "abacusai/dracarys-llama-3.1-70b-instruct"

# Safe bounds the AI is not allowed to exceed
_LR_MIN = 0.000_01
_LR_MAX = 0.001
_EPSILON_DECAY_MIN = 100_000
_EPSILON_DECAY_MAX = 1_000_000
_EPISODES_MIN = 10_000
_EPISODES_MAX = 100_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> float | None:
    try:
        result = float(v)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _derive_base_run_name(run_id: str) -> str:
    """Extract the user-given name from a run_id, stripping timestamp prefix and any _vN suffix."""
    match = re.match(r'^run_\d{4}-\d{2}-\d{2}_\d{4}(?:_(.+))?$', run_id)
    if not match:
        return ""
    name = match.group(1) or ""
    # Strip trailing _vN so chained runs all share the same base
    name = re.sub(r'_v\d+$', '', name)
    return name


def _write_decision_record(run_dir: Path, decision: dict) -> None:
    records_path = run_dir / "reports" / "autopilot_decisions.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision) + "\n")


# ---------------------------------------------------------------------------
# Stop-after-cycle flag
# ---------------------------------------------------------------------------

def get_settings() -> dict:
    """Return autopilot feature toggles (logic_enabled, ai_enabled)."""
    defaults = {"logic_enabled": True, "ai_enabled": True}
    if not _SETTINGS_PATH.exists():
        return defaults
    try:
        stored = _read_json(_SETTINGS_PATH)
        return {**defaults, **stored}
    except Exception:
        return defaults


def save_settings(payload: dict) -> dict:
    """Update one or both toggles and persist."""
    current = get_settings()
    if "logic_enabled" in payload:
        current["logic_enabled"] = bool(payload["logic_enabled"])
    if "ai_enabled" in payload:
        current["ai_enabled"] = bool(payload["ai_enabled"])
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    return current


def request_stop_after_cycle() -> None:
    """Signal the autopilot to stop after the current training block finishes."""
    _STOP_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STOP_FLAG_PATH.touch()


def clear_stop_request() -> None:
    """Clear a pending stop-after-cycle request so the autopilot resumes."""
    _STOP_FLAG_PATH.unlink(missing_ok=True)


def is_stop_requested() -> bool:
    """Return True if the user has requested stop-after-cycle."""
    return _STOP_FLAG_PATH.exists()


# ---------------------------------------------------------------------------
# AI advisor (called only when logic decides to stop)
# ---------------------------------------------------------------------------

def _call_ai_advisor(
    metrics: dict,
    current_config: dict,
    stop_reason: str,
    intervention_number: int,
    best_checkpoint_path: str | None,
) -> dict:
    """
    Ask the NVIDIA-hosted LLM to suggest one bounded fine-tune adjustment.

    Returns a dict with keys:
      action          "fine_tune" or "stop"
      reason          str
      next_payload    dict | None  (training job payload if action == "fine_tune")
      advisor         "ai"
    """
    try:
        from openai import OpenAI  # optional dependency; fail gracefully
    except ImportError:
        return {
            "action": "stop",
            "reason": "AI advisor unavailable (openai package not installed). Stopping.",
            "next_payload": None,
            "advisor": "ai",
        }

    prompt = f"""You are a hyperparameter tuning advisor for a Double DQN reinforcement learning agent
that learns to play a Scrum project-management game.

The deterministic autopilot has decided to STOP training for the following reason:
"{stop_reason}"

This is AI intervention {intervention_number} of {MAX_AI_INTERVENTIONS}.
After {MAX_AI_INTERVENTIONS} interventions the run must stop regardless of your suggestion.

=== Recent evaluation metrics ===
Evaluation windows analysed : {metrics.get("eval_windows_analyzed")}
Latest average reward        : {metrics.get("latest_reward")}
Reward improvement ratio     : {metrics.get("reward_improvement_ratio")}
Reward coefficient of var.   : {metrics.get("reward_cv")}
Bankruptcy rate              : {metrics.get("bankruptcy_rate")}
Invalid action rate          : {metrics.get("invalid_action_rate")}

=== Current hyperparameters ===
learning_rate          : {current_config.get("learning_rate")}
epsilon_decay_episodes : {current_config.get("epsilon_decay_episodes")}

=== Your task ===
Decide whether ONE specific hyperparameter change is likely to help the agent escape
the current plateau, or whether the run should truly stop.

Respond with a JSON object and nothing else:
{{
  "action": "fine_tune" or "stop",
  "reason": "<one sentence>",
  "learning_rate": <float between {_LR_MIN} and {_LR_MAX}, optional>,
  "epsilon_decay_episodes": <int between {_EPSILON_DECAY_MIN} and {_EPSILON_DECAY_MAX}, optional>,
  "episodes": <int between {_EPISODES_MIN} and {_EPISODES_MAX}, default {CONTINUE_EPISODES}>
}}

Only include learning_rate or epsilon_decay_episodes if you are changing them.
Do not suggest rule changes or anything outside these two hyperparameters."""

    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            top_p=1,
            max_tokens=256,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        return {
            "action": "stop",
            "reason": f"AI advisor call failed ({exc}). Stopping.",
            "next_payload": None,
            "advisor": "ai",
        }

    # Extract the JSON block from the response
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        suggestion = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {
            "action": "stop",
            "reason": f"AI advisor returned unparseable response. Stopping. Raw: {raw[:200]}",
            "next_payload": None,
            "advisor": "ai",
        }

    ai_action = suggestion.get("action", "stop")
    ai_reason = suggestion.get("reason", "")

    if ai_action != "fine_tune":
        return {
            "action": "stop",
            "reason": f"AI advisor confirmed stop. {ai_reason}",
            "next_payload": None,
            "advisor": "ai",
        }

    # Clamp values to safe bounds
    new_lr = current_config["learning_rate"]
    if "learning_rate" in suggestion:
        new_lr = max(_LR_MIN, min(_LR_MAX, float(suggestion["learning_rate"])))

    new_epsilon_decay = current_config["epsilon_decay_episodes"]
    if "epsilon_decay_episodes" in suggestion:
        new_epsilon_decay = max(
            _EPSILON_DECAY_MIN,
            min(_EPSILON_DECAY_MAX, int(suggestion["epsilon_decay_episodes"])),
        )

    episodes = max(_EPISODES_MIN, min(_EPISODES_MAX, int(suggestion.get("episodes", CONTINUE_EPISODES))))

    return {
        "action": "fine_tune",
        "reason": f"AI advisor (intervention {intervention_number}/{MAX_AI_INTERVENTIONS}): {ai_reason}",
        "next_payload": {
            "episodes": episodes,
            "learning_rate": new_lr,
            "epsilon_decay_episodes": new_epsilon_decay,
            "resume_from": best_checkpoint_path,
            "resume_mode": "fine-tune",
            "resume_episodes_mode": "incremental",
        },
        "advisor": "ai",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_run(run_id: str, context: dict | None = None) -> dict:
    """
    Analyze a completed run and return an autopilot decision dict.
    Pure analysis — does not enqueue any job or write to disk.

    Decision logic:
      - continue:              reward still improving (>2% over window)
      - lower_lr:             reward improving but variance is high
                              (capped at MAX_LR_REDUCTIONS consecutive reductions)
      - extend_epsilon_decay: reward flat + high invalid action rate
      - stop:                 plateau detected — AI advisor may override this
    """
    context = context or {}
    lr_reduction_count = int(context.get("lr_reduction_count", 0))

    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise ValueError(f"Run `{run_id}` not found.")

    eval_rows = _read_csv(run_dir / "reports" / "evaluation_history.csv")
    training_config = _read_json(run_dir / "training_config.json")

    # Prefer latest_scrum_model.pth (final episode of this run) for continuation so
    # each cycle advances the episode counter rather than looping from the best-reward
    # episode (which may be earlier than where training actually reached).
    latest_checkpoint = run_dir / "checkpoints" / "latest_scrum_model.pth"
    best_checkpoint = run_dir / "checkpoints" / "best_scrum_model.pth"
    if latest_checkpoint.exists():
        best_checkpoint_path = str(latest_checkpoint)
    elif best_checkpoint.exists():
        best_checkpoint_path = str(best_checkpoint)
    else:
        best_checkpoint_path = None

    rewards = [_safe_float(r.get("average_reward")) for r in eval_rows]
    rewards = [r for r in rewards if r is not None]

    last_row = eval_rows[-1] if eval_rows else {}
    invalid_action_rate = _safe_float(last_row.get("invalid_action_rate")) or 0.0
    bankruptcy_rate = _safe_float(last_row.get("bankruptcy_rate")) or 0.0

    current_lr = float(training_config.get("learning_rate", 0.0005))
    current_epsilon_decay = int(training_config.get("epsilon_decay_episodes", 450000))

    # Read the latest epsilon from the training log.
    log_rows = _read_csv(run_dir / "reports" / "logs.csv")
    latest_log = log_rows[-1] if log_rows else {}
    latest_epsilon = _safe_float(latest_log.get("epsilon"))
    if latest_epsilon is None:
        latest_epsilon = 1.0  # assume unexplored if no log yet

    if latest_epsilon > EPSILON_EXPLORE_THRESHOLD:
        # Agent is still heavily exploring — plateau detection is unreliable.
        action = "continue"
        reason = (
            f"Still in exploration phase (epsilon={latest_epsilon:.3f} > {EPSILON_EXPLORE_THRESHOLD}). "
            f"Continuing without plateau check."
        )
        improvement = None
        cv = None
    elif len(rewards) < PLATEAU_WINDOW:
        action = "continue"
        reason = (
            f"Only {len(rewards)} evaluation windows recorded; "
            f"need {PLATEAU_WINDOW} to assess plateau. Continuing."
        )
        improvement = None
        cv = None
    else:
        window = rewards[-PLATEAU_WINDOW:]
        first, last = window[0], window[-1]
        improvement = (last - first) / abs(first) if first != 0 else 0.0

        mean = sum(window) / len(window)
        std = math.sqrt(sum((r - mean) ** 2 for r in window) / len(window))
        cv = std / abs(mean) if mean != 0 else 0.0

        if improvement > IMPROVEMENT_MIN_RATIO:
            if cv > VARIANCE_THRESHOLD and lr_reduction_count < MAX_LR_REDUCTIONS:
                action = "lower_lr"
                reason = (
                    f"Reward improving ({improvement:.1%} over {PLATEAU_WINDOW} windows) "
                    f"but results are noisy (CV={cv:.2f}). Reducing learning rate "
                    f"(reduction {lr_reduction_count + 1}/{MAX_LR_REDUCTIONS})."
                )
            elif cv > VARIANCE_THRESHOLD:
                # LR reduction cap reached — treat high variance as a plateau signal.
                action = "stop"
                reason = (
                    f"Reward improving ({improvement:.1%}) but variance remains high (CV={cv:.2f}) "
                    f"after {lr_reduction_count} LR reductions. Stopping."
                )
            else:
                action = "continue"
                reason = (
                    f"Reward improving steadily ({improvement:.1%} over {PLATEAU_WINDOW} windows, "
                    f"CV={cv:.2f}). Continuing unchanged."
                )
        else:
            if invalid_action_rate > INVALID_ACTION_HIGH:
                action = "extend_epsilon_decay"
                reason = (
                    f"Reward plateaued ({improvement:.1%} over {PLATEAU_WINDOW} windows) "
                    f"with high invalid action rate ({invalid_action_rate:.1%}). Extending exploration."
                )
            else:
                action = "stop"
                reason = (
                    f"Reward plateaued ({improvement:.1%} over {PLATEAU_WINDOW} windows). "
                    f"Bankruptcy rate: {bankruptcy_rate:.1%}."
                )

    next_payload = None
    if action != "stop":
        # Apply LR floor so repeated reductions can't go below the safe minimum.
        new_lr = max(_LR_MIN, current_lr * LR_REDUCTION_FACTOR) if action == "lower_lr" else current_lr
        # Epsilon decay: extend by adding episodes on top of the current decay period
        # rather than scaling the absolute value, so the extension is meaningful
        # regardless of where in training the continuation starts.
        new_epsilon_decay = (
            current_epsilon_decay + int(current_epsilon_decay * (EPSILON_EXTENSION_FACTOR - 1.0))
            if action == "extend_epsilon_decay"
            else current_epsilon_decay
        )
        resume_mode = "strict" if action == "continue" else "fine-tune"
        next_payload = {
            "episodes": CONTINUE_EPISODES,
            "learning_rate": new_lr,
            "epsilon_decay_episodes": new_epsilon_decay,
            "resume_from": best_checkpoint_path,
            "resume_mode": resume_mode,
            "resume_episodes_mode": "incremental",
        }

    return {
        "run_id": run_id,
        "action": action,
        "reason": reason,
        "advisor": "logic",
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "eval_windows_analyzed": len(rewards),
            "latest_reward": rewards[-1] if rewards else None,
            "latest_epsilon": latest_epsilon,
            "reward_improvement_ratio": improvement,
            "reward_cv": cv,
            "bankruptcy_rate": bankruptcy_rate,
            "invalid_action_rate": invalid_action_rate,
        },
        "current_config": {
            "learning_rate": current_lr,
            "epsilon_decay_episodes": current_epsilon_decay,
        },
        "context": {
            "lr_reduction_count": lr_reduction_count,
        },
        "best_checkpoint_path": best_checkpoint_path,
        "next_payload": next_payload,
    }


def run_autopilot(run_id: str, dry_run: bool = False, context: dict | None = None) -> dict:
    """
    Analyze run, record the decision to disk, and enqueue the next job.

    context is passed forward through job payloads to track state across runs:
      ai_intervention_count  int  how many AI-advised fine-tunes have run so far

    Flow:
      1. Logic classifies the run.
      2. If logic says stop AND intervention count < MAX_AI_INTERVENTIONS:
             AI advisor suggests a fine-tune OR confirms stop.
      3. If user requested stop-after-cycle: override to stop.
      4. Write decision record.
      5. Enqueue next job (if not stopping), carrying context + autopilot_after_completion.
    """
    from jobs.queue_manager import enqueue_train_job  # avoid circular at module load

    context = context or {}
    ai_intervention_count = int(context.get("ai_intervention_count", 0))
    lr_reduction_count = int(context.get("lr_reduction_count", 0))
    # base_run_name is carried forward so the whole chain shares the original name.
    # continuation_version tracks what vN suffix the *next* run should get (starts at 2).
    base_run_name = context.get("base_run_name") or _derive_base_run_name(run_id)
    continuation_version = int(context.get("continuation_version", 2))

    decision = analyze_run(run_id, context=context)

    # --- AI advisor: only when logic says stop and budget remains ---
    settings = get_settings()
    ai_enabled = settings.get("ai_enabled", True)
    if not dry_run and decision["action"] == "stop" and ai_enabled and ai_intervention_count < MAX_AI_INTERVENTIONS:
        ai_result = _call_ai_advisor(
            metrics=decision["metrics"],
            current_config=decision["current_config"],
            stop_reason=decision["reason"],
            intervention_number=ai_intervention_count + 1,
            best_checkpoint_path=decision["best_checkpoint_path"],
        )
        decision["ai_advisor"] = ai_result
        if ai_result["action"] == "fine_tune":
            decision["action"] = "fine_tune"
            decision["reason"] = ai_result["reason"]
            decision["advisor"] = "ai"
            decision["next_payload"] = ai_result["next_payload"]

    # --- User-requested stop overrides everything except dry_run ---
    if not dry_run and decision["action"] != "stop" and is_stop_requested():
        decision["action"] = "stop"
        decision["reason"] = "Stop requested by user via stop-after-cycle flag. " + decision["reason"]
        decision["next_payload"] = None
        clear_stop_request()

    run_dir = RUNS_DIR / run_id

    if dry_run or decision["action"] == "stop" or not decision["next_payload"]:
        decision["job_enqueued"] = False
        _write_decision_record(run_dir, decision)
        return decision

    # Build versioned run name: keep original name + v2, v3, …
    versioned_run_name = f"{base_run_name}_v{continuation_version}" if base_run_name else f"v{continuation_version}"

    # Build next context counters
    next_ai_count = ai_intervention_count + 1 if decision["advisor"] == "ai" else ai_intervention_count
    next_lr_reduction_count = lr_reduction_count + 1 if decision["action"] == "lower_lr" else lr_reduction_count

    payload = {
        **decision["next_payload"],
        "run_name": versioned_run_name,
        "autopilot_after_completion": True,
        "autopilot_context": {
            "ai_intervention_count": next_ai_count,
            "lr_reduction_count": next_lr_reduction_count,
            "base_run_name": base_run_name,
            "continuation_version": continuation_version + 1,
        },
    }
    job = enqueue_train_job(payload)
    decision["job_enqueued"] = True
    decision["job_id"] = job["id"]
    decision["job_status"] = job["status"]
    _write_decision_record(run_dir, decision)
    return decision


def probe_ai_advisor(metrics: dict | None = None, current_config: dict | None = None) -> dict:
    """
    Call the AI advisor with supplied (or default test) metrics and return its response.
    Use this to verify the NVIDIA API connection and model are working.
    """
    test_metrics = metrics or {
        "eval_windows_analyzed": 4,
        "latest_reward": -12.5,
        "reward_improvement_ratio": 0.005,
        "reward_cv": 0.08,
        "bankruptcy_rate": 0.35,
        "invalid_action_rate": 0.04,
    }
    test_config = current_config or {
        "learning_rate": 0.0005,
        "epsilon_decay_episodes": 450000,
    }
    stop_reason = (
        "Reward plateaued (0.5% over 4 windows). Bankruptcy rate: 35.0%."
    )
    result = _call_ai_advisor(
        metrics=test_metrics,
        current_config=test_config,
        stop_reason=stop_reason,
        intervention_number=1,
        best_checkpoint_path=None,
    )
    return {"probe": True, "test_metrics": test_metrics, "test_config": test_config, "result": result}


def get_autopilot_history(run_id: str) -> list[dict]:
    """Return all autopilot decisions recorded for a run, oldest first."""
    run_dir = RUNS_DIR / run_id
    records_path = run_dir / "reports" / "autopilot_decisions.jsonl"
    if not records_path.exists():
        return []
    records = []
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
