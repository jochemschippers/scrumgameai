"""Implement rating behavior for the autopilot package."""

from __future__ import annotations

from services.app_paths import RUNS_DIR
from services.io_utils import read_csv_rows, read_json_or_empty, safe_float


def compute_run_rating(run_id: str) -> dict:
    """
    Compute a 0–100 quality score for a completed run from its evaluation history.

    Scoring breakdown (100 pts total):
      Bankruptcy rate   35 pts  — primary survival metric
      Average reward    35 pts  — primary learning objective
      Avg turns played  20 pts  — proxy for game depth / longevity
      Invalid actions   10 pts  — exploration quality signal

    Returns a dict with keys: score, grade, breakdown, data_source, episodes_evaluated.
    Returns grade="N/A" if no evaluation history is available.
    """
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise ValueError(f"Run `{run_id}` not found.")

    eval_rows = read_csv_rows(run_dir / "reports" / "evaluation_history.csv")

    # Fall back to dqn_metrics.json (final training summary) when evaluation_history.csv
    # has no rows — this happens for short runs that never hit the periodic eval interval.
    metrics_path = run_dir / "reports" / "dqn_metrics.json"
    if not eval_rows and metrics_path.exists():
        metrics_json = read_json_or_empty(metrics_path)
        bankruptcy_rate = safe_float(metrics_json.get("bankruptcy_rate"))
        average_reward = safe_float(metrics_json.get("average_reward_per_episode"))
        avg_loan_duration = safe_float(metrics_json.get("average_loan_duration"))
        invalid_action_rate = safe_float(metrics_json.get("invalid_action_rate"))
        latest_episode = safe_float(metrics_json.get("training_episodes"))
        data_source = "dqn_metrics"
    elif not eval_rows:
        return {
            "score": None,
            "grade": "N/A",
            "breakdown": {},
            "data_source": "evaluation_history",
            "episodes_evaluated": 0,
        }
    else:
        # Use average of last 3 windows for a stable estimate.
        sample = eval_rows[-3:]
        def _avg(key: str) -> float | None:
            """Calculate the mean value of a specified key across the sample rows."""
            vals = [safe_float(r.get(key)) for r in sample]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        bankruptcy_rate = _avg("bankruptcy_rate")
        average_reward = _avg("average_reward")
        avg_loan_duration = _avg("average_loan_duration")
        invalid_action_rate = _avg("invalid_action_rate")
        latest_episode = safe_float(eval_rows[-1].get("episode"))
        data_source = "evaluation_history"

    # --- Bankruptcy rate (35 pts) ---
    if bankruptcy_rate is None:
        br_score = 0
    elif bankruptcy_rate < 0.10:
        br_score = 35
    elif bankruptcy_rate < 0.20:
        br_score = 28
    elif bankruptcy_rate < 0.30:
        br_score = 20
    elif bankruptcy_rate < 0.50:
        br_score = 12
    elif bankruptcy_rate < 0.70:
        br_score = 5
    else:
        br_score = 0

    # --- Average reward (35 pts) ---
    if average_reward is None:
        rw_score = 0
    elif average_reward > 50_000:
        rw_score = 35
    elif average_reward > 0:
        rw_score = 25
    elif average_reward > -10_000:
        rw_score = 15
    elif average_reward > -25_000:
        rw_score = 8
    else:
        rw_score = 0

    # --- Avg loan duration (20 pts) ---
    # turns_with_loan resets to 0 each non-loan turn, so the episode value reflects
    # consecutive loan turns at game end. Low = agent cleared loans before finishing = good.
    if avg_loan_duration is None:
        turns_score = 0
    elif avg_loan_duration <= 1:
        turns_score = 20
    elif avg_loan_duration <= 2:
        turns_score = 15
    elif avg_loan_duration <= 3:
        turns_score = 10
    elif avg_loan_duration <= 4:
        turns_score = 5
    else:
        turns_score = 0

    # --- Invalid action rate (10 pts) ---
    if invalid_action_rate is None:
        ia_score = 10  # assume clean if no data
    elif invalid_action_rate < 0.01:
        ia_score = 10
    elif invalid_action_rate < 0.05:
        ia_score = 5
    else:
        ia_score = 0

    total = br_score + rw_score + turns_score + ia_score

    if total >= 90:
        grade = "S"
    elif total >= 80:
        grade = "A"
    elif total >= 70:
        grade = "B"
    elif total >= 60:
        grade = "C"
    elif total >= 50:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": total,
        "grade": grade,
        "breakdown": {
            "bankruptcy_rate_score": br_score,
            "bankruptcy_rate_max": 35,
            "reward_score": rw_score,
            "reward_max": 35,
            "turns_score": turns_score,
            "turns_max": 20,
            "invalid_action_score": ia_score,
            "invalid_action_max": 10,
        },
        "snapshot": {
            "bankruptcy_rate": bankruptcy_rate,
            "average_reward": average_reward,
            "average_loan_duration": avg_loan_duration,
            "invalid_action_rate": invalid_action_rate,
        },
        "data_source": data_source,
        "episodes_evaluated": int(latest_episode) if latest_episode is not None else None,
    }
