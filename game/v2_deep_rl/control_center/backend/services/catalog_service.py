from __future__ import annotations

import math
import json
import csv
from pathlib import Path
import re

from .app_paths import (
    CUSTOM_GAME_CONFIG_DIR,
    DEFAULT_GAME_CONFIG_PATH,
    DEFAULT_TRAINING_CONFIG_PATH,
    RUNS_DIR,
    TRAINING_CONFIG_DIR,
    ensure_engine_import_path,
)

ensure_engine_import_path()

from config_manager import (  # noqa: E402
    GameConfig,
    TrainingConfig,
    compute_rule_signature,
    compute_training_signature,
    load_game_config,
    load_training_config,
    save_game_config,
    save_training_config,
    validate_game_config,
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sanitize_json_value(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    return value


def _read_json_safe(path: Path):
    return _sanitize_json_value(_read_json(path))


def _safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _tail_csv_rows(path: Path, limit: int = 200) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit <= 0:
        return rows
    return rows[-limit:]


def _slugify_name(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "")).strip("_").lower()
    return slug or fallback


def _resolve_game_config_path(config_id_or_path: str) -> Path:
    candidate_path = Path(config_id_or_path)
    if candidate_path.exists():
        return candidate_path.resolve()

    if config_id_or_path == "default_game_config":
        return DEFAULT_GAME_CONFIG_PATH.resolve()

    for item in list_game_configs():
        if item["id"] == config_id_or_path:
            return Path(item["path"]).resolve()

    raise ValueError(f"Game config `{config_id_or_path}` was not found.")


def _resolve_training_config_path(config_id_or_path: str) -> Path:
    candidate_path = Path(config_id_or_path)
    if candidate_path.exists():
        return candidate_path.resolve()

    if config_id_or_path == "default_training_config":
        return DEFAULT_TRAINING_CONFIG_PATH.resolve()

    for item in list_training_configs():
        if item["id"] == config_id_or_path:
            return Path(item["path"]).resolve()

    raise ValueError(f"Training config `{config_id_or_path}` was not found.")


def list_game_configs() -> list[dict]:
    configs = []
    default_config = load_game_config(DEFAULT_GAME_CONFIG_PATH)
    configs.append(
        {
            "id": "default_game_config",
            "label": "Default Bundled Config",
            "path": str(DEFAULT_GAME_CONFIG_PATH),
            "rule_signature": compute_rule_signature(default_config),
            "products_count": default_config.products_count,
            "sprints_per_product": default_config.sprints_per_product,
            "config_name": default_config.config_name,
            "source": "bundled",
        }
    )

    if CUSTOM_GAME_CONFIG_DIR.exists():
        for config_path in sorted(CUSTOM_GAME_CONFIG_DIR.glob("*.json")):
            config = load_game_config(config_path)
            configs.append(
                {
                    "id": config_path.stem,
                    "label": config_path.name,
                    "path": str(config_path),
                    "rule_signature": compute_rule_signature(config),
                    "products_count": config.products_count,
                    "sprints_per_product": config.sprints_per_product,
                    "config_name": config.config_name,
                    "source": "custom",
                }
            )
    return configs


def get_game_config(config_id_or_path: str) -> dict:
    config_path = _resolve_game_config_path(config_id_or_path)
    config = load_game_config(config_path)
    return {
        "id": "default_game_config" if config_path.resolve() == DEFAULT_GAME_CONFIG_PATH.resolve() else config_path.stem,
        "label": config_path.name,
        "path": str(config_path),
        "source": "bundled" if config_path.resolve() == DEFAULT_GAME_CONFIG_PATH.resolve() else "custom",
        "rule_signature": compute_rule_signature(config),
        "config": config.to_dict(),
    }


def list_training_configs() -> list[dict]:
    configs = []
    default_config = load_training_config(DEFAULT_TRAINING_CONFIG_PATH)
    configs.append(
        {
            "id": "default_training_config",
            "label": "Default Bundled Training Config",
            "path": str(DEFAULT_TRAINING_CONFIG_PATH),
            "training_signature": compute_training_signature(default_config),
            "episodes": default_config.episodes,
            "learning_rate": default_config.learning_rate,
            "gamma": default_config.gamma,
            "source": "bundled",
        }
    )

    if TRAINING_CONFIG_DIR.exists():
        for config_path in sorted(TRAINING_CONFIG_DIR.glob("*.json")):
            if config_path.resolve() == DEFAULT_TRAINING_CONFIG_PATH.resolve():
                continue
            config = load_training_config(config_path)
            configs.append(
                {
                    "id": config_path.stem,
                    "label": config_path.name,
                    "path": str(config_path),
                    "training_signature": compute_training_signature(config),
                    "episodes": config.episodes,
                    "learning_rate": config.learning_rate,
                    "gamma": config.gamma,
                    "source": "custom",
                }
            )
    return configs


def get_training_config(config_id_or_path: str) -> dict:
    config_path = _resolve_training_config_path(config_id_or_path)
    config = load_training_config(config_path)
    return {
        "id": "default_training_config" if config_path.resolve() == DEFAULT_TRAINING_CONFIG_PATH.resolve() else config_path.stem,
        "label": config_path.name,
        "path": str(config_path),
        "source": "bundled" if config_path.resolve() == DEFAULT_TRAINING_CONFIG_PATH.resolve() else "custom",
        "training_signature": compute_training_signature(config),
        "config": config.to_dict(),
    }


def save_game_config_asset(payload: dict) -> dict:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Game config payload must include a `config` object.")

    game_config = GameConfig.from_dict(config_payload)
    target_id = payload.get("id")
    file_name = payload.get("file_name")

    if target_id:
        target_path = _resolve_game_config_path(target_id)
        if target_path.resolve() == DEFAULT_GAME_CONFIG_PATH.resolve():
            raise ValueError("Bundled default game config cannot be overwritten.")
        if target_path.parent.resolve() != CUSTOM_GAME_CONFIG_DIR.resolve():
            raise ValueError("Only managed custom game configs can be overwritten.")
    else:
        file_stub = _slugify_name(file_name or game_config.config_name, "game_config")
        target_path = CUSTOM_GAME_CONFIG_DIR / f"{file_stub}.json"

    CUSTOM_GAME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    save_game_config(game_config, target_path)
    return get_game_config(str(target_path))


def delete_game_config_asset(config_id_or_path: str) -> dict:
    target_path = _resolve_game_config_path(config_id_or_path)
    if target_path.resolve() == DEFAULT_GAME_CONFIG_PATH.resolve():
        raise ValueError("Bundled default game config cannot be deleted.")
    if target_path.parent.resolve() != CUSTOM_GAME_CONFIG_DIR.resolve():
        raise ValueError("Only managed custom game configs can be deleted.")
    if not target_path.exists():
        raise ValueError(f"Game config `{config_id_or_path}` was not found.")
    target_path.unlink()
    return {"deleted": True, "id": target_path.stem, "path": str(target_path)}


def validate_game_config_asset(payload: dict) -> dict:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Game config payload must include a `config` object.")
    game_config = GameConfig.from_dict(config_payload)
    validate_game_config(game_config)
    return {
        "valid": True,
        "rule_signature": compute_rule_signature(game_config),
        "products_count": game_config.products_count,
        "sprints_per_product": game_config.sprints_per_product,
        "actions_count": game_config.products_count + 1,
        "config_name": game_config.config_name,
    }


def save_training_config_asset(payload: dict) -> dict:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Training config payload must include a `config` object.")

    training_config = TrainingConfig.from_dict(config_payload)
    target_id = payload.get("id")
    file_name = payload.get("file_name")

    if target_id:
        target_path = _resolve_training_config_path(target_id)
        if target_path.resolve() == DEFAULT_TRAINING_CONFIG_PATH.resolve():
            raise ValueError("Bundled default training config cannot be overwritten.")
        if target_path.parent.resolve() != TRAINING_CONFIG_DIR.resolve():
            raise ValueError("Only managed custom training configs can be overwritten.")
    else:
        file_stub = _slugify_name(file_name or "training_config", "training_config")
        target_path = TRAINING_CONFIG_DIR / f"{file_stub}.json"

    TRAINING_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    save_training_config(training_config, target_path)
    return get_training_config(str(target_path))


def delete_training_config_asset(config_id_or_path: str) -> dict:
    target_path = _resolve_training_config_path(config_id_or_path)
    if target_path.resolve() == DEFAULT_TRAINING_CONFIG_PATH.resolve():
        raise ValueError("Bundled default training config cannot be deleted.")
    if target_path.parent.resolve() != TRAINING_CONFIG_DIR.resolve():
        raise ValueError("Only managed custom training configs can be deleted.")
    if not target_path.exists():
        raise ValueError(f"Training config `{config_id_or_path}` was not found.")
    target_path.unlink()
    return {"deleted": True, "id": target_path.stem, "path": str(target_path)}


def validate_training_config_asset(payload: dict) -> dict:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Training config payload must include a `config` object.")
    training_config = TrainingConfig.from_dict(config_payload)
    return {
        "valid": True,
        "training_signature": compute_training_signature(training_config),
        "episodes": training_config.episodes,
        "learning_rate": training_config.learning_rate,
        "gamma": training_config.gamma,
        "batch_size": training_config.batch_size,
    }


def list_runs() -> list[dict]:
    runs = []
    if not RUNS_DIR.exists():
        return runs

    for run_dir in sorted((path for path in RUNS_DIR.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True):
        metadata_path = run_dir / "run_metadata.json"
        metrics_path = run_dir / "reports" / "dqn_metrics.json"
        checkpoint_path = run_dir / "checkpoints" / "best_scrum_model.pth"
        metadata = _read_json_safe(metadata_path) if metadata_path.exists() else {}
        metrics = _read_json_safe(metrics_path) if metrics_path.exists() else {}

        runs.append(
            {
                "id": run_dir.name,
                "label": run_dir.name,
                "path": str(run_dir),
                "created_at": metadata.get("created_at"),
                "run_notes": metadata.get("run_notes", ""),
                "rule_signature": metadata.get("rule_signature"),
                "training_signature": metadata.get("training_signature"),
                "resume_mode": metadata.get("resume_mode"),
                "resume_checkpoint_path": metadata.get("resume_checkpoint_path"),
                "best_checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
                "metrics_path": str(metrics_path) if metrics_path.exists() else None,
                "average_reward_per_episode": metrics.get("average_reward_per_episode"),
                "bankruptcy_rate": metrics.get("bankruptcy_rate"),
            }
        )
    return runs


def get_run(run_id: str) -> dict | None:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    metadata_path = run_dir / "run_metadata.json"
    metrics_path = run_dir / "reports" / "dqn_metrics.json"
    game_config_path = run_dir / "game_config.json"
    training_config_path = run_dir / "training_config.json"

    return {
        "id": run_dir.name,
        "label": run_dir.name,
        "path": str(run_dir),
        "metadata": _read_json_safe(metadata_path) if metadata_path.exists() else {},
        "metrics": _read_json_safe(metrics_path) if metrics_path.exists() else {},
        "game_config": _read_json_safe(game_config_path) if game_config_path.exists() else None,
        "training_config": _read_json_safe(training_config_path) if training_config_path.exists() else None,
        "checkpoints": [
            {
                "name": checkpoint_path.name,
                "path": str(checkpoint_path),
            }
            for checkpoint_path in sorted((run_dir / "checkpoints").glob("*.pth"))
        ],
    }


def get_run_progress(run_id: str) -> dict | None:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    metadata_path = run_dir / "run_metadata.json"
    run_metadata = _read_json_safe(metadata_path) if metadata_path.exists() else {}
    training_config_path = run_dir / "training_config.json"
    training_config = _read_json_safe(training_config_path) if training_config_path.exists() else {}
    total_episodes = _safe_int(str(run_metadata.get("episodes_this_run", "")))
    if total_episodes is None:
        total_episodes = _safe_int(str(training_config.get("episodes", "")))
    start_episode = _safe_int(str(run_metadata.get("start_episode", ""))) or 1

    reports_dir = run_dir / "reports"
    training_rows = _tail_csv_rows(reports_dir / "logs.csv", limit=240)
    evaluation_rows = _tail_csv_rows(reports_dir / "evaluation_history.csv", limit=120)

    training_series = []
    for row in training_rows:
        episode = _safe_int(row.get("episode"))
        if episode is None:
            continue
        training_series.append(
            {
                "episode": episode,
                "episode_reward": _safe_float(row.get("episode_reward")),
                "rolling_average_reward": _safe_float(row.get("rolling_average_reward")),
                "mean_recent_loss": _safe_float(row.get("mean_recent_loss")),
                "average_ending_money": _safe_float(row.get("average_ending_money")),
                "epsilon": _safe_float(row.get("epsilon")),
            }
        )

    evaluation_series = []
    for row in evaluation_rows:
        episode = _safe_int(row.get("episode"))
        if episode is None:
            continue
        evaluation_series.append(
            {
                "episode": episode,
                "average_reward": _safe_float(row.get("average_reward")),
                "bankruptcy_rate": _safe_float(row.get("bankruptcy_rate")),
                "average_ending_money": _safe_float(row.get("average_ending_money")),
                "invalid_action_rate": _safe_float(row.get("invalid_action_rate")),
            }
        )

    latest_training = training_series[-1] if training_series else None
    latest_evaluation = evaluation_series[-1] if evaluation_series else None
    latest_episode = latest_training["episode"] if latest_training else 0
    completed_episodes = max(0, latest_episode - start_episode + 1)
    ratio = 0.0
    if total_episodes and total_episodes > 0:
        ratio = max(0.0, min(1.0, completed_episodes / total_episodes))

    return {
        "run_id": run_id,
        "job_id": None,
        "job_type": "train",
        "status": "completed",
        "run_dir": str(run_dir),
        "stdout_log_path": "",
        "error_message": None,
        "total_episodes": total_episodes,
        "start_episode": start_episode,
        "latest_episode": latest_episode,
        "completed_episodes": completed_episodes,
        "progress_ratio": ratio,
        "latest_training_row": latest_training,
        "latest_evaluation_row": latest_evaluation,
        "training_series": training_series,
        "evaluation_series": evaluation_series,
    }
