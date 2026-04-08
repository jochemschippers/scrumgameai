from __future__ import annotations

from pathlib import Path

from .app_paths import (
    CURRENT_CHECKPOINT_DIR,
    PLAYABLE_MODEL_V1_DIR,
    REFERENCE_V1_DIR,
    REPO_ROOT,
    RUNS_DIR,
    ensure_engine_import_path,
)
from .catalog_service import list_game_configs

ensure_engine_import_path()

from checkpoint_utils import build_agent_for_config, load_checkpoint_payload  # noqa: E402
from config_manager import compute_rule_signature, load_game_config  # noqa: E402


def _checkpoint_id(checkpoint_path: Path) -> str:
    return checkpoint_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _checkpoint_type(checkpoint_path: Path) -> str:
    if checkpoint_path.name == "best_scrum_model.pth" or checkpoint_path.name.startswith("best_scrum_model"):
        return "best"
    return "intermediate"


def _source_label(source_type: str, source_run: str | None) -> str:
    if source_type == "run":
        return source_run or "run"
    if source_type == "current_artifacts":
        return "current artifacts"
    if source_type == "reference_v1":
        return "reference v1"
    if source_type == "playable_model_v1":
        return "playableModelV1"
    return source_type


def _infer_shape_from_state_dict(state_dict) -> tuple[int | None, int | None]:
    first_weight = state_dict.get("network.0.weight")
    final_weight = state_dict.get("network.6.weight")
    state_dim = int(first_weight.shape[1]) if first_weight is not None and len(first_weight.shape) == 2 else None
    num_actions = int(final_weight.shape[0]) if final_weight is not None and len(final_weight.shape) == 2 else None
    return state_dim, num_actions


def _checkpoint_catalog_paths() -> list[tuple[Path, str, str | None]]:
    catalog = []

    if CURRENT_CHECKPOINT_DIR.exists():
        for checkpoint_path in sorted(CURRENT_CHECKPOINT_DIR.glob("*.pth")):
            catalog.append((checkpoint_path, "current_artifacts", "current_artifacts"))

    if REFERENCE_V1_DIR.exists():
        for checkpoint_path in sorted(REFERENCE_V1_DIR.glob("*.pth")):
            catalog.append((checkpoint_path, "reference_v1", None))

    if PLAYABLE_MODEL_V1_DIR.exists():
        for checkpoint_path in sorted(PLAYABLE_MODEL_V1_DIR.glob("*.pth")):
            catalog.append((checkpoint_path, "playable_model_v1", None))

    if RUNS_DIR.exists():
        for run_dir in sorted((path for path in RUNS_DIR.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True):
            checkpoint_dir = run_dir / "checkpoints"
            if not checkpoint_dir.exists():
                continue
            for checkpoint_path in sorted(checkpoint_dir.glob("*.pth")):
                catalog.append((checkpoint_path, "run", run_dir.name))

    return catalog


def _resolve_game_config_reference(game_config_id: str):
    candidate_path = Path(game_config_id)
    if candidate_path.exists():
        return load_game_config(candidate_path)

    for item in list_game_configs():
        if item["id"] == game_config_id or item["path"] == game_config_id:
            return load_game_config(item["path"])

    raise ValueError(f"Game config `{game_config_id}` was not found.")


def list_checkpoints() -> list[dict]:
    items = []
    for checkpoint_path, source_type, source_run in _checkpoint_catalog_paths():
        try:
            payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
            metadata = payload.get("metadata", {})
            state_dict = payload.get("model_state_dict", {})
            inferred_state_dim, inferred_num_actions = _infer_shape_from_state_dict(state_dict)
            checkpoint_state_dim = metadata.get("state_dim") or inferred_state_dim
            checkpoint_num_actions = metadata.get("num_actions") or inferred_num_actions
            legacy_checkpoint = bool(metadata.get("legacy_checkpoint", False))
            rule_signature = metadata.get("rule_signature")
            training_signature = metadata.get("training_signature")

            if legacy_checkpoint and not rule_signature:
                compatibility_status = "legacy-unknown"
            else:
                compatibility_status = "tracked"

            items.append(
                {
                    "id": _checkpoint_id(checkpoint_path),
                    "label": checkpoint_path.name,
                    "display_label": f"{_source_label(source_type, source_run)} | {checkpoint_path.name}",
                    "path": str(checkpoint_path),
                    "source_type": source_type,
                    "source_run": source_run,
                    "checkpoint_type": _checkpoint_type(checkpoint_path),
                    "checkpoint_format": "legacy" if legacy_checkpoint else "managed",
                    "legacy_read_only": legacy_checkpoint or source_type in {"reference_v1", "playable_model_v1"},
                    "rule_signature": rule_signature,
                    "training_signature": training_signature,
                    "state_dim": checkpoint_state_dim,
                    "num_actions": checkpoint_num_actions,
                    "compatibility_status": compatibility_status,
                }
            )
        except Exception as error:
            items.append(
                {
                    "id": _checkpoint_id(checkpoint_path),
                    "label": checkpoint_path.name,
                    "display_label": f"{_source_label(source_type, source_run)} | {checkpoint_path.name}",
                    "path": str(checkpoint_path),
                    "source_type": source_type,
                    "source_run": source_run,
                    "checkpoint_type": _checkpoint_type(checkpoint_path),
                    "checkpoint_format": "unknown",
                    "legacy_read_only": True,
                    "compatibility_status": "error",
                    "error": str(error),
                }
            )
    return items


def get_checkpoint_by_id(checkpoint_id: str) -> dict | None:
    for item in list_checkpoints():
        if item["id"] == checkpoint_id:
            return item
    return None


def get_checkpoint_compatibility(checkpoint_id: str, game_config_id: str) -> dict:
    checkpoint = get_checkpoint_by_id(checkpoint_id)
    if checkpoint is None:
        raise ValueError(f"Checkpoint `{checkpoint_id}` was not found.")

    target_game_config = _resolve_game_config_reference(game_config_id)
    target_rule_signature = compute_rule_signature(target_game_config)
    target_agent, _ = build_agent_for_config(target_game_config)

    payload = load_checkpoint_payload(checkpoint["path"], map_location="cpu")
    metadata = payload.get("metadata", {})
    checkpoint_rule_signature = metadata.get("rule_signature")
    state_dict = payload.get("model_state_dict", {})
    inferred_state_dim, inferred_num_actions = _infer_shape_from_state_dict(state_dict)
    checkpoint_state_dim = metadata.get("state_dim") or inferred_state_dim
    checkpoint_num_actions = metadata.get("num_actions") or inferred_num_actions
    legacy_checkpoint = bool(metadata.get("legacy_checkpoint", False))

    shape_compatible = (
        checkpoint_state_dim == target_agent.state_dim
        and checkpoint_num_actions == target_agent.num_actions
    )

    if checkpoint_rule_signature is None:
        strict_status = "legacy-unknown"
    elif checkpoint_rule_signature == target_rule_signature:
        strict_status = "compatible"
    else:
        strict_status = "incompatible"

    if strict_status == "compatible":
        fine_tune_status = "compatible" if shape_compatible else "incompatible"
    elif checkpoint_rule_signature is None:
        fine_tune_status = "legacy-shape-compatible" if shape_compatible else "incompatible"
    else:
        fine_tune_status = "compatible" if shape_compatible else "incompatible"

    return {
        "checkpoint_id": checkpoint["id"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_format": checkpoint["checkpoint_format"],
        "legacy_read_only": checkpoint["legacy_read_only"],
        "checkpoint_rule_signature": checkpoint_rule_signature,
        "target_rule_signature": target_rule_signature,
        "checkpoint_state_dim": checkpoint_state_dim,
        "checkpoint_num_actions": checkpoint_num_actions,
        "target_state_dim": target_agent.state_dim,
        "target_num_actions": target_agent.num_actions,
        "shape_compatible": shape_compatible,
        "strict_resume_status": strict_status,
        "fine_tune_status": fine_tune_status,
        "message": (
            "Legacy checkpoint has no stored rule signature; only shape-based fine-tune guidance is available."
            if checkpoint_rule_signature is None
            else "Checkpoint compatibility was evaluated against the selected game config."
        ),
        "legacy_checkpoint": legacy_checkpoint,
    }
