from __future__ import annotations

from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
CONTROL_CENTER_DIR = BACKEND_DIR.parent
ENGINE_ROOT = BACKEND_DIR.parent.parent
REPO_ROOT = ENGINE_ROOT.parent.parent
ARTIFACTS_DIR = ENGINE_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CURRENT_CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
REFERENCE_V1_DIR = ARTIFACTS_DIR / "reference_v1"
PLAYABLE_MODEL_V1_DIR = REPO_ROOT / "playableModelV1"
CONFIGS_DIR = ENGINE_ROOT / "configs"
CUSTOM_GAME_CONFIG_DIR = CONFIGS_DIR / "custom"
TRAINING_CONFIG_DIR = CONFIGS_DIR / "training"
DEFAULT_GAME_CONFIG_PATH = CONFIGS_DIR / "default_game_config.json"
DEFAULT_TRAINING_CONFIG_PATH = CONFIGS_DIR / "default_training_config.json"


def ensure_engine_import_path() -> None:
    """Make the deep-RL engine modules importable from the backend package."""
    engine_path = str(ENGINE_ROOT)
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
