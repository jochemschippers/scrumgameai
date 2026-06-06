"""
Checkpoint Persistence and Compatibility Utilities for RL Models.

This module provides functions to save and load PyTorch model weights along with
compatibility metadata (rules, hyperparameters, state dimension, action dimension).

Three-File Checkpoint Scheme:
  1. Main Checkpoint (`[name].pth`):
     - Stores the full model state, target model state, optimizer state, training steps counter,
       and the COMPLETE ReplayBuffer.
     - Used for resumption of interrupted training runs. Very large due to replay buffer.
  2. Sidecar Metadata (`[name].json`):
     - Stores only the game configurations and signatures.
     - Rationale: The FastAPI backend and dashboards read these JSON sidecars to populate
       run catalog lists without incurring the heavy CPU/RAM overhead of parsing multi-megabyte
       PyTorch checkpoint files.
  3. Inference-Only Policy (`[name].policy.pth`):
     - Stores only the model weights and basic config metadata (no optimizer/replay buffer).
     - Rationale: Used for fast loading during game simulation/demos to minimize memory footprint.

Rule Signatures and Invariants:
  - GameConfig components are hashed to generate a unique `rule_signature`.
  - When loading a checkpoint, we compare the signature of the model against the environment configuration.
  - If they do not match, we raise a ValueError to prevent sizing issues in PyTorch linear layers.

Connections:
  - Imported by: `training.train_dqn.py` (saves model periodically and after evaluation)
  - Imported by: `play.play_best_dqn_game.py` and backend play/eval routes to run trained agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from config.config_manager import (
    GameConfig,
    TrainingConfig,
    compute_rule_signature,
    compute_training_signature,
    load_game_config,
)
from game_runtime.scrum_game_env import ScrumGameEnv
from rl.dqn_agent import DQNAgent, encode_state


def build_agent_for_config(
    game_config: GameConfig,
    learning_rate: float = 0.0005,
    gamma: float = 0.85,
    replay_capacity: int = 100000,
    batch_size: int = 128,
    target_update_frequency: int = 2000,
    device: str | None = None,
) -> tuple[DQNAgent, ScrumGameEnv]:
    """
    Construct a stateful DQNAgent and ScrumGameEnv whose network sizes match the game config.
    
    Args:
        game_config: Rule configurations defining products and observation shapes.
        
    Returns:
        tuple: (DQNAgent, ScrumGameEnv) properly configured.
    """
    env = ScrumGameEnv(game_config=game_config)
    # Determine input size dynamically by running environment reset and encoding the output dict
    state_dim = len(encode_state(env.reset(seed=42), env))
    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=env.num_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        target_update_frequency=target_update_frequency,
        device=device,
    )
    return agent, env


def save_checkpoint(
    checkpoint_path: str | Path,
    agent: DQNAgent,
    game_config: GameConfig,
    training_config: TrainingConfig | None = None,
    extra_metadata: dict[str, Any] | None = None,
):
    """
    Save the model, training state, replay buffers, and generate sidecar files.
    
    Creates:
      - `[path].pth` (Full train checkpoint)
      - `[path].json` (Fast sidecar metadata)
      - `[path].policy.pth` (Compact inference-only weights)
    """
    checkpoint_path = Path(checkpoint_path)
    metadata = {
        "format_version": 2,
        "rule_signature": compute_rule_signature(game_config),
        "training_signature": (
            compute_training_signature(training_config) if training_config is not None else None
        ),
        "game_config": game_config.to_dict(),
        "training_config": training_config.to_dict() if training_config is not None else None,
        "state_dim": agent.state_dim,
        "num_actions": agent.num_actions,
        "device": agent.device,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    training_state = agent.training_state_dict()
    
    # 1. Save Full Checkpoint (for resuming training)
    torch.save(
        {
            "model_state_dict": agent.policy_network.state_dict(),
            "target_model_state_dict": agent.target_network.state_dict(),
            **training_state,
            "metadata": metadata,
        },
        checkpoint_path,
    )

    # 2. Save JSON Sidecar (for UI/API query speed)
    sidecar_path = checkpoint_path.with_suffix(".json")
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    # 3. Save Policy-Only Checkpoint (for lightweight inference deployments)
    inference_path = checkpoint_path.with_suffix(".policy.pth")
    torch.save(
        {
            "model_state_dict": agent.policy_network.state_dict(),
            "target_model_state_dict": agent.target_network.state_dict(),
            "metadata": metadata,
        },
        inference_path,
    )


def backfill_checkpoint_sidecars(runs_dir: Path) -> int:
    """
    Scan directories for `.pth` files and generate matching `.json` sidecars if missing.
    Useful for back-populating catalogs after server migrations.
    """
    written = 0
    for pth_path in sorted(runs_dir.rglob("*.pth")):
        if pth_path.name.endswith(".policy.pth"):
            continue
        sidecar = pth_path.with_suffix(".json")
        if sidecar.exists():
            continue
        try:
            payload = torch.load(pth_path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            with sidecar.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
            written += 1
            print(f"  wrote {sidecar.name} for {pth_path.parent.parent.name}/{pth_path.name}")
        except Exception as exc:
            print(f"  skipped {pth_path.name}: {exc}")
    return written


if __name__ == "__main__":
    import sys
    engine_root = Path(__file__).resolve().parents[1]
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else engine_root / "artifacts" / "runs"
    print(f"Backfilling checkpoint sidecars under {target} ...")
    n = backfill_checkpoint_sidecars(target)
    print(f"Done — {n} sidecar(s) written.")


def load_checkpoint_payload(checkpoint_path: str | Path, map_location=None) -> dict:
    """
    Load raw checkpoint weights and build a standardized dictionary format,
    handling legacy file conversions gracefully.
    """
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload

    # Fallback structure for older versions that only saved weights
    return {
        "model_state_dict": payload,
        "metadata": {
            "format_version": 1,
            "rule_signature": None,
            "training_signature": None,
            "game_config": None,
            "training_config": None,
            "state_dim": None,
            "num_actions": None,
            "legacy_checkpoint": True,
        },
    }


def checkpoint_game_config(payload: dict, fallback_game_config: GameConfig | None = None) -> GameConfig:
    """Extract and validate the game config stored in the checkpoint's metadata."""
    metadata = payload.get("metadata", {})
    embedded_config = metadata.get("game_config")
    if embedded_config is not None:
        return GameConfig.from_dict(embedded_config)
    if fallback_game_config is not None:
        return fallback_game_config
    return load_game_config()


def validate_checkpoint_compatibility(
    payload: dict,
    game_config: GameConfig,
    strict_signature: bool = True,
) -> dict:
    """
    Ensure the checkpoint is physically compatible with the game's products and observation format.
    Raises RuntimeError if signatures are mismatched to prevent loading invalid network sizes.
    """
    metadata = payload.get("metadata", {})
    checkpoint_rule_signature = metadata.get("rule_signature")
    current_rule_signature = compute_rule_signature(game_config)

    if strict_signature and checkpoint_rule_signature and checkpoint_rule_signature != current_rule_signature:
        raise RuntimeError(
            "Checkpoint rule signature does not match the selected game config. "
            "Train or load a model for the same board, dice, incident, and refinement rules."
        )

    return {
        "checkpoint_rule_signature": checkpoint_rule_signature,
        "current_rule_signature": current_rule_signature,
        "legacy_checkpoint": bool(metadata.get("legacy_checkpoint", False)),
    }


def load_agent_from_checkpoint(
    checkpoint_path: str | Path,
    game_config: GameConfig | None = None,
    strict_signature: bool = True,
) -> tuple[DQNAgent, ScrumGameEnv, dict]:
    """
    Load a checkpoint file and instantiate a fully state-restored DQNAgent and ScrumGameEnv.
    """
    checkpoint_path = Path(checkpoint_path)
    payload = load_checkpoint_payload(checkpoint_path)
    resolved_game_config = checkpoint_game_config(payload, fallback_game_config=game_config)
    compatibility = validate_checkpoint_compatibility(
        payload,
        resolved_game_config,
        strict_signature=strict_signature,
    )

    training_config_payload = payload.get("metadata", {}).get("training_config")
    training_config = (
        TrainingConfig.from_dict(training_config_payload)
        if training_config_payload is not None
        else None
    )

    # Reconstruct the networks
    agent, env = build_agent_for_config(
        resolved_game_config,
        learning_rate=(
            training_config.learning_rate if training_config is not None else 0.0005
        ),
        gamma=(training_config.gamma if training_config is not None else 0.85),
    )

    # Load weights
    state_dict = payload["model_state_dict"]
    agent.policy_network.load_state_dict(state_dict)
    agent.target_network.load_state_dict(payload.get("target_model_state_dict", state_dict))
    agent.policy_network.eval()
    agent.target_network.eval()

    # Load training counters (optimizer settings and replay buffer)
    agent.load_training_state_dict(payload, include_replay=True)

    metadata = dict(payload.get("metadata", {}))
    metadata.update(compatibility)
    metadata["resolved_game_config"] = resolved_game_config
    metadata["resolved_training_config"] = training_config
    metadata["checkpoint_path"] = str(checkpoint_path)
    return agent, env, metadata


def load_agent_for_inference(
    checkpoint_path: str | Path,
    game_config: GameConfig | None = None,
    strict_signature: bool = True,
) -> tuple[DQNAgent, ScrumGameEnv, dict]:
    """
    Lightweight checkpoint load function used exclusively for run simulation and evaluation.
    
    Sets `replay_capacity=1` and bypasses loading optimizer state and massive replay histories,
    speeding up loading and minimizing RAM usage.
    """
    checkpoint_path = Path(checkpoint_path)
    inference_path = checkpoint_path.with_suffix(".policy.pth")
    # Prefer loading policy-only weights if available
    payload_path = inference_path if inference_path.exists() else checkpoint_path
    payload = load_checkpoint_payload(payload_path, map_location="cpu")
    
    # Auto-generate policy-only checkpoint if it does not exist
    if payload_path == checkpoint_path:
        try:
            torch.save(
                {
                    "model_state_dict": payload["model_state_dict"],
                    "target_model_state_dict": payload.get(
                        "target_model_state_dict",
                        payload["model_state_dict"],
                    ),
                    "metadata": payload.get("metadata", {}),
                },
                inference_path,
            )
            payload_path = inference_path
        except Exception:
            pass
            
    resolved_game_config = checkpoint_game_config(payload, fallback_game_config=game_config)
    compatibility = validate_checkpoint_compatibility(
        payload,
        resolved_game_config,
        strict_signature=strict_signature,
    )

    training_config_payload = payload.get("metadata", {}).get("training_config")
    training_config = (
        TrainingConfig.from_dict(training_config_payload)
        if training_config_payload is not None
        else None
    )

    # Initialize agent with minimal replay capacity to conserve RAM
    agent, env = build_agent_for_config(
        resolved_game_config,
        learning_rate=(
            training_config.learning_rate if training_config is not None else 0.0005
        ),
        gamma=(training_config.gamma if training_config is not None else 0.85),
        replay_capacity=1,
    )

    state_dict = payload["model_state_dict"]
    agent.policy_network.load_state_dict(state_dict)
    agent.target_network.load_state_dict(payload.get("target_model_state_dict", state_dict))
    agent.policy_network.eval()
    agent.target_network.eval()

    metadata = dict(payload.get("metadata", {}))
    metadata.update(compatibility)
    metadata["resolved_game_config"] = resolved_game_config
    metadata["resolved_training_config"] = training_config
    metadata["checkpoint_path"] = str(checkpoint_path)
    metadata["inference_checkpoint_path"] = str(payload_path)
    return agent, env, metadata

