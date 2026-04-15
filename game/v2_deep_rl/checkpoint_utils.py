from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from config_manager import (
    GameConfig,
    TrainingConfig,
    compute_rule_signature,
    compute_training_signature,
    load_game_config,
)
from dqn_agent import DQNAgent, encode_state
from scrum_game_env import ScrumGameEnv


def build_agent_for_config(
    game_config: GameConfig,
    learning_rate: float = 0.0005,
    gamma: float = 0.85,
    replay_capacity: int = 100000,
    batch_size: int = 128,
    target_update_frequency: int = 2000,
    device=None,
):
    """Construct an agent whose network shape matches one game config."""
    env = ScrumGameEnv(game_config=game_config)
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
    checkpoint_path,
    agent,
    game_config: GameConfig,
    training_config: TrainingConfig | None = None,
    extra_metadata: dict[str, Any] | None = None,
):
    """Save a checkpoint bundle with model weights and compatibility metadata."""
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
    torch.save(
        {
            "model_state_dict": agent.policy_network.state_dict(),
            "target_model_state_dict": agent.target_network.state_dict(),
            **training_state,
            "metadata": metadata,
        },
        checkpoint_path,
    )


def load_checkpoint_payload(checkpoint_path, map_location=None):
    """Load a checkpoint and normalize legacy raw state-dict files."""
    payload = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload

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


def checkpoint_game_config(payload, fallback_game_config: GameConfig | None = None) -> GameConfig:
    """Resolve the game config embedded in a checkpoint or fall back to a provided/default one."""
    metadata = payload.get("metadata", {})
    embedded_config = metadata.get("game_config")
    if embedded_config is not None:
        return GameConfig.from_dict(embedded_config)
    if fallback_game_config is not None:
        return fallback_game_config
    return load_game_config()


def validate_checkpoint_compatibility(
    payload,
    game_config: GameConfig,
    strict_signature: bool = True,
):
    """Validate that one checkpoint belongs to the requested ruleset."""
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
    checkpoint_path,
    game_config: GameConfig | None = None,
    strict_signature: bool = True,
):
    """Load a checkpoint into a compatible agent and return agent, env, and metadata."""
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

    agent, env = build_agent_for_config(
        resolved_game_config,
        learning_rate=(
            training_config.learning_rate if training_config is not None else 0.0005
        ),
        gamma=(training_config.gamma if training_config is not None else 0.85),
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
    return agent, env, metadata
