"""
Config Package.

This package handles game rule structures, training hyperparameters, config
loading/saving, and prototype mapping.

Exposes:
  - GameConfig: Structured data representing scrum game rules.
  - TrainingConfig: Deep RL training settings.
  - load_game_config, load_training_config: Config loading helpers.
  - save_game_config, save_training_config: Config saving helpers.
  - compute_rule_signature, compute_training_signature: Stability hashing functions.
"""

from __future__ import annotations

from .config_manager import (
    GameConfig,
    TrainingConfig,
    DiceRuleConfig,
    RefinementConfig,
    RefinementRuleConfig,
    IncidentConfig,
    IncidentCardConfig,
    load_game_config,
    load_training_config,
    save_game_config,
    save_training_config,
    compute_rule_signature,
    compute_training_signature,
    normalize_product_key,
)

__all__ = [
    "GameConfig",
    "TrainingConfig",
    "DiceRuleConfig",
    "RefinementConfig",
    "RefinementRuleConfig",
    "IncidentConfig",
    "IncidentCardConfig",
    "load_game_config",
    "load_training_config",
    "save_game_config",
    "save_training_config",
    "compute_rule_signature",
    "compute_training_signature",
    "normalize_product_key",
]
