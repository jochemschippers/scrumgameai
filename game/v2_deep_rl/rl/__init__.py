"""
RL Package.

This package implements the Double DQN agent, neural network model,
and checkpoint loading/saving utilities.

Exposes:
  - DQNAgent: Double DQN agent implementation.
  - encode_state: Observation dictionary normalizer.
  - load_agent_from_checkpoint, load_agent_for_inference: Checkpoint loading helpers.
  - save_checkpoint: Checkpoint saving helper.
"""

from __future__ import annotations

from .dqn_agent import DQNAgent, encode_state
from .checkpoint_utils import (
    load_agent_from_checkpoint,
    load_agent_for_inference,
    save_checkpoint,
)

__all__ = [
    "DQNAgent",
    "encode_state",
    "load_agent_from_checkpoint",
    "load_agent_for_inference",
    "save_checkpoint",
]
