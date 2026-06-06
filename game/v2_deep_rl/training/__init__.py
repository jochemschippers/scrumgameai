"""
Training Package.

This package houses the training execution script and core evaluation methods.

Exposes:
  - train_dqn_agent: Standard Double DQN model training routine.
  - evaluate_dqn_agent: Greedy evaluation checker using epsilon=0.
"""

from __future__ import annotations

from .train_dqn import train_dqn_agent, evaluate_dqn_agent

__all__ = [
    "train_dqn_agent",
    "evaluate_dqn_agent",
]
