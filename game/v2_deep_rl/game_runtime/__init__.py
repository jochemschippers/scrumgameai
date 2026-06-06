"""
Game Runtime Package.

This package houses the stateful Scrum Game simulation environment.

Exposes:
  - ScrumGameEnv: Core simulation environment implementing Gym-like interface.
  - discretize_state: Helper function to convert raw states to discrete hashable bins.
"""

from __future__ import annotations

from .scrum_game_env import ScrumGameEnv, discretize_state

__all__ = [
    "ScrumGameEnv",
    "discretize_state",
]
