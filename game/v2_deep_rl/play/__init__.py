"""
Play Package.

This package orchestrates match executions, including parallel match sessions
(independent seats) and shared-board multiplayer matches.

Exposes:
  - Controller, HumanController, RandomController, HeuristicController, ModelController: Seat players.
  - start_parallel_match, play_round: Parallel seats match loop.
  - start_shared_match, play_shared_round: Shared board match loop.
  - choose_profile_action: Profile-based action selection.
"""

from __future__ import annotations

from .match_runner import (
    Controller,
    HumanController,
    RandomController,
    HeuristicController,
    ModelController,
    start_parallel_match,
    play_round,
)
from .shared_match_runner import start_shared_match, play_shared_round
from .deployment_profiles import choose_profile_action

__all__ = [
    "Controller",
    "HumanController",
    "RandomController",
    "HeuristicController",
    "ModelController",
    "start_parallel_match",
    "play_round",
    "start_shared_match",
    "play_shared_round",
    "choose_profile_action",
]
