"""
Game Rules Package.

This package defines the deck of incident cards, refinement logic, and
domain-randomization helpers.

Exposes:
  - IncidentCard, IncidentDeck: Incident engine elements.
  - build_incident_cards: Factory for creating incident cards.
  - ConfiguredRefinementModel: Grooming rules roll applicator.
  - sample_game_config: Domain-randomization sampler.
"""

from __future__ import annotations

from .cards import IncidentCard, IncidentDeck, build_incident_cards
from .refinements import ConfiguredRefinementModel
from .rule_randomization import sample_game_config

__all__ = [
    "IncidentCard",
    "IncidentDeck",
    "build_incident_cards",
    "ConfiguredRefinementModel",
    "sample_game_config",
]
