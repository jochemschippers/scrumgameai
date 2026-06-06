"""
Incident Cards and Deck Management for the Scrum Game.

This module governs the deck of "incident cards" that introduce random economic
events at the end of each round (e.g., market crashes, subsidies, competitor actions).
These events temporarily or permanently modify the point values (money) of board cells.

Core Concepts:
  - IncidentCard: A specific event (e.g., "Demand Collapse Red") with a defined effect.
  - IncidentDeck: The stateful manager of the card deck, implementing draw, discard,
    and weight-based duplicate rules.
  - Severity Multiplier: A float from the environment that scales incident card amounts.

Card Effect Types:
  - set_future_product_to_zero: Wipes out the value of all remaining future sprints for specific products.
  - adjust_future_products: Modifies remaining future sprint values by a fixed delta.
  - adjust_specific_sprint_globally: Adjusts a specific sprint index (e.g., Sprint 1) across all products.
  - set_specific_sprint_exact: Sets a specific sprint of a specific product to an exact cash value.

Connections:
  - Configured via: `config_manager.GameConfig.incident`
  - Instantiated by: `game_runtime.scrum_game_env` (which draws and applies cards to board values)
  - Synchronized in multiplayer via: `play.shared_match_runner.SharedMatchState`
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from config.config_manager import GameConfig, normalize_product_key


@dataclass(frozen=True)
class IncidentCard:
    """
    Represents an individual incident card and defines its effect on the environment board.
    
    Attributes:
        card_id: Unique identifier for the card (e.g., 401).
        name: Short display name (e.g., "Demand Collapse Red").
        description: Informational text describing the event.
        effect_type: The formula code used to apply the card (e.g. 'adjust_future_products').
        target_product_keys: The product keys this incident affects (e.g. ('red',)).
        delta_money: Currency adjustment amount (positive or negative).
        target_sprint: The specific sprint index (1-based) to modify, if applicable.
        set_value_money: The target absolute money value for exact-value cards.
        future_only: If True, the effect only applies to sprints that haven't been completed.
        weight: Determines how many copies of this card are added to the deck.
    """

    card_id: int
    name: str
    description: str
    effect_type: str
    target_product_keys: tuple[str, ...] = ()
    delta_money: int = 0
    target_sprint: int | None = None
    set_value_money: int | None = None
    future_only: bool = True
    weight: float = 1.0

    def _scaled_delta(self, env, value: int) -> int:
        """Scale a monetary delta by the environment's incident severity multiplier."""
        return int(round(value * env.incident_severity_multiplier))

    def _scaled_exact_value(self, env, value: int) -> int:
        """Scale an exact target value by the environment's incident severity multiplier."""
        return int(round(value * env.incident_severity_multiplier))

    def apply_effect(self, env):
        """
        Apply this card's economic adjustment directly to the environment's board values.
        
        Args:
            env (ScrumGameEnv): The active game environment containing board arrays
                                and helper methods like `is_sprint_future` and `add_incident_delta`.
        """
        # Map target product string keys (e.g. 'red') to their 1-based index in the environment
        target_product_ids = [
            env.product_lookup[normalize_product_key(product_key)]
            for product_key in self.target_product_keys
            if normalize_product_key(product_key) in env.product_lookup
        ]

        # Case 1: Zero out all remaining future sprints of target products.
        if self.effect_type == "set_future_product_to_zero":
            for product_id in target_product_ids:
                for sprint_id in range(1, env.sprints_per_product + 1):
                    if env.is_sprint_future(product_id, sprint_id, future_only=self.future_only):
                        env.set_incident_value(product_id, sprint_id, 0)
            return

        # Case 2: Apply a delta (addition/subtraction) to all future sprints of target products.
        if self.effect_type == "adjust_future_products":
            scaled_delta = self._scaled_delta(env, self.delta_money)
            for product_id in target_product_ids:
                for sprint_id in range(1, env.sprints_per_product + 1):
                    if env.is_sprint_future(product_id, sprint_id, future_only=self.future_only):
                        env.add_incident_delta(product_id, sprint_id, scaled_delta)
            return

        # Case 3: Apply a delta to a specific sprint (e.g. Sprint 1) across ALL products globally.
        if self.effect_type == "adjust_specific_sprint_globally":
            scaled_delta = self._scaled_delta(env, self.delta_money)
            for product_id in range(1, env.products_count + 1):
                if env.is_sprint_future(product_id, self.target_sprint, future_only=self.future_only):
                    env.add_incident_delta(product_id, self.target_sprint, scaled_delta)
            return

        # Case 4: Force a specific future sprint of target products to an exact value.
        if self.effect_type == "set_specific_sprint_exact":
            if self.set_value_money is None:
                raise ValueError("set_specific_sprint_exact requires set_value_money.")
            scaled_value = self._scaled_exact_value(env, self.set_value_money)
            for product_id in target_product_ids:
                if env.is_sprint_future(product_id, self.target_sprint, future_only=self.future_only):
                    env.set_incident_value(product_id, self.target_sprint, scaled_value)
            return

        raise ValueError(f"Unsupported incident effect type: {self.effect_type}")


class IncidentDeck:
    """
    Deck controller managing drawing, shuffles, discards, and card duplication weights.
    """

    def __init__(self, cards: list[IncidentCard]):
        """
        Initialize the deck.
        
        Args:
            cards: The list of raw IncidentCard templates defined by the config.
        """
        self.all_cards = list(cards)
        self.draw_pile: list[IncidentCard] = []
        self.discard_pile: list[IncidentCard] = []
        self.shuffle()

    def _expanded_card_pool(self) -> list[IncidentCard]:
        """
        Duplicate cards according to their weight attribute.
        If card weight is 2.0, two copies are placed in the expanded pool.
        """
        expanded = []
        for card in self.all_cards:
            copies = max(1, int(round(card.weight)))
            expanded.extend([card] * copies)
        return expanded

    def shuffle(self):
        """Reset the draw pile by cloning and expanding the card templates, then shuffle."""
        self.draw_pile = self._expanded_card_pool()
        random.shuffle(self.draw_pile)
        self.discard_pile = []

    def reshuffle_discard_pile(self):
        """Move discard pile cards back into the draw pile and shuffle them."""
        if not self.discard_pile:
            return
        self.draw_pile = list(self.discard_pile)
        random.shuffle(self.draw_pile)
        self.discard_pile = []

    def draw(self) -> IncidentCard:
        """
        Pop and return the top card from the draw pile.
        Automatically reshuffles the discard pile if the draw pile runs dry.
        """
        if not self.draw_pile:
            self.reshuffle_discard_pile()
        if not self.draw_pile:
            raise RuntimeError("Incident deck is empty and cannot be reshuffled.")
        card = self.draw_pile.pop()
        self.discard_pile.append(card)
        return card


def build_incident_cards(game_config: GameConfig) -> list[IncidentCard]:
    """
    Factory function to convert GameConfig IncidentCardConfig models into stateful IncidentCard objects.
    """
    return [
        IncidentCard(
            card_id=card_config.card_id,
            name=card_config.name,
            description=card_config.description,
            effect_type=card_config.effect_type,
            target_product_keys=tuple(card_config.target_products),
            delta_money=card_config.delta_money,
            target_sprint=card_config.target_sprint,
            set_value_money=card_config.set_value_money,
            future_only=card_config.future_only,
            weight=card_config.weight,
        )
        for card_config in game_config.incident.cards
    ]

