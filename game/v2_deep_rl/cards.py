from dataclasses import dataclass
import random


@dataclass(frozen=True)
class IncidentCard:
    """One incident card and its rule-faithful effect metadata."""

    card_id: int
    name: str
    description: str
    effect_type: str
    target_products: tuple[int, ...] = ()
    delta_money: int = 0
    target_sprint: int | None = None
    set_value_money: int | None = None
    future_only: bool = True

    def apply_effect(self, env):
        """Apply the incident effect to the environment state."""
        if self.effect_type == "set_future_product_to_zero":
            for product_id in self.target_products:
                for sprint_id in range(1, env.sprints_per_product + 1):
                    if env.is_sprint_future(product_id, sprint_id, future_only=self.future_only):
                        env.set_incident_value(product_id, sprint_id, 0)
            return

        if self.effect_type == "adjust_future_products":
            for product_id in self.target_products:
                for sprint_id in range(1, env.sprints_per_product + 1):
                    if env.is_sprint_future(product_id, sprint_id, future_only=self.future_only):
                        env.add_incident_delta(product_id, sprint_id, self.delta_money)
            return

        if self.effect_type == "adjust_specific_sprint_globally":
            for product_id in range(1, env.products_count + 1):
                if env.is_sprint_future(product_id, self.target_sprint, future_only=self.future_only):
                    env.add_incident_delta(product_id, self.target_sprint, self.delta_money)
            return

        if self.effect_type == "set_specific_sprint_exact":
            for product_id in self.target_products:
                if env.is_sprint_future(product_id, self.target_sprint, future_only=self.future_only):
                    env.set_incident_value(product_id, self.target_sprint, self.set_value_money)
            return

        raise ValueError(f"Unsupported incident effect type: {self.effect_type}")


class IncidentDeck:
    """Deck with draw and discard mechanics for the Scrum Game incidents."""

    def __init__(self, cards):
        self.all_cards = list(cards)
        self.draw_pile = []
        self.discard_pile = []
        self.shuffle()

    def shuffle(self):
        """Reset the draw pile from the full card list."""
        self.draw_pile = list(self.all_cards)
        random.shuffle(self.draw_pile)
        self.discard_pile = []

    def reshuffle_discard_pile(self):
        """Move discard pile back into the draw pile when needed."""
        if not self.discard_pile:
            return
        self.draw_pile = list(self.discard_pile)
        random.shuffle(self.draw_pile)
        self.discard_pile = []

    def draw(self):
        """Draw the next incident card, reshuffling the discard pile if needed."""
        if not self.draw_pile:
            self.reshuffle_discard_pile()
        if not self.draw_pile:
            raise RuntimeError("Incident deck is empty and cannot be reshuffled.")
        card = self.draw_pile.pop()
        self.discard_pile.append(card)
        return card


def build_classical_incident_cards(product_names):
    """
    Build the incident deck from the rule material.

    The manuals explicitly describe the following physical incident cards:
    1. Red demand collapse to zero
    2. Orange and blue competitor pressure
    3. Government subsidy for all first sprints
    4. Yellow demand boost
    5. Black product breakthrough on sprint 4

    The simulator variables mention 8 incident cards, but only these concrete
    effects were documented in the provided manuals. The deck is therefore
    source-faithful to the cards that are actually available in the gamedata.
    """
    product_lookup = {name.lower(): index + 1 for index, name in enumerate(product_names)}

    return [
        IncidentCard(
            card_id=401,
            name="Demand Collapse Red",
            description="The demand of the red product drops dramatically. All future red sprints are worth zero.",
            effect_type="set_future_product_to_zero",
            target_products=(product_lookup["red"],),
        ),
        IncidentCard(
            card_id=402,
            name="New Competitors",
            description="Orange and blue products lose value due to new competitors entering the market.",
            effect_type="adjust_future_products",
            target_products=(product_lookup["orange"], product_lookup["blue"]),
            delta_money=-5000,
        ),
        IncidentCard(
            card_id=403,
            name="Government Subsidy",
            description="All first sprints gain a subsidy bonus.",
            effect_type="adjust_specific_sprint_globally",
            target_sprint=1,
            delta_money=5000,
        ),
        IncidentCard(
            card_id=404,
            name="Yellow Demand Boost",
            description="All future yellow sprints gain value.",
            effect_type="adjust_future_products",
            target_products=(product_lookup["yellow"],),
            delta_money=5000,
        ),
        IncidentCard(
            card_id=405,
            name="Black Product Breakthrough",
            description="The fourth black sprint becomes worth 100,000.",
            effect_type="set_specific_sprint_exact",
            target_products=(product_lookup["black"],),
            target_sprint=4,
            set_value_money=100000,
        ),
    ]
