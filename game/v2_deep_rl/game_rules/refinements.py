"""
Refinement Rules Engine for the Scrum Game.

This module implements the "refinement model" (e.g., standard ID 301 rules),
which models the unpredictability of product backlog grooming. During a turn,
if the active product is worked on, the team rolls a die (default 20-sided)
to see if their refinement efforts succeeded (adding features to future sprints)
or backfired (reducing features in the final sprint of that product).

Rules (Standard Model ID 301):
  - Increase Roll (e.g., d20 result 1-2 or 1-3 depending on product key):
    Adds 1 feature to ALL future (uncompleted) sprints of that product.
  - Decrease Roll (e.g., d20 result 19-20):
    Subtracts 1 feature from the LAST future (uncompleted) sprint of that product.
  - Neutral Roll:
    No effect.

Connections:
  - Configured via: `config_manager.GameConfig.refinement`
  - Instantiated by: `game_runtime.scrum_game_env.ScrumGameEnv`
  - Invoked during: `env.step()` after action resolution.
"""

from __future__ import annotations

import random

from config.config_manager import RefinementConfig, normalize_product_key


class ConfiguredRefinementModel:
    """
    Stateful refinement engine applying rolls and modifications to product features.
    """

    def __init__(self, refinement_config: RefinementConfig):
        """
        Initialize the refinement engine.
        
        Args:
            refinement_config: The refinement settings containing product roll lists.
        """
        self.refinement_config = refinement_config
        # Quick-lookup structure mapping product keys to their respective roll sets
        self.rule_lookup = {
            rule.product_key: {
                "increase_rolls": set(rule.increase_rolls),
                "decrease_rolls": set(rule.decrease_rolls),
            }
            for rule in refinement_config.product_rules
        }

    def apply(self, env, product_id: int) -> dict[str, Any]:
        """
        Roll a die and apply the refinement outcome to the board's future sprints.
        
        Args:
            env (ScrumGameEnv): The active environment instance.
            product_id (int): 1-based index of the product being worked on.
            
        Returns:
            dict: Log of the roll result containing:
                  - "roll": the rolled number
                  - "effect": "increase", "decrease", or "none"
                  - "target_product": the 1-based product ID
                  - "future_sprints_changed": list of modified sprint records
        """
        product_name = env.product_names[product_id - 1]
        product_key = normalize_product_key(product_name)
        
        # Retrieve refinement rolls. Default to empty sets if not configured.
        rules = self.rule_lookup.get(
            product_key,
            {"increase_rolls": set(), "decrease_rolls": set()},
        )
        
        # Roll the dice! (Usually d20)
        roll = random.randint(1, self.refinement_config.die_sides)

        result = {
            "roll": roll,
            "effect": "none",
            "target_product": product_id,
            "future_sprints_changed": [],
        }

        # Gather list of 1-based sprint IDs that are still in the future for this product
        future_sprints = [
            sprint_id
            for sprint_id in range(1, env.sprints_per_product + 1)
            if env.is_sprint_future(product_id, sprint_id, future_only=True)
        ]

        # If there are no future sprints left (product is fully completed), refinement has no target
        if not future_sprints:
            return result

        # Case 1: Refinement is successful (Increase roll). Add +1 feature to ALL future sprints.
        if roll in rules["increase_rolls"]:
            for sprint_id in future_sprints:
                env.add_refinement_delta(product_id, sprint_id, 1)
                result["future_sprints_changed"].append({"sprint_id": sprint_id, "delta": 1})
            result["effect"] = "increase"
            return result

        # Case 2: Refinement is unsuccessful (Decrease roll). Subtract -1 feature from the LAST future sprint.
        if roll in rules["decrease_rolls"]:
            last_future_sprint = future_sprints[-1]
            env.add_refinement_delta(product_id, last_future_sprint, -1)
            result["future_sprints_changed"].append({"sprint_id": last_future_sprint, "delta": -1})
            result["effect"] = "decrease"
            return result

        # Case 3: Neutral roll (no changes)
        return result

