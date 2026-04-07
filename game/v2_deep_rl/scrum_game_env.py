import random

from cards import IncidentDeck, build_classical_incident_cards
from refinements import StandardRefinementModel301


class ScrumGameEnv:
    """Advanced Gym-style environment for the deep-RL Scrum Game branch."""

    def __init__(
        self,
        classical_setup=True,
        incidents_active=True,
        refinements_active=True,
        players_count=1,
        allow_player_specific_incidents=False,
    ):
        self.classical_setup = classical_setup
        self.incidents_active = incidents_active
        self.refinements_active = refinements_active
        self.players_count = players_count
        self.allow_player_specific_incidents = allow_player_specific_incidents

        # Classical defaults from the prototype and manuals.
        self.products_count = 7
        self.sprints_per_product = 4
        self.max_turns = 6
        self.num_actions = 1 + self.products_count

        self.product_names = [
            "Yellow",
            "Blue",
            "Red",
            "Orange",
            "Green",
            "Purple",
            "Black",
        ]

        self.starting_money = 25000
        self.ring_value = 5000
        self.cost_continue = 0
        self.cost_switch_mid = 5000
        self.cost_switch_after = 0
        self.mandatory_loan_amount = 50000
        self.loan_interest = 5000
        self.penalty_negative = 1000
        self.penalty_positive = 5000

        self.daily_scrums_per_sprint = 5
        self.daily_scrum_target = 12

        self.board_ring_values = [
            [4, 2, 1, 1],
            [5, 3, 2, 1],
            [6, 4, 3, 2],
            [5, 3, 2, 1],
            [4, 3, 2, 1],
            [5, 3, 2, 1],
            [7, 4, 3, 2],
        ]
        self.board_features = [
            [3, 3, 2, 1],
            [2, 2, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 1],
            [3, 2, 2, 1],
            [2, 2, 1, 1],
            [1, 1, 1, 1],
        ]

        self.max_base_sprint_value = max(max(row) for row in self.board_ring_values) * self.ring_value
        self.max_visible_sprint_value = 100000
        self.max_interest_reference = self.loan_interest * 6
        self.max_refinement_reference = 6

        self.refinement_engine = StandardRefinementModel301(self.product_names)
        self.incident_cards = build_classical_incident_cards(self.product_names)
        self.incident_deck = IncidentDeck(self.incident_cards)

        self.win_probability_lookup = self._build_win_probability_lookup()

        # Episode bookkeeping.
        self.current_money = self.starting_money
        self.current_product = 1
        self.turn_count = 0
        self.loans_taken = 0
        self.turns_with_loan = 0
        self.just_took_mandatory_loan = False
        self.loan_active = False
        self.interest_due = 0

        # Product-level progress and modifiers.
        self.product_next_sprints = [1] * self.products_count
        self.refinement_feature_deltas = [
            [0 for _ in range(self.sprints_per_product)] for _ in range(self.products_count)
        ]
        self.incident_value_deltas = [
            [0 for _ in range(self.sprints_per_product)] for _ in range(self.products_count)
        ]
        self.incident_value_overrides = [
            [None for _ in range(self.sprints_per_product)] for _ in range(self.products_count)
        ]

        # Current visible observation fields.
        self.current_sprint = 1
        self.features_required = 0
        self.sprint_value = 0
        self.win_probability = 0.0
        self.expected_value = 0.0
        self.remaining_turns = self.max_turns
        self.is_last_sprint = False
        self.debt_ratio = 0.0
        self.switch_is_free = False
        self.current_product_completed = False
        self.current_refinement_delta = 0
        self.current_incident_delta = 0

        # Current visible incident state.
        self.incident_active = 0
        self.current_incident_id = 0
        self.current_incident_name = "None"
        self.current_incident_scope = 0.0

        self._refresh_observation_fields()

    def reset(self, seed=None):
        """Reset the environment to the classical starting state."""
        if seed is not None:
            random.seed(seed)

        self.current_money = self.starting_money
        self.current_product = 1
        self.turn_count = 0
        self.loans_taken = 0
        self.turns_with_loan = 0
        self.just_took_mandatory_loan = False
        self.loan_active = False
        self.interest_due = 0

        self.product_next_sprints = [1] * self.products_count
        self.refinement_feature_deltas = [
            [0 for _ in range(self.sprints_per_product)] for _ in range(self.products_count)
        ]
        self.incident_value_deltas = [
            [0 for _ in range(self.sprints_per_product)] for _ in range(self.products_count)
        ]
        self.incident_value_overrides = [
            [None for _ in range(self.sprints_per_product)] for _ in range(self.products_count)
        ]

        self.incident_deck = IncidentDeck(build_classical_incident_cards(self.product_names))
        self.incident_active = 0
        self.current_incident_id = 0
        self.current_incident_name = "None"
        self.current_incident_scope = 0.0

        self._refresh_observation_fields()
        return self._get_state()

    def step(self, action):
        """
        Execute one environment turn.

        Actions:
        - 0: Continue with the currently selected product and play that sprint
        - 1..7: Switch to Product N, pay the relevant cost, and play that sprint
        """
        if action not in range(self.num_actions):
            raise ValueError(f"Action must be in [0, {self.num_actions - 1}].")

        self.turn_count += 1
        old_money = self.current_money
        self.just_took_mandatory_loan = False

        info = {
            "action": action,
            "action_name": self.action_name(action),
            "loan_triggered": False,
            "invalid_action": False,
            "interest_paid": 0,
            "switch_cost_paid": 0,
            "incident_triggered": False,
            "incident_card_id": 0,
            "incident_card_name": "None",
            "refinement_roll": None,
            "refinement_effect": "none",
            "selected_product": self.current_product if action == 0 else action,
        }

        if self.loan_active and self.interest_due > 0:
            interest_paid = self._apply_required_payment(self.interest_due, info)
            self.current_money -= interest_paid
            info["interest_paid"] = interest_paid

        played_product = self.current_product
        valid_turn = True

        if action == 0:
            if self.current_product_completed:
                valid_turn = False
                info["invalid_action"] = True
                info["invalid_action_reason"] = "continue_on_completed_product"
                result = "Invalid"
            else:
                result = self._resolve_sprint_for_product(self.current_product, info)
                played_product = self.current_product
        else:
            target_product = action
            info["selected_product"] = target_product

            if target_product == self.current_product:
                valid_turn = False
                info["invalid_action"] = True
                info["invalid_action_reason"] = "self_switch"
                result = "Invalid"
            elif self._is_product_complete(target_product):
                valid_turn = False
                info["invalid_action"] = True
                info["invalid_action_reason"] = "switch_to_completed_product"
                result = "Invalid"
            else:
                switch_cost = self.cost_switch_after if self.current_product_completed else self.cost_switch_mid
                switch_cost = self._apply_required_payment(switch_cost, info)
                self.current_money -= switch_cost
                info["switch_cost_paid"] = switch_cost

                self.current_product = target_product
                played_product = target_product
                result = self._resolve_sprint_for_product(target_product, info)

        if valid_turn and self.refinements_active:
            refinement_result = self.refinement_engine.apply(self, played_product)
            info["refinement_roll"] = refinement_result["roll"]
            info["refinement_effect"] = refinement_result["effect"]
            info["refinement_changes"] = refinement_result["future_sprints_changed"]
        else:
            info["refinement_changes"] = []

        if self.incidents_active:
            incident_card = self.incident_deck.draw()
            incident_card.apply_effect(self)
            self.incident_active = 1
            self.current_incident_id = incident_card.card_id
            self.current_incident_name = incident_card.name
            self.current_incident_scope = self._encode_incident_scope(incident_card)
            info["incident_triggered"] = True
            info["incident_card_id"] = incident_card.card_id
            info["incident_card_name"] = incident_card.name
        else:
            self.incident_active = 0
            self.current_incident_id = 0
            self.current_incident_name = "None"
            self.current_incident_scope = 0.0

        if self.loan_active:
            self.turns_with_loan += 1
        else:
            self.turns_with_loan = 0

        self._refresh_observation_fields()

        reward = self.calculate_reward(
            old_money=old_money,
            new_money=self.current_money,
            action=action,
            result=result,
            info=info,
        )

        done = False
        terminal_reason = None
        if self.current_money < 0:
            reward -= 100000
            done = True
            terminal_reason = "bankruptcy"
        elif self.turn_count >= self.max_turns:
            done = True
            terminal_reason = "max_turns_reached"
        elif all(self._is_product_complete(product_id) for product_id in range(1, self.products_count + 1)):
            done = True
            terminal_reason = "all_products_completed"

        info["result"] = result
        info["turn_count"] = self.turn_count
        info["ending_money"] = self.current_money
        if terminal_reason is not None:
            info["terminal_reason"] = terminal_reason

        return self._get_state(), reward, done, info

    def calculate_reward(self, old_money, new_money, action, result, info):
        """Calculate the shaped reward used for deep-RL training."""
        reward = new_money - old_money

        if self.loan_active:
            reward -= 500 * self.turns_with_loan

        if result == "Success" and not self.loan_active:
            reward += 2000

        if action > 0 and old_money < 10000 and result in {"Success", "Failure"}:
            reward += 1000

        if self.just_took_mandatory_loan:
            reward -= 20000

        if info.get("invalid_action"):
            reward -= 2000

        if info.get("refinement_effect") == "decrease":
            reward += 500
        elif info.get("refinement_effect") == "increase":
            reward -= 250

        return reward

    def action_name(self, action):
        """Convert an action id to a readable label."""
        if action == 0:
            return "Continue"
        return f"Switch to {self.product_names[action - 1]}"

    def build_reference_state(self, product_id, sprint_id, current_money=25000, loan_active=False, interest_due=0):
        """Build a synthetic reference observation for dashboard visualizations."""
        target_product_id = max(1, min(self.products_count, int(product_id)))
        target_sprint_id = max(1, min(self.sprints_per_product, int(sprint_id)))

        self.product_next_sprints = [1] * self.products_count
        self.product_next_sprints[target_product_id - 1] = target_sprint_id
        self.refinement_feature_deltas = [
            [0 for _ in range(self.sprints_per_product)] for _ in range(self.products_count)
        ]
        self.incident_value_deltas = [
            [0 for _ in range(self.sprints_per_product)] for _ in range(self.products_count)
        ]
        self.incident_value_overrides = [
            [None for _ in range(self.sprints_per_product)] for _ in range(self.products_count)
        ]

        self.current_money = current_money
        self.current_product = target_product_id
        self.turn_count = 0
        self.loans_taken = 1 if loan_active else 0
        self.loan_active = loan_active
        self.interest_due = interest_due
        self.turns_with_loan = 1 if loan_active else 0
        self.just_took_mandatory_loan = False
        self.incident_active = 0
        self.current_incident_id = 0
        self.current_incident_name = "None"
        self.current_incident_scope = 0.0

        self._refresh_observation_fields()
        return self._get_state()

    def is_sprint_future(self, product_id, sprint_id, future_only=True):
        """Return whether a sprint is still available for incident/refinement effects."""
        next_sprint = self.product_next_sprints[product_id - 1]
        if next_sprint > self.sprints_per_product:
            return False
        if future_only:
            return sprint_id >= next_sprint
        return sprint_id >= 1

    def add_incident_delta(self, product_id, sprint_id, delta_money):
        """Apply an additive incident value change to a sprint."""
        self.incident_value_deltas[product_id - 1][sprint_id - 1] += delta_money

    def set_incident_value(self, product_id, sprint_id, absolute_money_value):
        """Override a sprint value due to an incident card."""
        self.incident_value_overrides[product_id - 1][sprint_id - 1] = absolute_money_value

    def add_refinement_delta(self, product_id, sprint_id, delta_features):
        """Apply a refinement feature change to a future sprint."""
        current_delta = self.refinement_feature_deltas[product_id - 1][sprint_id - 1]
        new_delta = current_delta + delta_features
        base_features = self.board_features[product_id - 1][sprint_id - 1]
        self.refinement_feature_deltas[product_id - 1][sprint_id - 1] = max(1 - base_features, new_delta)

    def _resolve_sprint_for_product(self, product_id, info):
        """Resolve one full sprint for the selected product."""
        product_state = self._compute_product_state(product_id)
        if product_state["completed"]:
            info["invalid_action"] = True
            info["invalid_action_reason"] = "played_completed_product"
            return "Invalid"

        self.current_product = product_id

        scrum_result = self._play_daily_scrums(product_state["features_required"])
        net_result = scrum_result["net_result"]
        payout = self._calculate_sprint_payout(net_result, product_state["sprint_value"])
        self.current_money += payout

        info["played_product"] = product_id
        info["played_sprint"] = product_state["next_sprint"]
        info["product_name"] = self.product_names[product_id - 1]
        info["features_required"] = product_state["features_required"]
        info["sprint_value"] = product_state["sprint_value"]
        info["daily_scrums"] = scrum_result["daily_scrums"]
        info["net_result"] = net_result
        info["payout"] = payout
        info["success"] = net_result <= 0

        if net_result <= 0:
            self.product_next_sprints[product_id - 1] += 1
            if self._is_product_complete(product_id):
                info["product_completed"] = True
            return "Success"

        return "Failure"

    def _refresh_observation_fields(self):
        """Refresh the visible observation fields from product progress."""
        current_product_state = self._compute_product_state(self.current_product)

        self.current_sprint = current_product_state["next_sprint"]
        self.features_required = current_product_state["features_required"]
        self.sprint_value = current_product_state["sprint_value"]
        self.win_probability = current_product_state["win_probability"]
        self.expected_value = current_product_state["expected_value"]
        self.current_product_completed = current_product_state["completed"]
        self.current_refinement_delta = current_product_state["refinement_delta"]
        self.current_incident_delta = current_product_state["incident_delta"]

        self.remaining_turns = max(self.max_turns - self.turn_count, 0)
        self.is_last_sprint = (
            self.current_sprint == self.sprints_per_product and not self.current_product_completed
        )
        self.loan_active = self.loans_taken > 0
        self.interest_due = self.loan_interest * self.loans_taken
        self.debt_ratio = self._calculate_debt_ratio()
        self.switch_is_free = self.current_product_completed

    def _get_state(self):
        """Return the advanced observation used by the deep-RL branch."""
        target_next_sprints = []
        target_features_required = []
        target_sprint_values = []
        target_win_probabilities = []
        target_expected_values = []
        target_is_completed = []
        target_incident_deltas = []
        target_refinement_deltas = []
        target_incident_flags = []

        for product_id in range(1, self.products_count + 1):
            product_state = self._compute_product_state(product_id)
            target_next_sprints.append(product_state["next_sprint"])
            target_features_required.append(product_state["features_required"])
            target_sprint_values.append(product_state["sprint_value"])
            target_win_probabilities.append(product_state["win_probability"])
            target_expected_values.append(product_state["expected_value"])
            target_is_completed.append(int(product_state["completed"]))
            target_incident_deltas.append(product_state["incident_delta"])
            target_refinement_deltas.append(product_state["refinement_delta"])
            target_incident_flags.append(int(product_state["incident_delta"] != 0 or product_state["incident_override_active"]))

        return {
            "current_money": self.current_money,
            "current_product": self.current_product,
            "current_sprint": self.current_sprint,
            "features_required": self.features_required,
            "sprint_value": self.sprint_value,
            "loan_active": self.loan_active,
            "interest_due": self.interest_due,
            "win_probability": self.win_probability,
            "expected_value": self.expected_value,
            "remaining_turns": self.remaining_turns,
            "is_last_sprint": int(self.is_last_sprint),
            "debt_ratio": self.debt_ratio,
            "switch_is_free": int(self.switch_is_free),
            "incident_active": int(self.incident_active),
            "current_incident_id": self.current_incident_id,
            "current_incident_scope": self.current_incident_scope,
            "current_incident_delta": self.current_incident_delta,
            "current_refinement_delta": self.current_refinement_delta,
            "current_product_completed": int(self.current_product_completed),
            "target_next_sprints": target_next_sprints,
            "target_features_required": target_features_required,
            "target_sprint_values": target_sprint_values,
            "target_win_probabilities": target_win_probabilities,
            "target_expected_values": target_expected_values,
            "target_is_completed": target_is_completed,
            "target_incident_deltas": target_incident_deltas,
            "target_refinement_deltas": target_refinement_deltas,
            "target_incident_flags": target_incident_flags,
        }

    def _compute_product_state(self, product_id):
        """Compute the visible state of one product's next sprint."""
        product_index = product_id - 1
        next_sprint = self.product_next_sprints[product_index]
        completed = next_sprint > self.sprints_per_product

        if completed:
            return {
                "next_sprint": 0,
                "features_required": 0,
                "sprint_value": 0,
                "win_probability": 0.0,
                "expected_value": 0.0,
                "completed": True,
                "incident_delta": 0,
                "refinement_delta": 0,
                "incident_override_active": False,
            }

        sprint_index = next_sprint - 1
        base_features = self.board_features[product_index][sprint_index]
        refinement_delta = self.refinement_feature_deltas[product_index][sprint_index]
        features_required = max(1, base_features + refinement_delta)

        base_value = self.board_ring_values[product_index][sprint_index] * self.ring_value
        incident_delta = self.incident_value_deltas[product_index][sprint_index]
        incident_override = self.incident_value_overrides[product_index][sprint_index]

        if incident_override is not None:
            sprint_value = incident_override + incident_delta
            incident_override_active = True
        else:
            sprint_value = base_value + incident_delta
            incident_override_active = False

        sprint_value = max(0, sprint_value)
        win_probability = self._get_win_probability_for_features(features_required)
        expected_value = sprint_value * win_probability

        return {
            "next_sprint": next_sprint,
            "features_required": features_required,
            "sprint_value": sprint_value,
            "win_probability": win_probability,
            "expected_value": expected_value,
            "completed": False,
            "incident_delta": sprint_value - base_value,
            "refinement_delta": refinement_delta,
            "incident_override_active": incident_override_active,
        }

    def _get_win_probability_for_features(self, features_required):
        """Map effective feature count to the corresponding success probability."""
        if features_required <= 1:
            return self.win_probability_lookup[1]
        if features_required == 2:
            return self.win_probability_lookup[2]
        return self.win_probability_lookup[3]

    def _encode_incident_scope(self, incident_card):
        """Encode incident scope as a normalized float for the network."""
        if incident_card.effect_type == "adjust_specific_sprint_globally":
            return 1.0
        return len(incident_card.target_products) / max(self.products_count, 1)

    def _is_product_complete(self, product_id):
        """Return whether a product has already cleared its last sprint."""
        return self.product_next_sprints[product_id - 1] > self.sprints_per_product

    def _play_daily_scrums(self, features_required):
        """Resolve the exact classical 5 Daily Scrum mechanic."""
        dice_count, dice_sides = self._get_dice_setup(features_required)
        daily_scrums = []
        net_result = 0

        for scrum_index in range(self.daily_scrums_per_sprint):
            rolls = [random.randint(1, dice_sides) for _ in range(dice_count)]
            roll_total = sum(rolls)
            scrum_difference = roll_total - self.daily_scrum_target
            net_result += scrum_difference
            daily_scrums.append(
                {
                    "scrum_number": scrum_index + 1,
                    "dice_count": dice_count,
                    "dice_sides": dice_sides,
                    "rolls": rolls,
                    "roll_total": roll_total,
                    "difference_from_target": scrum_difference,
                }
            )

        return {"daily_scrums": daily_scrums, "net_result": net_result}

    def _get_dice_setup(self, features_required):
        """Map feature count to the correct classical dice set."""
        if features_required <= 1:
            return 1, 20
        if features_required == 2:
            return 2, 10
        return 3, 6

    def _build_win_probability_lookup(self):
        """Precompute exact 5-scrum success probabilities for each dice regime."""
        lookup = {}
        for feature_key in (1, 2, 3):
            dice_count, dice_sides = self._get_dice_setup(feature_key)
            single_scrum_distribution = self._single_scrum_sum_distribution(dice_count, dice_sides)

            five_scrum_distribution = {0: 1.0}
            for _ in range(self.daily_scrums_per_sprint):
                five_scrum_distribution = self._convolve_distributions(
                    five_scrum_distribution,
                    single_scrum_distribution,
                )

            success_threshold = self.daily_scrums_per_sprint * self.daily_scrum_target
            success_probability = sum(
                probability
                for total_roll, probability in five_scrum_distribution.items()
                if total_roll <= success_threshold
            )
            lookup[feature_key] = success_probability
        return lookup

    def _single_scrum_sum_distribution(self, dice_count, dice_sides):
        """Return the exact probability distribution for one scrum roll total."""
        distribution = {0: 1.0}
        for _ in range(dice_count):
            next_distribution = {}
            for current_total, current_probability in distribution.items():
                for die_face in range(1, dice_sides + 1):
                    next_total = current_total + die_face
                    next_distribution[next_total] = (
                        next_distribution.get(next_total, 0.0) + current_probability / dice_sides
                    )
            distribution = next_distribution
        return distribution

    def _convolve_distributions(self, left_distribution, right_distribution):
        """Convolve two discrete probability distributions."""
        result = {}
        for left_total, left_probability in left_distribution.items():
            for right_total, right_probability in right_distribution.items():
                combined_total = left_total + right_total
                result[combined_total] = (
                    result.get(combined_total, 0.0) + left_probability * right_probability
                )
        return result

    def _calculate_sprint_payout(self, net_result, sprint_value):
        """Convert the scrum net result into the money outcome for one sprint."""
        if net_result == 0:
            return sprint_value
        if net_result < 0:
            return sprint_value - (self.penalty_negative * abs(net_result))
        return -(self.penalty_positive * net_result)

    def _apply_required_payment(self, required_amount, info):
        """Trigger mandatory loans automatically when a required payment cannot be made."""
        if required_amount <= 0:
            return 0

        while self.current_money < required_amount:
            self.current_money += self.mandatory_loan_amount
            self.loans_taken += 1
            self.loan_active = True
            self.interest_due = self.loan_interest * self.loans_taken
            self.just_took_mandatory_loan = True
            info["loan_triggered"] = True
            info["loans_taken"] = self.loans_taken

        return required_amount

    def _calculate_debt_ratio(self):
        """Estimate the financial burden of outstanding interest relative to cash."""
        if self.current_money <= 0:
            return 2.0
        return min(self.interest_due / self.current_money, 2.0)


def discretize_state(state):
    """Coarsen the advanced observation into a compact hashable tuple."""
    if not isinstance(state, dict):
        raise TypeError("The advanced deep-RL environment returns dictionary states.")

    current_money = state["current_money"]
    current_product = int(state["current_product"])
    current_sprint = int(state["current_sprint"])
    win_probability = float(state["win_probability"])
    expected_value = float(state["expected_value"])
    remaining_turns = int(state["remaining_turns"])
    debt_ratio = float(state["debt_ratio"])
    switch_is_free = int(bool(state["switch_is_free"]))
    current_incident_id = int(state.get("current_incident_id", 0))

    if current_money < 0:
        money_bucket = "Bankrupt"
    elif current_money < 10000:
        money_bucket = "Low"
    elif current_money < 30000:
        money_bucket = "Medium"
    elif current_money < 60000:
        money_bucket = "High"
    else:
        money_bucket = "VeryHigh"

    if win_probability < 0.35:
        probability_bucket = "Low"
    elif win_probability < 0.65:
        probability_bucket = "Medium"
    else:
        probability_bucket = "High"

    if expected_value < 5000:
        expected_value_bucket = "Low"
    elif expected_value < 15000:
        expected_value_bucket = "Medium"
    else:
        expected_value_bucket = "High"

    if debt_ratio == 0:
        debt_bucket = "None"
    elif debt_ratio < 0.25:
        debt_bucket = "Manageable"
    elif debt_ratio < 0.75:
        debt_bucket = "Heavy"
    else:
        debt_bucket = "Critical"

    return (
        money_bucket,
        current_product,
        current_sprint,
        probability_bucket,
        expected_value_bucket,
        remaining_turns,
        debt_bucket,
        switch_is_free,
        current_incident_id,
    )
