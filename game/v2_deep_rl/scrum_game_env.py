import random


class ScrumGameEnv:
    """Gym-style environment for the advanced deep-RL Scrum Game branch."""

    def __init__(
        self,
        classical_setup=True,
        enable_incidents=True,
        enable_refinements=True,
        incident_probability=0.2,
        refinement_probability=0.3,
        max_refinement_bonus=2,
        incident_delta_options=(-10000, -5000, 5000, 10000),
    ):
        self.classical_setup = classical_setup
        self.enable_incidents = enable_incidents
        self.enable_refinements = enable_refinements
        self.incident_probability = incident_probability
        self.refinement_probability = refinement_probability
        self.max_refinement_bonus = max_refinement_bonus
        self.incident_delta_options = tuple(incident_delta_options)

        # Classical board configuration derived from the project prototype.
        self.products_count = 7
        self.sprints_per_product = 4
        self.max_turns = 6
        self.num_actions = 1 + self.products_count

        # Economy values from the classical setup.
        self.starting_money = 25000
        self.ring_value = 5000
        self.cost_continue = 0
        self.cost_switch_mid = 5000
        self.cost_switch_after = 0
        self.mandatory_loan_amount = 50000
        self.loan_interest = 5000
        self.penalty_negative = 1000
        self.penalty_positive = 5000

        # Scrum mechanic values from the rule data.
        self.daily_scrums_per_sprint = 5
        self.daily_scrum_target = 12

        # Classical 7 x 4 board matrix from the prototype.
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

        self.max_abs_incident_delta = max(abs(delta) for delta in self.incident_delta_options) if self.incident_delta_options else 1
        self.max_base_sprint_value = max(max(row) for row in self.board_ring_values) * self.ring_value
        self.max_visible_sprint_value = self.max_base_sprint_value + self.max_abs_incident_delta
        self.max_interest_reference = self.loan_interest * 6

        self.win_probability_lookup = self._build_win_probability_lookup()

        # Dynamic environment state.
        self.current_money = self.starting_money
        self.current_product = 1
        self.current_sprint = 1
        self.features_required = 0
        self.sprint_value = 0
        self.win_probability = 0.0
        self.expected_value = 0.0
        self.remaining_turns = self.max_turns
        self.is_last_sprint = False
        self.debt_ratio = 0.0
        self.switch_is_free = False
        self.current_incident_delta = 0
        self.current_refinement_bonus = 0
        self.current_product_completed = False
        self.loan_active = False
        self.interest_due = 0

        # Product-level state.
        self.product_next_sprints = [1] * self.products_count
        self.product_refinements = [0] * self.products_count
        self.product_incident_deltas = [0] * self.products_count

        # Episode bookkeeping.
        self.turn_count = 0
        self.loans_taken = 0
        self.turns_with_loan = 0
        self.just_took_mandatory_loan = False

        self._roll_new_incidents()
        self._refresh_observation_fields()

    def reset(self, seed=None):
        """Reset the environment to the classical starting state."""
        if seed is not None:
            random.seed(seed)

        self.current_money = self.starting_money
        self.current_product = 1
        self.loan_active = False
        self.interest_due = 0

        self.product_next_sprints = [1] * self.products_count
        self.product_refinements = [0] * self.products_count
        self.product_incident_deltas = [0] * self.products_count

        self.turn_count = 0
        self.loans_taken = 0
        self.turns_with_loan = 0
        self.just_took_mandatory_loan = False

        self._roll_new_incidents()
        self._refresh_observation_fields()
        return self._get_state()

    def step(self, action):
        """
        Execute one environment turn.

        Action space:
        0 -> Continue on the active product
        1..7 -> Switch to Product N
        """
        if action not in range(self.num_actions):
            raise ValueError(f"Action must be in [0, {self.num_actions - 1}].")

        self.turn_count += 1
        old_money = self.current_money
        info = {
            "action": action,
            "action_name": self.action_name(action),
            "loan_triggered": False,
            "selected_product": action if action > 0 else self.current_product,
            "invalid_action": False,
            "interest_paid": 0,
            "switch_cost_paid": 0,
            "incident_triggered": False,
            "incident_deltas": list(self.product_incident_deltas),
            "refinement_gained": 0,
        }
        self.just_took_mandatory_loan = False

        if self.loan_active and self.interest_due > 0:
            interest_paid = self._apply_required_payment(self.interest_due, info)
            self.current_money -= interest_paid
            info["interest_paid"] = interest_paid

        if action == 0:
            result = self._handle_continue_action(info)
        else:
            result = self._handle_switch_action(action, info)

        if self.loan_active:
            self.turns_with_loan += 1
        else:
            self.turns_with_loan = 0

        self._roll_new_incidents()
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

        if result == "Switch" and action > 0 and old_money < 10000:
            reward += 1000

        if self.just_took_mandatory_loan:
            reward -= 20000

        if info.get("invalid_action"):
            reward -= 2000

        if result == "Success" and info.get("refinement_gained", 0) > 0:
            reward += 500

        return reward

    def action_name(self, action):
        """Convert an action id to a readable label."""
        if action == 0:
            return "Continue"
        return f"Switch to Product {action}"

    def build_reference_state(self, product_id, sprint_id, current_money=25000, loan_active=False, interest_due=0):
        """
        Build a synthetic reference observation for dashboard visualizations.

        This keeps the active cell configurable while leaving all other products
        at sprint 1 with no incidents or refinements.
        """
        target_product_id = max(1, min(self.products_count, int(product_id)))
        target_sprint_id = max(1, min(self.sprints_per_product, int(sprint_id)))

        self.product_next_sprints = [1] * self.products_count
        self.product_next_sprints[target_product_id - 1] = target_sprint_id
        self.product_refinements = [0] * self.products_count
        self.product_incident_deltas = [0] * self.products_count

        self.current_money = current_money
        self.current_product = target_product_id
        self.turn_count = 0
        self.loans_taken = 1 if loan_active else 0
        self.loan_active = loan_active
        self.interest_due = interest_due
        self.turns_with_loan = 1 if loan_active else 0
        self.just_took_mandatory_loan = False

        self._refresh_observation_fields()
        return self._get_state()

    def _handle_continue_action(self, info):
        """Resolve the current sprint on the active product."""
        if self.current_product_completed or self.current_sprint == 0:
            info["invalid_action"] = True
            info["invalid_action_reason"] = "cannot_continue_completed_product"
            return "Invalid"

        scrum_result = self._play_daily_scrums(self.features_required)
        net_result = scrum_result["net_result"]
        payout = self._calculate_sprint_payout(net_result, self.sprint_value)

        self.current_money += payout
        info["daily_scrums"] = scrum_result["daily_scrums"]
        info["net_result"] = net_result
        info["payout"] = payout
        info["success"] = net_result <= 0

        if net_result <= 0:
            product_index = self.current_product - 1
            if self.enable_refinements and self.product_next_sprints[product_index] < self.sprints_per_product:
                if random.random() < self.refinement_probability:
                    previous_bonus = self.product_refinements[product_index]
                    self.product_refinements[product_index] = min(
                        self.max_refinement_bonus,
                        self.product_refinements[product_index] + 1,
                    )
                    if self.product_refinements[product_index] > previous_bonus:
                        info["refinement_gained"] = 1

            self.product_next_sprints[product_index] += 1
            if self._is_product_complete(self.current_product):
                info["product_completed"] = True
            return "Success"

        return "Failure"

    def _handle_switch_action(self, action, info):
        """Switch the active focus to a chosen product."""
        target_product = action
        info["selected_product"] = target_product

        if target_product == self.current_product:
            info["invalid_action"] = True
            info["invalid_action_reason"] = "switch_to_current_product"
            return "Invalid"

        if self._is_product_complete(target_product):
            info["invalid_action"] = True
            info["invalid_action_reason"] = "switch_to_completed_product"
            return "Invalid"

        switch_cost = 0 if self.switch_is_free else self.cost_switch_mid
        switch_cost = self._apply_required_payment(switch_cost, info)
        self.current_money -= switch_cost
        info["switch_cost_paid"] = switch_cost

        self.current_product = target_product
        return "Switch"

    def _refresh_observation_fields(self):
        """Refresh the observable state derived from product progress."""
        current_product_state = self._compute_product_state(self.current_product)

        self.current_sprint = current_product_state["next_sprint"]
        self.features_required = current_product_state["features_required"]
        self.sprint_value = current_product_state["sprint_value"]
        self.win_probability = current_product_state["win_probability"]
        self.expected_value = current_product_state["expected_value"]
        self.current_incident_delta = current_product_state["incident_delta"]
        self.current_refinement_bonus = current_product_state["refinement_bonus"]
        self.current_product_completed = current_product_state["completed"]

        self.remaining_turns = max(self.max_turns - self.turn_count, 0)
        self.is_last_sprint = self.current_sprint == self.sprints_per_product and not self.current_product_completed
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

        for product_id in range(1, self.products_count + 1):
            product_state = self._compute_product_state(product_id)
            target_next_sprints.append(product_state["next_sprint"])
            target_features_required.append(product_state["features_required"])
            target_sprint_values.append(product_state["sprint_value"])
            target_win_probabilities.append(product_state["win_probability"])
            target_expected_values.append(product_state["expected_value"])
            target_is_completed.append(int(product_state["completed"]))
            target_incident_deltas.append(product_state["incident_delta"])

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
            "current_incident_delta": self.current_incident_delta,
            "current_refinement_bonus": self.current_refinement_bonus,
            "current_product_completed": int(self.current_product_completed),
            "target_next_sprints": target_next_sprints,
            "target_features_required": target_features_required,
            "target_sprint_values": target_sprint_values,
            "target_win_probabilities": target_win_probabilities,
            "target_expected_values": target_expected_values,
            "target_is_completed": target_is_completed,
            "target_incident_deltas": target_incident_deltas,
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
                "refinement_bonus": self.product_refinements[product_index],
            }

        refinement_bonus = self.product_refinements[product_index]
        base_features = self.board_features[product_index][next_sprint - 1]
        features_required = max(1, base_features - refinement_bonus)
        base_sprint_value = self.board_ring_values[product_index][next_sprint - 1] * self.ring_value
        incident_delta = self.product_incident_deltas[product_index]
        sprint_value = max(0, base_sprint_value + incident_delta)
        win_probability = self._get_win_probability_for_features(features_required)
        expected_value = sprint_value * win_probability

        return {
            "next_sprint": next_sprint,
            "features_required": features_required,
            "sprint_value": sprint_value,
            "win_probability": win_probability,
            "expected_value": expected_value,
            "completed": False,
            "incident_delta": incident_delta,
            "refinement_bonus": refinement_bonus,
        }

    def _get_win_probability_for_features(self, features_required):
        """Map effective feature count to the corresponding success probability."""
        if features_required <= 1:
            return self.win_probability_lookup[1]
        if features_required == 2:
            return self.win_probability_lookup[2]
        return self.win_probability_lookup[3]

    def _roll_new_incidents(self):
        """Roll visible incident modifiers for the next decision state."""
        if not self.enable_incidents:
            self.product_incident_deltas = [0] * self.products_count
            return

        new_deltas = []
        for product_id in range(1, self.products_count + 1):
            if self._is_product_complete(product_id):
                new_deltas.append(0)
            elif random.random() < self.incident_probability:
                new_deltas.append(random.choice(self.incident_delta_options))
            else:
                new_deltas.append(0)

        self.product_incident_deltas = new_deltas

    def _is_product_complete(self, product_id):
        """Return whether a product has already cleared sprint 4."""
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

        return {
            "daily_scrums": daily_scrums,
            "net_result": net_result,
        }

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
    if isinstance(state, dict):
        current_money = state["current_money"]
        current_product = int(state["current_product"])
        current_sprint = int(state["current_sprint"])
        win_probability = float(state["win_probability"])
        expected_value = float(state["expected_value"])
        remaining_turns = int(state["remaining_turns"])
        debt_ratio = float(state["debt_ratio"])
        switch_is_free = int(bool(state["switch_is_free"]))
    else:
        raise TypeError("The advanced deep-RL environment returns dictionary states.")

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
    )
