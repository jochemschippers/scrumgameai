import random


class ScrumGameEnv:
    """Gym-style environment for the classical Scrum Game setup."""

    def __init__(self):
        # Classical board configuration taken from the game data prototype.
        self.products_count = 7
        self.sprints_per_product = 4
        self.max_turns = 6

        # Real economy values from the classical setup.
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

        # State variables required by the existing training code.
        self.current_money = self.starting_money
        self.current_product = 1
        self.current_sprint = 1
        self.features_required = self._get_current_features()
        self.sprint_value = self._get_current_sprint_value()
        self.loan_active = False
        self.interest_due = 0

        # Episode bookkeeping.
        self.turn_count = 0
        self.loans_taken = 0

    def reset(self, seed=None):
        """Reset the environment to the classical starting state."""
        if seed is not None:
            random.seed(seed)

        self.current_money = self.starting_money
        self.current_product = 1
        self.current_sprint = 1
        self.features_required = self._get_current_features()
        self.sprint_value = self._get_current_sprint_value()
        self.loan_active = False
        self.interest_due = 0
        self.turn_count = 0
        self.loans_taken = 0

        return self._get_state()

    def step(self, action):
        """
        Execute one environment turn.

        Supported actions:
        0 -> Continue current sprint
        1 -> Switch to the next product and start at sprint 1 there

        Compatibility note:
        Older agent code in this repository still assumes three actions.
        To avoid crashing those scripts, action 2 is treated as action 0.
        """
        original_action = action
        if action == 2:
            action = 0

        if action not in (0, 1):
            raise ValueError("Action must be 0 (Continue) or 1 (Switch).")

        self.turn_count += 1
        reward = 0
        info = {
            "action": action,
            "original_action": original_action,
            "loan_triggered": False,
        }

        # Loan interest is charged at the start of every turn after a loan exists.
        if self.loan_active and self.interest_due > 0:
            interest_paid = self._apply_required_payment(self.interest_due, info)
            self.current_money -= interest_paid
            reward -= interest_paid
            info["interest_paid"] = interest_paid
        else:
            info["interest_paid"] = 0

        if action == 1:
            # Switching mid-product costs money. Finishing sprint 4 already moves
            # the player to the next product for free, so this action represents
            # an active mid-stream switch.
            switch_cost = self.cost_switch_mid
            switch_cost = self._apply_required_payment(switch_cost, info)
            self.current_money -= switch_cost
            reward -= switch_cost

            self.current_product = self._get_next_product_id()
            self.current_sprint = 1
            self._refresh_observation_fields()
            info["result"] = "Switched product."
            info["switch_cost_paid"] = switch_cost

        else:
            info["switch_cost_paid"] = 0

            # A full turn always resolves exactly five Daily Scrums.
            scrum_result = self._play_daily_scrums(self.features_required)
            net_result = scrum_result["net_result"]
            payout = self._calculate_sprint_payout(net_result)

            self.current_money += payout
            reward += payout

            info["result"] = "Sprint resolved."
            info["daily_scrums"] = scrum_result["daily_scrums"]
            info["net_result"] = net_result
            info["payout"] = payout
            info["success"] = net_result <= 0

            if net_result <= 0:
                self._advance_after_success()
            else:
                # On failure, the player stays on the same product/sprint and can
                # decide on the next turn whether to continue or switch away.
                self._refresh_observation_fields()

        done = False
        terminal_reason = None

        if self.current_money < 0:
            reward -= 100000
            done = True
            terminal_reason = "bankruptcy"
        elif self.turn_count >= self.max_turns:
            done = True
            terminal_reason = "max_turns_reached"

        info["turn_count"] = self.turn_count
        if terminal_reason is not None:
            info["terminal_reason"] = terminal_reason

        next_state = self._get_state()
        return next_state, reward, done, info

    def _get_state(self):
        """Return the current state tuple used by the training scripts."""
        return (
            self.current_money,
            self.current_product,
            self.current_sprint,
            self.features_required,
            self.sprint_value,
            self.loan_active,
            self.interest_due,
        )

    def _refresh_observation_fields(self):
        """Refresh the observable sprint fields from the current board position."""
        self.features_required = self._get_current_features()
        self.sprint_value = self._get_current_sprint_value()
        self.loan_active = self.loans_taken > 0
        self.interest_due = self.loan_interest * self.loans_taken

    def _get_current_features(self):
        """Look up the feature count for the current board coordinate."""
        return self.board_features[self.current_product - 1][self.current_sprint - 1]

    def _get_current_sprint_value(self):
        """Look up the current sprint value in money, not ring count."""
        ring_count = self.board_ring_values[self.current_product - 1][self.current_sprint - 1]
        return ring_count * self.ring_value

    def _get_next_product_id(self):
        """
        Move to the next product in a circular order.

        The board rules allow choosing another product. Because the environment
        uses a small discrete action space, switching deterministically advances
        to the next product ID.
        """
        return (self.current_product % self.products_count) + 1

    def _advance_after_success(self):
        """
        Advance the board position after a successful sprint.

        Sprint 1-3 success moves to the next sprint on the same product.
        Sprint 4 success completes that product and moves to sprint 1 of the
        next product for free, matching the classical setup.
        """
        if self.current_sprint < self.sprints_per_product:
            self.current_sprint += 1
        else:
            self.current_product = self._get_next_product_id()
            self.current_sprint = 1

        self._refresh_observation_fields()

    def _play_daily_scrums(self, features_required):
        """
        Resolve the exact classical 5 Daily Scrum mechanic.

        Dice by feature count:
        - 1 feature  -> 1 x D20
        - 2 features -> 2 x D10
        - 3+ features -> 3 x D6

        For each Daily Scrum:
        - roll the dice
        - subtract the target value (12 by default)
        - accumulate the signed difference

        Classical outcome rule:
        - net <= 0 -> sprint succeeds
        - net > 0  -> sprint fails
        """
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

    def _calculate_sprint_payout(self, net_result):
        """
        Convert the scrum net result into the actual money outcome.

        Classical payout rules:
        - net == 0: sprint succeeds, full payout
        - net < 0: sprint succeeds, payout reduced by penalty_negative * |net|
        - net > 0: sprint fails, only a penalty is applied
        """
        if net_result == 0:
            return self.sprint_value

        if net_result < 0:
            return self.sprint_value - (self.penalty_negative * abs(net_result))

        return -(self.penalty_positive * net_result)

    def _apply_required_payment(self, required_amount, info):
        """
        Trigger mandatory loans automatically when a required payment cannot be made.

        The rules do not allow voluntary loans. Instead, if the player cannot pay
        a required cost, the environment injects a mandatory loan automatically.
        """
        if required_amount <= 0:
            return 0

        while self.current_money < required_amount:
            self.current_money += self.mandatory_loan_amount
            self.loans_taken += 1
            self.loan_active = True
            self.interest_due = self.loan_interest * self.loans_taken
            info["loan_triggered"] = True
            info["loans_taken"] = self.loans_taken

        return required_amount


def discretize_state(state):
    """
    Convert a raw Scrum Game state into a compact hashable tuple.

    The environment now uses real money values, so the money buckets are scaled
    to the classical economy instead of the earlier toy setup.
    """
    if isinstance(state, dict):
        current_money = state["current_money"]
        current_product = int(state["current_product"])
        current_sprint = int(state["current_sprint"])
        features_required = int(state["features_required"])
        sprint_value = int(state["sprint_value"])
        loan_active = int(bool(state["loan_active"]))
        interest_due = int(state["interest_due"])
    else:
        current_money = state[0]
        current_product = int(state[1])
        current_sprint = int(state[2])
        features_required = int(state[3])
        sprint_value = int(state[4])
        loan_active = int(bool(state[5]))
        interest_due = int(state[6])

    if current_money < 0:
        money_bucket = "Bankrupt"
    elif current_money < 10000:
        money_bucket = "Low"
    elif current_money < 20000:
        money_bucket = "Medium"
    elif current_money < 35000:
        money_bucket = "High"
    else:
        money_bucket = "VeryHigh"

    if sprint_value <= 5000:
        sprint_value_bucket = "Small"
    elif sprint_value <= 15000:
        sprint_value_bucket = "Medium"
    elif sprint_value <= 25000:
        sprint_value_bucket = "Large"
    else:
        sprint_value_bucket = "VeryLarge"

    if interest_due == 0:
        interest_bucket = "None"
    elif interest_due <= 5000:
        interest_bucket = "Low"
    elif interest_due <= 10000:
        interest_bucket = "Medium"
    else:
        interest_bucket = "High"

    return (
        money_bucket,
        current_product,
        current_sprint,
        features_required,
        sprint_value_bucket,
        loan_active,
        interest_bucket,
    )
