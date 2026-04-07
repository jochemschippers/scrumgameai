import random


class StandardRefinementModel301:
    """
    Standard refinement logic derived from the Scrum Game manuals.

    Product groups:
    - Card 1: Yellow, Red -> increase on 1-2, decrease on 19-20
    - Card 2: Orange, Green, Purple -> increase on 1-3, decrease on 19-20
    - Card 3: Blue -> increase on 1-4, decrease on 19-20
    - Card 4: Black -> increase on 1, decrease on 20
    """

    def __init__(self, product_names):
        self.product_lookup = {name.lower(): index + 1 for index, name in enumerate(product_names)}
        self.group_rules = {
            "yellow": {"increase_rolls": {1, 2}, "decrease_rolls": {19, 20}},
            "red": {"increase_rolls": {1, 2}, "decrease_rolls": {19, 20}},
            "orange": {"increase_rolls": {1, 2, 3}, "decrease_rolls": {19, 20}},
            "green": {"increase_rolls": {1, 2, 3}, "decrease_rolls": {19, 20}},
            "purple": {"increase_rolls": {1, 2, 3}, "decrease_rolls": {19, 20}},
            "blue": {"increase_rolls": {1, 2, 3, 4}, "decrease_rolls": {19, 20}},
            "black": {"increase_rolls": {1}, "decrease_rolls": {20}},
        }

    def apply(self, env, product_id):
        """
        Apply one refinement roll to the product that was played this turn.

        Increase:
        - add one feature to every future sprint of that product

        Decrease:
        - remove one feature from the last future sprint of that product
        """
        product_name = env.product_names[product_id - 1].lower()
        rules = self.group_rules[product_name]
        roll = random.randint(1, 20)

        result = {
            "roll": roll,
            "effect": "none",
            "target_product": product_id,
            "future_sprints_changed": [],
        }

        future_sprints = [
            sprint_id
            for sprint_id in range(1, env.sprints_per_product + 1)
            if env.is_sprint_future(product_id, sprint_id, future_only=True)
        ]

        if not future_sprints:
            return result

        if roll in rules["increase_rolls"]:
            for sprint_id in future_sprints:
                env.add_refinement_delta(product_id, sprint_id, 1)
                result["future_sprints_changed"].append({"sprint_id": sprint_id, "delta": 1})
            result["effect"] = "increase"
            return result

        if roll in rules["decrease_rolls"]:
            last_future_sprint = future_sprints[-1]
            env.add_refinement_delta(product_id, last_future_sprint, -1)
            result["future_sprints_changed"].append({"sprint_id": last_future_sprint, "delta": -1})
            result["effect"] = "decrease"
            return result

        return result
