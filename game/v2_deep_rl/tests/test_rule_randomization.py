"""Test rule randomization behavior."""

from __future__ import annotations

import random

from config.config_manager import load_game_config
from game_rules.rule_randomization import sample_game_config


# Verify sample game config keeps model dimensions stable.
def test_sample_game_config_keeps_model_dimensions_stable():
    base = load_game_config()
    sampled = sample_game_config(base, random.Random(123))

    assert sampled.products_count == base.products_count
    assert sampled.sprints_per_product == base.sprints_per_product
    assert len(sampled.dice_rules) == len(base.dice_rules)
    assert sampled.product_names == base.product_names


# Verify sample game config obeys overridden bounds.
def test_sample_game_config_obeys_overridden_bounds():
    base = load_game_config()
    sampled = sample_game_config(
        base,
        random.Random(123),
        bounds={
            "starting_money": [20000, 20000],
            "max_turns": [8, 8],
            "incident_draw_probability": [0.4, 0.4],
            "incident_severity_multiplier": [1.5, 1.5],
            "dice_sides": [12, 12],
            "dice_count": [2, 2],
        },
    )

    assert sampled.starting_money == 20000
    assert sampled.max_turns == 8
    assert sampled.incident.draw_probability == 0.4
    assert sampled.incident.severity_multiplier == 1.5
    assert {rule.dice_sides for rule in sampled.dice_rules} == {12}
    assert {rule.dice_count for rule in sampled.dice_rules} == {2}


# Verify sample game config is deterministic for seed.
def test_sample_game_config_is_deterministic_for_seed():
    base = load_game_config()
    left = sample_game_config(base, random.Random(999)).to_dict()
    right = sample_game_config(base, random.Random(999)).to_dict()

    assert left == right
