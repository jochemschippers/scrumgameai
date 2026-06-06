"""
Evaluation Package.

This package houses scripts for multi-seed robustness analysis of trained agents.

Exposes:
  - evaluate_across_seeds: Runs greedy trials across specified seeds.
  - evaluate_one_seed: Runs a single greedy trial for a given seed.
"""

from __future__ import annotations

from .evaluate_ddqn_robustness import evaluate_across_seeds, evaluate_one_seed

__all__ = [
    "evaluate_across_seeds",
    "evaluate_one_seed",
]
