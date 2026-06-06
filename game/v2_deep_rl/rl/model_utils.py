"""
Model Serialization and Reporting Utilities.

This module provides utility functions to persist training logs, metrics,
reports, and tabular Q-tables. By writing these outputs in human-readable
formats (JSON/CSV), the results are easy to parse for assessments, graphs,
and spreadsheets.

Connections:
  - Imported by: `training.train_dqn.py` to write final run metrics and training logs.
  - Imported by: `evaluation.evaluate_ddqn_robustness.py` to output comparative CSV summaries.
"""

import csv
import json
from pathlib import Path


def ensure_directory(path: str | Path):
    """Ensure that the parent directory tree for a target path is fully created."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_q_table(agent_name: str, q_table: dict, output_dir: str = "artifacts/models") -> str:
    """
    Save a tabular RL Q-table as a human-readable JSON array of state-action pairs.
    
    Used primarily for legacy or baseline tabular Q-learning comparison agents.
    
    Args:
        agent_name: Prefix name for the output JSON file.
        q_table: Dictionary mapping discrete states (tuples) to lists of action Q-values.
        output_dir: Directory path where the file is stored.
        
    Returns:
        str: Absolute or relative output file path.
    """
    ensure_directory(output_dir)
    output_path = Path(output_dir) / f"{agent_name}_q_table.json"

    serializable_q_table = []
    for state, q_values in q_table.items():
        serializable_q_table.append(
            {
                "state": list(state),
                "q_values": q_values,
            }
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable_q_table, file, indent=2)

    return str(output_path)


def save_metrics_json(metrics: dict, output_path: str | Path):
    """
    Save dictionary metrics as formatted JSON (e.g. final average rewards, wins).
    """
    ensure_directory(Path(output_path).parent)

    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_metrics_csv(rows: list[dict], output_path: str | Path):
    """
    Save a list of dictionary rows to a CSV file.
    
    Renders structured experiment logs so they can be easily loaded in Excel or Python pandas.
    """
    ensure_directory(Path(output_path).parent)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with Path(output_path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_text_report(report_text: str, output_path: str | Path):
    """
    Write a plain-text or Markdown summary report file.
    """
    ensure_directory(Path(output_path).parent)
    Path(output_path).write_text(report_text, encoding="utf-8")

