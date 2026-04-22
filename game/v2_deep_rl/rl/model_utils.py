import csv
import json
from pathlib import Path


def ensure_directory(path):
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_q_table(agent_name, q_table, output_dir="artifacts/models"):
    """
    Save a tabular RL Q-table as JSON.

    JSON is used instead of pickle so the saved artifact remains readable and
    easy to inspect when discussing learned policies in the report.
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


def save_metrics_json(metrics, output_path):
    """Save experiment metrics as formatted JSON."""
    ensure_directory(Path(output_path).parent)

    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_metrics_csv(rows, output_path):
    """Save comparison rows to CSV for easy inclusion in tables and spreadsheets."""
    ensure_directory(Path(output_path).parent)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with Path(output_path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_text_report(report_text, output_path):
    """Save a plain-text or Markdown report artifact."""
    ensure_directory(Path(output_path).parent)
    Path(output_path).write_text(report_text, encoding="utf-8")
