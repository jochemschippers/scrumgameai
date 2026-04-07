from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import altair as alt
import pandas as pd
import streamlit as st
import torch

from dqn_agent import DQNAgent, encode_state
from scrum_game_env import ScrumGameEnv


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
LOG_PATH = REPORTS_DIR / "logs.csv"
EVALUATION_LOG_PATH = REPORTS_DIR / "evaluation_history.csv"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_scrum_model.pth"
PROCESS_STATE_PATH = REPORTS_DIR / "training_process.json"
TRAINING_STDOUT_PATH = REPORTS_DIR / "training_stdout.log"


def ensure_runtime_dirs():
    """Ensure the runtime directories used by the dashboard exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_json_file(file_path):
    """Load JSON from disk and return None if the file is missing or malformed."""
    if not file_path.exists():
        return None

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def save_json_file(file_path, payload):
    """Persist JSON payload to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def remove_process_state():
    """Remove persisted training process metadata."""
    if PROCESS_STATE_PATH.exists():
        PROCESS_STATE_PATH.unlink()


def is_pid_running(pid):
    """Return True when a process id exists on the host."""
    if pid is None or pid <= 0:
        return False

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_training_process_state():
    """Return persisted training state only if the process is still alive."""
    process_state = load_json_file(PROCESS_STATE_PATH)
    if not process_state:
        return None

    pid = int(process_state.get("pid", 0))
    if is_pid_running(pid):
        return process_state

    remove_process_state()
    return None


def launch_training_process():
    """Start the DDQN trainer as an asynchronous background process."""
    ensure_runtime_dirs()
    active_process = get_training_process_state()
    if active_process:
        return False, f"Training already running (PID: {active_process['pid']})."

    with TRAINING_STDOUT_PATH.open("ab") as stdout_handle:
        process = subprocess.Popen(
            ["python3", "train_dqn.py"],
            cwd=BASE_DIR,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    process_state = {
        "pid": process.pid,
        "started_at_epoch": time.time(),
        "command": ["python3", "train_dqn.py"],
        "stdout_log": str(TRAINING_STDOUT_PATH),
    }
    save_json_file(PROCESS_STATE_PATH, process_state)
    return True, f"Training launched in background (PID: {process.pid})."


def stop_training_process():
    """Terminate the tracked background training process."""
    process_state = get_training_process_state()
    if not process_state:
        remove_process_state()
        return False, "No active training process was found."

    pid = int(process_state["pid"])

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        remove_process_state()
        return False, f"Training process {pid} was already gone."
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as error:
            return False, f"Could not stop training process {pid}: {error}"

    deadline = time.time() + 10
    while time.time() < deadline:
        if not is_pid_running(pid):
            remove_process_state()
            return True, f"Training process {pid} stopped."
        time.sleep(0.2)

    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as error:
            return False, f"Training process {pid} did not stop cleanly: {error}"

    remove_process_state()
    return True, f"Training process {pid} was force-killed."


def format_elapsed_time(seconds_elapsed):
    """Render elapsed seconds as a compact human-readable duration."""
    total_seconds = int(max(seconds_elapsed, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def load_training_log():
    """Load the live training CSV if it exists."""
    if not LOG_PATH.exists():
        return None

    training_log = pd.read_csv(LOG_PATH)
    fallback_columns = {
        "average_loan_duration": 0.0,
        "bankruptcy_count": 0.0,
        "average_ending_money": 0.0,
        "invalid_action_count": 0.0,
    }
    for column_name, fallback_value in fallback_columns.items():
        if column_name not in training_log.columns:
            training_log[column_name] = fallback_value

    return training_log


def load_evaluation_log():
    """Load the periodic evaluation history if it exists."""
    if not EVALUATION_LOG_PATH.exists():
        return None
    return pd.read_csv(EVALUATION_LOG_PATH)


def list_checkpoints():
    """List available checkpoints for sidebar selection."""
    if not CHECKPOINT_DIR.exists():
        return []

    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pth"))
    best_first = []
    for checkpoint in checkpoints:
        if checkpoint.name == "best_scrum_model.pth":
            best_first.insert(0, checkpoint)
        else:
            best_first.append(checkpoint)
    return best_first


def load_dqn_policy(checkpoint_path):
    """Load a selected DDQN checkpoint into a greedy policy."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = BASE_DIR / checkpoint_path
    if not checkpoint_path.exists():
        return None, None

    env = ScrumGameEnv()
    state_dim = len(encode_state(env.reset(seed=42), env))
    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=env.num_actions,
        learning_rate=0.0005,
        gamma=0.85,
    )

    try:
        state_dict = torch.load(checkpoint_path, map_location=agent.device)
        agent.policy_network.load_state_dict(state_dict)
        agent.target_network.load_state_dict(state_dict)
        agent.policy_network.eval()
        agent.target_network.eval()
        return agent, None
    except RuntimeError as error:
        return None, str(error)


def action_label(action_id):
    """Convert an action id to a compact readable label."""
    if action_id == 0:
        return "Continue"
    return f"Switch P{action_id}"


def action_short_label(action_id):
    """Convert an action id to a short heatmap label."""
    if action_id == 0:
        return "C"
    return f"P{action_id}"


def build_strategy_map(agent, current_money=25000):
    """Create a heatmap-ready board dataframe from the current DDQN policy."""
    env = ScrumGameEnv()
    cells = []

    for product_id in range(1, env.products_count + 1):
        for sprint_id in range(1, env.sprints_per_product + 1):
            state = env.build_reference_state(
                product_id=product_id,
                sprint_id=sprint_id,
                current_money=current_money,
            )
            state_vector = encode_state(state, env)
            q_values = agent.predict_q_values(state_vector)

            best_action = max(range(env.num_actions), key=lambda action: q_values[action])
            ranked_q_values = sorted(q_values, reverse=True)
            second_best = ranked_q_values[1] if len(ranked_q_values) > 1 else ranked_q_values[0]
            confidence = q_values[best_action] - second_best

            cells.append(
                {
                    "product": f"Product {product_id}",
                    "product_id": product_id,
                    "sprint": f"Sprint {sprint_id}",
                    "sprint_id": sprint_id,
                    "preferred_action_id": best_action,
                    "preferred_action": action_label(best_action),
                    "label": action_short_label(best_action),
                    "confidence": confidence,
                    "q_continue": q_values[0],
                    "best_q_value": q_values[best_action],
                    "win_probability": state["win_probability"],
                    "expected_value": state["expected_value"],
                    "remaining_turns": state["remaining_turns"],
                    "incident_active": state["incident_active"],
                    "current_incident_id": state["current_incident_id"],
                }
            )

    return pd.DataFrame(cells)


def render_strategy_heatmap(strategy_df):
    """Render the 7 x 4 board heatmap with preferred actions."""
    heatmap = (
        alt.Chart(strategy_df)
        .mark_rect()
        .encode(
            x=alt.X("product:N", sort=[f"Product {index}" for index in range(1, 8)], title="Current Product"),
            y=alt.Y("sprint:N", sort=[f"Sprint {index}" for index in range(1, 5)], title="Current Sprint"),
            color=alt.Color("confidence:Q", scale=alt.Scale(scheme="yellowgreenblue"), title="Policy Confidence"),
            tooltip=[
                "product",
                "sprint",
                "preferred_action",
                alt.Tooltip("best_q_value:Q", format=".2f"),
                alt.Tooltip("q_continue:Q", format=".2f"),
                alt.Tooltip("win_probability:Q", format=".3f"),
                alt.Tooltip("expected_value:Q", format=".2f"),
                "current_incident_id",
            ],
        )
        .properties(height=300)
    )

    labels = (
        alt.Chart(strategy_df)
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(
            x=alt.X("product:N", sort=[f"Product {index}" for index in range(1, 8)]),
            y=alt.Y("sprint:N", sort=[f"Sprint {index}" for index in range(1, 5)]),
            text="label:N",
        )
    )

    st.altair_chart(heatmap + labels, use_container_width=True)


def render_switch_target_heatmap(strategy_df):
    """Render only the cells where the agent prefers to switch away."""
    switch_df = strategy_df[strategy_df["preferred_action_id"] > 0].copy()

    if switch_df.empty:
        st.info("The selected checkpoint currently prefers Continue on all reference cells.")
        return

    heatmap = (
        alt.Chart(switch_df)
        .mark_rect()
        .encode(
            x=alt.X("product:N", sort=[f"Product {index}" for index in range(1, 8)], title="Current Product"),
            y=alt.Y("sprint:N", sort=[f"Sprint {index}" for index in range(1, 5)], title="Current Sprint"),
            color=alt.Color("preferred_action_id:Q", scale=alt.Scale(scheme="teals"), title="Preferred Switch Target"),
            tooltip=[
                "product",
                "sprint",
                "preferred_action",
                alt.Tooltip("confidence:Q", format=".2f"),
            ],
        )
        .properties(height=300)
    )

    labels = (
        alt.Chart(switch_df)
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(
            x=alt.X("product:N", sort=[f"Product {index}" for index in range(1, 8)]),
            y=alt.Y("sprint:N", sort=[f"Sprint {index}" for index in range(1, 5)]),
            text="label:N",
        )
    )

    st.altair_chart(heatmap + labels, use_container_width=True)


def render_probability_impact_plot(strategy_df):
    """Render the win-probability vs continue-value correlation scatter plot."""
    scatter = (
        alt.Chart(strategy_df)
        .mark_circle(size=120)
        .encode(
            x=alt.X("win_probability:Q", title="Win Probability"),
            y=alt.Y("q_continue:Q", title="Q-Value for Continue"),
            color=alt.Color("preferred_action:N", title="Preferred Action"),
            tooltip=[
                "product",
                "sprint",
                "preferred_action",
                alt.Tooltip("win_probability:Q", format=".3f"),
                alt.Tooltip("q_continue:Q", format=".2f"),
                alt.Tooltip("best_q_value:Q", format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(scatter, use_container_width=True)


def render_action_frequency_chart(training_log):
    """Render a line chart showing how often each action appears in the logs."""
    action_columns = [column for column in training_log.columns if column.startswith("action_") and column.endswith("_count")]
    if not action_columns:
        st.info("No per-action counts found in the training log yet.")
        return

    renamed = {}
    for column in action_columns:
        action_id = int(column.split("_")[1])
        renamed[column] = action_label(action_id)

    action_history = training_log[["episode"] + action_columns].rename(columns=renamed)
    melted = action_history.melt(id_vars="episode", var_name="Action", value_name="Count")

    chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("Count:Q", title="Action Count per 100 Episodes"),
            color=alt.Color("Action:N"),
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_invalid_action_chart(training_log):
    """Render invalid action counts over time."""
    if "invalid_action_count" not in training_log.columns:
        st.info("No invalid action counts found in the training log yet.")
        return

    chart = (
        alt.Chart(training_log)
        .mark_line(color="#ff6b6b")
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("invalid_action_count:Q", title="Invalid Actions per 100 Episodes"),
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def run_live_demo(agent, seed=42):
    """Play one greedy rollout with the selected DDQN checkpoint."""
    env = ScrumGameEnv()
    state = env.reset(seed=seed)
    state_vector = encode_state(state, env)
    done = False
    steps = []
    total_reward = 0
    turn_number = 1

    while not done:
        q_values = agent.predict_q_values(state_vector)
        action = max(range(env.num_actions), key=lambda action_id: q_values[action_id])
        next_state, reward, done, info = env.step(action)
        next_state_vector = encode_state(next_state, env)

        steps.append(
            {
                "Turn": turn_number,
                "Product": f"Product {state['current_product']}",
                "Sprint": state["current_sprint"],
                "Win Probability": round(state["win_probability"], 3),
                "Expected Value": round(state["expected_value"], 2),
                "Action": action_label(action),
                "Outcome": info["result"],
                "Reward": reward,
                "Bank": next_state["current_money"],
            }
        )

        total_reward += reward
        state = next_state
        state_vector = next_state_vector
        turn_number += 1

    return pd.DataFrame(steps), total_reward


def find_best_demo_seed(agent, search_count=20):
    """Search a small range of seeds and return the strongest demo rollout."""
    best_seed = 0
    best_reward = float("-inf")

    for seed in range(search_count):
        _, total_reward = run_live_demo(agent, seed=seed)
        if total_reward > best_reward:
            best_reward = total_reward
            best_seed = seed

    return best_seed, best_reward


def render_server_controls():
    """Render sidebar controls for starting and stopping background training."""
    st.sidebar.header("Server Controls")
    active_process = get_training_process_state()

    if active_process:
        elapsed = format_elapsed_time(time.time() - active_process.get("started_at_epoch", time.time()))
        st.sidebar.success(f"Training in Progress (PID: {active_process['pid']})")
        st.sidebar.caption(f"Elapsed: {elapsed}")
    else:
        st.sidebar.info("No background training job is currently running.")

    launch_disabled = active_process is not None
    if st.sidebar.button("🚀 Launch 500k Episode Training", use_container_width=True, disabled=launch_disabled):
        try:
            success, message = launch_training_process()
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.warning(message)
            st.rerun()
        except Exception as error:
            st.sidebar.error(f"Could not launch training: {error}")

    if st.sidebar.button("🛑 Stop Training", use_container_width=True, disabled=active_process is None):
        success, message = stop_training_process()
        if success:
            st.sidebar.warning(message)
        else:
            st.sidebar.error(message)
        st.rerun()

    if TRAINING_STDOUT_PATH.exists():
        st.sidebar.caption(f"Trainer log: {TRAINING_STDOUT_PATH.name}")

    return active_process


ensure_runtime_dirs()
st.set_page_config(page_title="Scrum Game DDQN Dashboard", layout="wide")
st.title("Scrum Game Command Center")
st.caption("Live monitoring for the advanced 8-action Double DQN branch.")

active_process = render_server_controls()

st.sidebar.header("Dashboard")
checkpoint_options = list_checkpoints()
checkpoint_labels = [checkpoint.name for checkpoint in checkpoint_options]
selected_checkpoint_label = st.sidebar.selectbox(
    "Checkpoint",
    checkpoint_labels if checkpoint_labels else ["best_scrum_model.pth"],
    index=0,
)
selected_checkpoint_path = (
    checkpoint_options[checkpoint_labels.index(selected_checkpoint_label)]
    if checkpoint_options
    else BEST_CHECKPOINT_PATH
)

auto_refresh = st.sidebar.toggle("🔄 Auto-Refresh", value=active_process is not None)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", min_value=2, max_value=30, value=5, step=1)
if st.sidebar.button("Refresh Charts", use_container_width=True):
    st.rerun()

demo_seed = st.sidebar.number_input("Demo seed", min_value=0, value=42, step=1)
scan_demo_seeds = st.sidebar.checkbox("Find best demo seed from range", value=False)
demo_search_count = st.sidebar.number_input("Demo seed scan size", min_value=5, value=20, step=5)
reference_money = st.sidebar.number_input("Heatmap reference money", min_value=0, value=25000, step=5000)

training_log = load_training_log()
evaluation_log = load_evaluation_log()
agent, checkpoint_error = load_dqn_policy(selected_checkpoint_path)

if training_log is None or training_log.empty:
    st.warning(
        "No training log found yet. Launch training from the sidebar or run `py train_dqn.py` "
        "to start logging to `artifacts/reports/logs.csv`."
    )
else:
    latest = training_log.iloc[-1]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Latest Episode", int(latest["episode"]))
    metric_col2.metric("Latest Reward", f"{latest['episode_reward']:.2f}")
    metric_col3.metric("Rolling Avg Reward", f"{latest['rolling_average_reward']:.2f}")
    metric_col4.metric("Epsilon", f"{latest['epsilon']:.4f}")

    debt_col1, debt_col2, debt_col3 = st.columns(3)
    debt_col1.metric("Average Loan Duration", f"{latest['average_loan_duration']:.2f}")
    debt_col2.metric("Bankruptcy Count / 100", int(latest["bankruptcy_count"]))
    debt_col3.metric("Average Ending Money", f"{latest['average_ending_money']:.2f}")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Training Reward")
        reward_chart = (
            alt.Chart(training_log)
            .transform_fold(["episode_reward", "rolling_average_reward"], as_=["metric", "value"])
            .mark_line()
            .encode(
                x=alt.X("episode:Q", title="Episode"),
                y=alt.Y("value:Q", title="Reward"),
                color=alt.Color("metric:N", title="Metric"),
            )
            .properties(height=320)
        )
        st.altair_chart(reward_chart, use_container_width=True)

    with chart_col2:
        st.subheader("Debt Stress")
        debt_chart = (
            alt.Chart(training_log)
            .transform_fold(
                ["average_loan_duration", "bankruptcy_count", "average_ending_money"],
                as_=["metric", "value"],
            )
            .mark_line()
            .encode(
                x=alt.X("episode:Q", title="Episode"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("metric:N", title="Metric"),
            )
            .properties(height=320)
        )
        st.altair_chart(debt_chart, use_container_width=True)

    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        st.subheader("Recent Loss")
        loss_chart = (
            alt.Chart(training_log)
            .mark_line()
            .encode(
                x=alt.X("episode:Q", title="Episode"),
                y=alt.Y("mean_recent_loss:Q", title="Recent Mean Loss"),
            )
            .properties(height=260)
        )
        st.altair_chart(loss_chart, use_container_width=True)

    with diag_col2:
        st.subheader("Replay Buffer Size")
        buffer_chart = (
            alt.Chart(training_log)
            .mark_line()
            .encode(
                x=alt.X("episode:Q", title="Episode"),
                y=alt.Y("replay_buffer_size:Q", title="Replay Buffer Size"),
            )
            .properties(height=260)
        )
        st.altair_chart(buffer_chart, use_container_width=True)

    st.subheader("Per-Action Frequency")
    render_action_frequency_chart(training_log)

    st.subheader("Invalid Actions")
    render_invalid_action_chart(training_log)

if evaluation_log is not None and not evaluation_log.empty:
    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        st.subheader("Evaluation Reward And Ending Money")
        reward_chart = (
            alt.Chart(evaluation_log)
            .transform_fold(["average_reward", "average_ending_money"], as_=["metric", "value"])
            .mark_line()
            .encode(
                x=alt.X("episode:Q", title="Checkpoint Episode"),
                y=alt.Y("value:Q", title="Reward / Ending Money"),
                color=alt.Color("metric:N", title="Metric"),
            )
            .properties(height=280)
        )
        st.altair_chart(reward_chart, use_container_width=True)

    with eval_col2:
        st.subheader("Evaluation Risk Metrics")
        risk_columns = [
            column
            for column in ["bankruptcy_rate", "invalid_action_rate", "average_loan_duration"]
            if column in evaluation_log.columns
        ]
        if risk_columns:
            risk_chart = (
                alt.Chart(evaluation_log)
                .transform_fold(risk_columns, as_=["metric", "value"])
                .mark_line()
                .encode(
                    x=alt.X("episode:Q", title="Checkpoint Episode"),
                    y=alt.Y("value:Q", title="Risk Metric"),
                    color=alt.Color("metric:N", title="Metric"),
                )
                .properties(height=280)
            )
            st.altair_chart(risk_chart, use_container_width=True)
        else:
            st.info("No evaluation risk metrics found yet.")

if checkpoint_error is not None:
    st.error(
        "The selected checkpoint is incompatible with the latest network shape. "
        "Use a new checkpoint from the advanced 8-action DDQN training run."
    )
elif agent is None:
    st.info("No compatible DDQN checkpoint found yet. Strategy maps and live demo will appear after training saves one.")
else:
    strategy_df = build_strategy_map(agent, current_money=int(reference_money))

    st.subheader("Strategy Heatmap")
    render_strategy_heatmap(strategy_df)

    st.subheader("Switch Target Heatmap")
    render_switch_target_heatmap(strategy_df)

    st.subheader("Probability vs Continue Q-Value")
    render_probability_impact_plot(strategy_df)

    st.subheader("Live Demo")
    if scan_demo_seeds:
        best_seed, best_seed_reward = find_best_demo_seed(agent, search_count=int(demo_search_count))
        st.caption(f"Best seed in 0..{int(demo_search_count) - 1}: {best_seed} (reward {best_seed_reward:.2f})")
        demo_seed = best_seed

    demo_table, total_demo_reward = run_live_demo(agent, seed=int(demo_seed))
    st.metric("Demo Total Reward", f"{total_demo_reward:.2f}")
    st.dataframe(demo_table, use_container_width=True, hide_index=True)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
