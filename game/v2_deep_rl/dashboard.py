from pathlib import Path
import time

import altair as alt
import pandas as pd
import streamlit as st
import torch

from dqn_agent import DQNAgent, encode_state
from scrum_game_env import ScrumGameEnv


LOG_PATH = Path("artifacts/reports/logs.csv")
EVALUATION_LOG_PATH = Path("artifacts/reports/evaluation_history.csv")
CHECKPOINT_DIR = Path("artifacts/checkpoints")
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_scrum_model.pth"


def load_training_log():
    """Load the live training CSV if it exists."""
    if not LOG_PATH.exists():
        return None
    training_log = pd.read_csv(LOG_PATH)

    fallback_columns = {
        "average_loan_duration": 0.0,
        "bankruptcy_count": 0.0,
        "average_ending_money": 0.0,
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
            state = env.build_reference_state(product_id=product_id, sprint_id=sprint_id, current_money=current_money)
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
            color=alt.Color(
                "confidence:Q",
                scale=alt.Scale(scheme="yellowgreenblue"),
                title="Policy Confidence",
            ),
            tooltip=[
                "product",
                "sprint",
                "preferred_action",
                alt.Tooltip("best_q_value:Q", format=".2f"),
                alt.Tooltip("q_continue:Q", format=".2f"),
                alt.Tooltip("win_probability:Q", format=".3f"),
                alt.Tooltip("expected_value:Q", format=".2f"),
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
            color=alt.Color(
                "preferred_action_id:Q",
                scale=alt.Scale(scheme="teals"),
                title="Preferred Switch Target",
            ),
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


st.set_page_config(page_title="Scrum Game DDQN Dashboard", layout="wide")
st.title("Scrum Game Command Center")
st.caption("Live monitoring for the advanced 8-action Double DQN branch.")

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

auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 seconds", value=True)
demo_seed = st.sidebar.number_input("Demo seed", min_value=0, value=42, step=1)
scan_demo_seeds = st.sidebar.checkbox("Find best demo seed from range", value=False)
demo_search_count = st.sidebar.number_input("Demo seed scan size", min_value=5, value=20, step=5)
reference_money = st.sidebar.number_input("Heatmap reference money", min_value=0, value=25000, step=5000)

training_log = load_training_log()
evaluation_log = load_evaluation_log()
agent, checkpoint_error = load_dqn_policy(selected_checkpoint_path)

if training_log is None or training_log.empty:
    st.warning("No training log found yet. Run `py train_dqn.py` to start logging to `artifacts/reports/logs.csv`.")
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
            .transform_fold(
                ["episode_reward", "rolling_average_reward"],
                as_=["metric", "value"],
            )
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

if evaluation_log is not None and not evaluation_log.empty:
    st.subheader("Evaluation Over Time")
    evaluation_chart = (
        alt.Chart(evaluation_log)
        .transform_fold(
            ["average_reward", "average_ending_money", "bankruptcy_rate"],
            as_=["metric", "value"],
        )
        .mark_line()
        .encode(
            x=alt.X("episode:Q", title="Checkpoint Episode"),
            y=alt.Y("value:Q", title="Evaluation Metric"),
            color=alt.Color("metric:N", title="Metric"),
        )
        .properties(height=300)
    )
    st.altair_chart(evaluation_chart, use_container_width=True)

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
    time.sleep(5)
    st.rerun()
