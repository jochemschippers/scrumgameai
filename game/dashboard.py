from pathlib import Path
import time

import altair as alt
import pandas as pd
import streamlit as st
import torch

from dqn_agent import DQNAgent, encode_state
from scrum_game_env import ScrumGameEnv


LOG_PATH = Path("artifacts/deep_rl/reports/logs.csv")
CHECKPOINT_PATH = Path("artifacts/deep_rl/checkpoints/best_scrum_model.pth")


def load_training_log():
    """Load the live training CSV if it exists."""
    if not LOG_PATH.exists():
        return None
    training_log = pd.read_csv(LOG_PATH)

    # Support older log files until a fresh training run regenerates them.
    for column_name in ["average_loan_duration", "bankruptcy_count"]:
        if column_name not in training_log.columns:
            training_log[column_name] = 0.0

    return training_log


def load_dqn_policy():
    """Load the latest DQN checkpoint into a greedy policy."""
    if not CHECKPOINT_PATH.exists():
        return None, None

    agent = DQNAgent(
        state_dim=8,
        num_actions=2,
        learning_rate=0.0005,
        gamma=0.85,
    )
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=agent.device)
        agent.policy_network.load_state_dict(state_dict)
        agent.target_network.load_state_dict(state_dict)
        agent.policy_network.eval()
        agent.target_network.eval()
        return agent, None
    except RuntimeError as error:
        return None, str(error)


def standard_state_for_cell(env, product_id, sprint_id, current_money=25000, loan_active=False, interest_due=0):
    """Build a standard financial state for one board cell."""
    features_required = env.board_features[product_id - 1][sprint_id - 1]
    sprint_value = env.board_ring_values[product_id - 1][sprint_id - 1] * env.ring_value

    if features_required <= 1:
        win_probability = env.win_probability_lookup[1]
    elif features_required == 2:
        win_probability = env.win_probability_lookup[2]
    else:
        win_probability = env.win_probability_lookup[3]

    return (
        current_money,
        product_id,
        sprint_id,
        features_required,
        sprint_value,
        loan_active,
        interest_due,
        win_probability,
    )


def build_strategy_map(agent):
    """Create a heatmap-ready board dataframe from the current DQN policy."""
    env = ScrumGameEnv()
    cells = []

    for product_id in range(1, env.products_count + 1):
        for sprint_id in range(1, env.sprints_per_product + 1):
            state = standard_state_for_cell(env, product_id, sprint_id)
            state_vector = encode_state(state, env)
            state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=agent.device).unsqueeze(0)

            with torch.no_grad():
                q_values = agent.policy_network(state_tensor).squeeze(0).cpu().tolist()

            q_continue, q_switch = q_values
            preferred_action = "Continue" if q_continue >= q_switch else "Switch"
            confidence = q_continue - q_switch

            cells.append(
                {
                    "product": f"Product {product_id}",
                    "product_id": product_id,
                    "sprint": f"Sprint {sprint_id}",
                    "sprint_id": sprint_id,
                    "preferred_action": preferred_action,
                    "q_continue": q_continue,
                    "q_switch": q_switch,
                    "confidence": confidence,
                    "win_probability": state[7],
                    "label": "C" if preferred_action == "Continue" else "S",
                }
            )

    return pd.DataFrame(cells)


def render_strategy_heatmap(strategy_df):
    """Render the 7 x 4 board strategy heatmap."""
    heatmap = (
        alt.Chart(strategy_df)
        .mark_rect()
        .encode(
            x=alt.X("product:N", sort=[f"Product {index}" for index in range(1, 8)], title="Product"),
            y=alt.Y("sprint:N", sort=[f"Sprint {index}" for index in range(1, 5)], title="Sprint"),
            color=alt.Color(
                "confidence:Q",
                scale=alt.Scale(domainMid=0, scheme="redyellowgreen"),
                title="Continue vs Switch Confidence",
            ),
            tooltip=[
                "product",
                "sprint",
                "preferred_action",
                alt.Tooltip("q_continue:Q", format=".2f"),
                alt.Tooltip("q_switch:Q", format=".2f"),
                alt.Tooltip("win_probability:Q", format=".3f"),
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


def render_probability_impact_plot(strategy_df):
    """Render the win-probability vs Q-value correlation scatter plot."""
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
                alt.Tooltip("win_probability:Q", format=".3f"),
                alt.Tooltip("q_continue:Q", format=".2f"),
                alt.Tooltip("q_switch:Q", format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(scatter, use_container_width=True)


def run_live_demo(seed=42):
    """Play one greedy rollout with the current DQN checkpoint."""
    agent, load_error = load_dqn_policy()
    if agent is None:
        return None

    env = ScrumGameEnv()
    state = env.reset(seed=seed)
    state_vector = encode_state(state, env)
    done = False
    steps = []
    total_reward = 0
    turn_number = 1

    while not done:
        action = agent.choose_action(state_vector, epsilon=0.0)
        next_state, reward, done, info = env.step(action)
        next_state_vector = encode_state(next_state, env)

        steps.append(
            {
                "Turn": turn_number,
                "Product": f"Product {state[1]}",
                "Sprint": state[2],
                "Win Probability": round(state[7], 3),
                "Action": "Continue" if action == 0 else "Switch",
                "Outcome": "Success" if info.get("success") else ("Failure" if "success" in info else "Switch"),
                "Reward": reward,
                "Bank": next_state[0],
            }
        )

        total_reward += reward
        state = next_state
        state_vector = next_state_vector
        turn_number += 1

    return pd.DataFrame(steps), total_reward


st.set_page_config(page_title="Scrum Game DQN Dashboard", layout="wide")
st.title("Scrum Game Command Center")
st.caption("Live monitoring for DQN training, board strategy, debt stress, and greedy demo playback.")

auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 seconds", value=True)
demo_seed = st.sidebar.number_input("Demo seed", min_value=0, value=42, step=1)

training_log = load_training_log()
agent, checkpoint_error = load_dqn_policy()

if training_log is None or training_log.empty:
    st.warning("No training log found yet. Run `py train_dqn.py` to start logging to `artifacts/deep_rl/reports/logs.csv`.")
else:
    latest = training_log.iloc[-1]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Latest Episode", int(latest["episode"]))
    metric_col2.metric("Latest Reward", f"{latest['episode_reward']:.2f}")
    metric_col3.metric("Rolling Avg Reward", f"{latest['rolling_average_reward']:.2f}")
    metric_col4.metric("Epsilon", f"{latest['epsilon']:.4f}")

    debt_col1, debt_col2 = st.columns(2)
    debt_col1.metric("Average Loan Duration", f"{latest['average_loan_duration']:.2f}")
    debt_col2.metric("Bankruptcy Count / 100", int(latest["bankruptcy_count"]))

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
                ["average_loan_duration", "bankruptcy_count"],
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

if checkpoint_error is not None:
    st.error(
        "The current DQN checkpoint is incompatible with the latest network shape. "
        "This usually means the checkpoint was trained before the `win_probability` feature was added. "
        "Delete `artifacts/deep_rl/checkpoints/best_scrum_model.pth` and retrain with `py train_dqn.py`."
    )
elif agent is None:
    st.info("No DQN checkpoint found yet. Strategy heatmap, probability plots, and live demo will appear after the first checkpoint is saved.")
else:
    strategy_df = build_strategy_map(agent)

    st.subheader("Strategy Heatmap")
    render_strategy_heatmap(strategy_df)

    st.subheader("Probability vs Continue Q-Value")
    render_probability_impact_plot(strategy_df)

    st.subheader("Live Demo")
    demo_result = run_live_demo(seed=int(demo_seed))

    if demo_result is None:
        st.info("No checkpoint found yet. A demo will appear after the first model checkpoint is saved.")
    else:
        demo_table, total_demo_reward = demo_result
        st.metric("Demo Total Reward", f"{total_demo_reward:.2f}")
        st.dataframe(demo_table, use_container_width=True, hide_index=True)

if auto_refresh:
    time.sleep(5)
    st.rerun()
