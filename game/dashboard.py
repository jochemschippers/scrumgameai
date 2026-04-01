from pathlib import Path
import time

import pandas as pd
import streamlit as st
import torch

from dqn_agent import DQNAgent, encode_state
from scrum_game_env import ScrumGameEnv


LOG_PATH = Path("artifacts/deep_rl/reports/logs.csv")
METRICS_PATH = Path("artifacts/deep_rl/reports/dqn_metrics.json")
CHECKPOINT_PATH = Path("artifacts/deep_rl/checkpoints/best_scrum_model.pth")


def load_training_log():
    """Load the live training CSV if it exists."""
    if not LOG_PATH.exists():
        return None
    return pd.read_csv(LOG_PATH)


def load_dqn_policy():
    """Load the latest DQN checkpoint into a greedy agent."""
    if not CHECKPOINT_PATH.exists():
        return None

    agent = DQNAgent(
        state_dim=8,
        num_actions=2,
        learning_rate=0.0005,
        gamma=0.85,
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location=agent.device)
    agent.policy_network.load_state_dict(state_dict)
    agent.target_network.load_state_dict(state_dict)
    agent.policy_network.eval()
    agent.target_network.eval()
    return agent


def run_live_demo(seed=42):
    """Play one greedy rollout with the current DQN checkpoint."""
    agent = load_dqn_policy()
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
st.title("Scrum Game Deep RL Dashboard")
st.caption("Live monitoring for DQN training and greedy demo playback.")

auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 seconds", value=True)
demo_seed = st.sidebar.number_input("Demo seed", min_value=0, value=42, step=1)

training_log = load_training_log()

if training_log is None or training_log.empty:
    st.warning("No training log found yet. Run `py train_dqn.py` to start logging to `artifacts/deep_rl/reports/logs.csv`.")
else:
    latest = training_log.iloc[-1]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Latest Episode", int(latest["episode"]))
    metric_col2.metric("Latest Reward", f"{latest['episode_reward']:.2f}")
    metric_col3.metric("Rolling Avg Reward", f"{latest['rolling_average_reward']:.2f}")
    metric_col4.metric("Epsilon", f"{latest['epsilon']:.4f}")

    st.subheader("Training Reward")
    st.line_chart(
        training_log.set_index("episode")[["episode_reward", "rolling_average_reward"]],
        height=320,
    )

    st.subheader("Training Diagnostics")
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        st.line_chart(training_log.set_index("episode")[["mean_recent_loss"]], height=260)
    with diag_col2:
        st.line_chart(training_log.set_index("episode")[["replay_buffer_size"]], height=260)

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
