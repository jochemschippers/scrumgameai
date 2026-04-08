import argparse
import csv
from datetime import datetime
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from checkpoint_utils import build_agent_for_config, load_agent_from_checkpoint, save_checkpoint
from config_manager import (
    GameConfig,
    TrainingConfig,
    compute_rule_signature,
    compute_training_signature,
    load_game_config,
    load_training_config,
    save_game_config,
    save_training_config,
)
from dqn_agent import encode_state
from model_utils import save_metrics_json
from scrum_game_env import ScrumGameEnv


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"


def ensure_deep_rl_directories():
    """Create the deep-RL artifact folders used by training."""
    checkpoint_dir = BASE_DIR / "artifacts" / "checkpoints"
    plot_dir = BASE_DIR / "artifacts" / "plots"
    report_dir = BASE_DIR / "artifacts" / "reports"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, plot_dir, report_dir


def create_timestamped_run_directory():
    """Create a unique timestamped run directory under artifacts/runs."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = datetime.now().strftime("run_%Y-%m-%d_%H%M")
    candidate = RUNS_DIR / base_name
    suffix = 1
    while candidate.exists():
        candidate = RUNS_DIR / f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def resolve_output_directories(run_dir=None):
    """Resolve either the legacy flat artifact directories or a timestamped run folder."""
    if run_dir is None:
        checkpoint_dir, plot_dir, report_dir = ensure_deep_rl_directories()
        return checkpoint_dir, plot_dir, report_dir, None, ARTIFACTS_DIR

    run_path = Path(run_dir)
    if not run_path.is_absolute():
        run_path = BASE_DIR / run_path

    checkpoint_dir = run_path / "checkpoints"
    plot_dir = run_path / "plots"
    report_dir = run_path / "reports"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, plot_dir, report_dir, run_path, run_path


def initialize_training_log(log_path, num_actions):
    """Create or overwrite the live DQN training log CSV."""
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "epsilon",
                "episode_reward",
                "rolling_average_reward",
                "mean_recent_loss",
                "replay_buffer_size",
                "average_loan_duration",
                "bankruptcy_count",
                "average_ending_money",
                "invalid_action_count",
            ]
            + [f"action_{action_id}_count" for action_id in range(num_actions)]
        )


def initialize_evaluation_log(log_path):
    """Create or overwrite the periodic evaluation CSV."""
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "average_reward",
                "bankruptcy_rate",
                "average_ending_money",
                "average_loan_duration",
                "invalid_action_rate",
            ]
        )


def append_training_log(
    log_path,
    episode,
    epsilon,
    episode_reward,
    rolling_average_reward,
    mean_recent_loss,
    replay_buffer_size,
    average_loan_duration,
    bankruptcy_count,
    average_ending_money,
    invalid_action_count,
    action_counts,
):
    """Append one summarized training record for the dashboard."""
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                episode,
                epsilon,
                episode_reward,
                rolling_average_reward,
                mean_recent_loss,
                replay_buffer_size,
                average_loan_duration,
                bankruptcy_count,
                average_ending_money,
                invalid_action_count,
            ]
            + list(action_counts)
        )


def append_evaluation_log(log_path, episode, metrics):
    """Append one periodic evaluation record."""
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                episode,
                metrics["average_reward"],
                metrics["bankruptcy_rate"],
                metrics["average_ending_money"],
                metrics["average_loan_duration"],
                metrics["invalid_action_rate"],
            ]
        )


def rolling_average(values, window_size=500):
    """Compute a rolling mean for the DDQN learning curve."""
    if not values:
        return []

    smoothed = []
    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        window = values[start_index : index + 1]
        smoothed.append(sum(window) / len(window))
    return smoothed


def epsilon_by_episode(
    episode,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay_episodes=450000,
):
    """
    Linearly decay epsilon very slowly across the first 450,000 episodes.

    The 8-action branch needs more exploration than the earlier binary-action setup.
    """
    if episode >= epsilon_decay_episodes:
        return epsilon_min

    progress = episode / float(epsilon_decay_episodes)
    return epsilon_start - (epsilon_start - epsilon_min) * progress


def resolve_training_config(
    training_config: TrainingConfig | None = None,
    training_config_path=None,
    num_episodes=None,
    learning_rate=None,
    gamma=None,
    checkpoint_interval=None,
    evaluation_interval=None,
    evaluation_episodes=None,
    seed=None,
    run_notes="",
):
    """Load the base training config and apply CLI/function overrides."""
    base_config = training_config or load_training_config(training_config_path)
    payload = base_config.to_dict()

    if num_episodes is not None:
        payload["episodes"] = int(num_episodes)
    if learning_rate is not None:
        payload["learning_rate"] = float(learning_rate)
    if gamma is not None:
        payload["gamma"] = float(gamma)
    if checkpoint_interval is not None:
        payload["checkpoint_interval"] = int(checkpoint_interval)
    if evaluation_interval is not None:
        payload["evaluation_interval"] = int(evaluation_interval)
    if evaluation_episodes is not None:
        payload["evaluation_episodes"] = int(evaluation_episodes)
    if seed is not None:
        payload["seed"] = int(seed)
    if run_notes:
        payload["run_notes"] = str(run_notes)

    return TrainingConfig.from_dict(payload)


def initialize_agent_from_checkpoint(
    agent,
    checkpoint_path,
    target_game_config: GameConfig,
    resume_mode: str,
):
    """Warm-start an agent from an existing checkpoint under the requested resume mode."""
    if not checkpoint_path:
        return None

    if resume_mode not in {"strict", "fine-tune"}:
        raise ValueError("resume_mode must be either 'strict' or 'fine-tune'.")

    strict_signature = resume_mode == "strict"
    try:
        _, _, checkpoint_metadata = load_agent_from_checkpoint(
            checkpoint_path,
            game_config=target_game_config,
            strict_signature=strict_signature,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"Could not resume from checkpoint `{checkpoint_path}` under mode `{resume_mode}`: {error}"
        ) from error

    payload = torch.load(checkpoint_path, map_location=agent.device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    try:
        agent.policy_network.load_state_dict(state_dict)
        agent.target_network.load_state_dict(state_dict)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint tensor shapes do not match the requested game config. "
            "Resume/fine-tune only works when the model architecture is still compatible."
        ) from error

    agent.policy_network.train()
    agent.target_network.eval()
    return {
        "resume_checkpoint_path": str(checkpoint_path),
        "resume_mode": resume_mode,
        "resume_checkpoint_rule_signature": (
            checkpoint_metadata.get("checkpoint_rule_signature")
            or checkpoint_metadata.get("rule_signature")
        ),
        "resume_target_rule_signature": compute_rule_signature(target_game_config),
        "resume_legacy_checkpoint": checkpoint_metadata.get("legacy_checkpoint", False),
    }


def evaluate_dqn_agent(agent, num_episodes=1000, seed=1042, game_config: GameConfig | None = None):
    """
    Evaluate the DDQN greedily with epsilon fixed at 0.

    Evaluation uses a separate environment instance and no optimization steps.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    env = ScrumGameEnv(game_config=game_config)
    evaluation_rewards = []
    ending_monies = []
    bankruptcy_count = 0
    loan_durations = []
    action_counts = [0] * env.num_actions
    invalid_action_count = 0

    for episode in range(num_episodes):
        state = env.reset(seed=seed + episode)
        state_vector = encode_state(state, env)
        done = False
        cumulative_reward = 0

        while not done:
            action = agent.choose_action(state_vector, epsilon=0.0)
            action_counts[action] += 1
            next_state, reward, done, info = env.step(action)
            next_state_vector = encode_state(next_state, env)
            if info.get("invalid_action"):
                invalid_action_count += 1

            cumulative_reward += reward
            state_vector = next_state_vector

            if done and info.get("terminal_reason") == "bankruptcy":
                bankruptcy_count += 1
                ending_monies.append(next_state["current_money"])
                loan_durations.append(env.turns_with_loan)
            elif done:
                ending_monies.append(next_state["current_money"])
                loan_durations.append(env.turns_with_loan)

        evaluation_rewards.append(cumulative_reward)

    return {
        "rewards": evaluation_rewards,
        "average_reward": sum(evaluation_rewards) / len(evaluation_rewards),
        "average_ending_money": sum(ending_monies) / len(ending_monies),
        "bankruptcy_rate": bankruptcy_count / len(evaluation_rewards),
        "average_loan_duration": sum(loan_durations) / len(loan_durations),
        "invalid_action_rate": invalid_action_count / max(sum(action_counts), 1),
        "action_counts": action_counts,
    }


def save_training_plot(training_rewards, output_path):
    """Save the DDQN rolling-average training curve."""
    smoothed_rewards = rolling_average(training_rewards, window_size=500)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label="Rolling Average Reward (500 episodes)", linewidth=2)
    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.title("Double DQN Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_dqn_agent(
    num_episodes=500000,
    learning_rate=0.0005,
    gamma=0.85,
    checkpoint_interval=10000,
    evaluation_interval=10000,
    evaluation_episodes=100,
    seed=42,
    run_dir=None,
    run_notes="",
    game_config: GameConfig | None = None,
    game_config_path=None,
    training_config: TrainingConfig | None = None,
    training_config_path=None,
    resume_from=None,
    resume_mode="strict",
):
    """Train a Double DQN agent on the advanced Scrum Game environment."""
    resolved_game_config = game_config or load_game_config(game_config_path)
    resolved_training_config = resolve_training_config(
        training_config=training_config,
        training_config_path=training_config_path,
        num_episodes=num_episodes,
        learning_rate=learning_rate,
        gamma=gamma,
        checkpoint_interval=checkpoint_interval,
        evaluation_interval=evaluation_interval,
        evaluation_episodes=evaluation_episodes,
        seed=seed,
        run_notes=run_notes,
    )

    random.seed(resolved_training_config.seed)
    torch.manual_seed(resolved_training_config.seed)

    checkpoint_dir, plot_dir, report_dir, run_path, config_root = resolve_output_directories(run_dir)

    env = ScrumGameEnv(game_config=resolved_game_config)
    initial_state = env.reset(seed=resolved_training_config.seed)
    state_dim = len(encode_state(initial_state, env))
    num_actions = env.num_actions

    log_path = report_dir / "logs.csv"
    evaluation_log_path = report_dir / "evaluation_history.csv"
    initialize_training_log(log_path, num_actions)
    initialize_evaluation_log(evaluation_log_path)

    game_config_output_path = config_root / "game_config.json"
    training_config_output_path = config_root / "training_config.json"
    save_game_config(resolved_game_config, game_config_output_path)
    save_training_config(resolved_training_config, training_config_output_path)

    run_metadata = {
        "run_name": run_path.name if run_path is not None else "current_artifacts",
        "run_path": str(run_path) if run_path is not None else str(config_root),
        "created_at": datetime.now().isoformat(),
        "training_episodes": resolved_training_config.episodes,
        "learning_rate": resolved_training_config.learning_rate,
        "gamma": resolved_training_config.gamma,
        "checkpoint_interval": resolved_training_config.checkpoint_interval,
        "evaluation_interval": resolved_training_config.evaluation_interval,
        "evaluation_episodes": resolved_training_config.evaluation_episodes,
        "seed": resolved_training_config.seed,
        "run_notes": resolved_training_config.run_notes,
        "game_config_path": str(game_config_output_path),
        "training_config_path": str(training_config_output_path),
        "rule_signature": compute_rule_signature(resolved_game_config),
        "training_signature": compute_training_signature(resolved_training_config),
        "resume_checkpoint_path": str(resume_from) if resume_from else None,
        "resume_mode": resume_mode if resume_from else None,
    }

    if run_path is not None:
        save_metrics_json(run_metadata, str(run_path / "run_metadata.json"))
    else:
        save_metrics_json(run_metadata, str(report_dir / "run_metadata.json"))

    agent, _ = build_agent_for_config(
        resolved_game_config,
        learning_rate=resolved_training_config.learning_rate,
        gamma=resolved_training_config.gamma,
        replay_capacity=resolved_training_config.replay_capacity,
        batch_size=resolved_training_config.batch_size,
        target_update_frequency=resolved_training_config.target_update_frequency,
    )
    resume_metadata = initialize_agent_from_checkpoint(
        agent,
        resume_from,
        resolved_game_config,
        resume_mode,
    )
    if resume_metadata:
        run_metadata.update(resume_metadata)
        if run_path is not None:
            save_metrics_json(run_metadata, str(run_path / "run_metadata.json"))
        else:
            save_metrics_json(run_metadata, str(report_dir / "run_metadata.json"))

    training_rewards = []
    training_losses = []
    episode_loan_durations = []
    bankruptcy_flags = []
    ending_monies = []
    recent_action_counts = []
    invalid_action_flags = []

    best_average_reward = float("-inf")
    best_checkpoint_path = checkpoint_dir / "best_scrum_model.pth"

    for episode in range(1, resolved_training_config.episodes + 1):
        state = env.reset(seed=resolved_training_config.seed + episode)
        state_vector = encode_state(state, env)
        done = False
        cumulative_reward = 0
        bankruptcy_this_episode = 0
        invalid_actions_this_episode = 0
        episode_action_counts = [0] * num_actions

        epsilon = epsilon_by_episode(
            episode - 1,
            epsilon_start=resolved_training_config.epsilon_start,
            epsilon_min=resolved_training_config.epsilon_min,
            epsilon_decay_episodes=resolved_training_config.epsilon_decay_episodes,
        )

        while not done:
            action = agent.choose_action(state_vector, epsilon=epsilon)
            episode_action_counts[action] += 1

            next_state, reward, done, info = env.step(action)
            next_state_vector = encode_state(next_state, env)
            if info.get("invalid_action"):
                invalid_actions_this_episode += 1

            agent.store_transition(state_vector, action, reward, next_state_vector, done)
            loss = agent.train_step()

            if loss is not None:
                training_losses.append(loss)

            cumulative_reward += reward
            state_vector = next_state_vector

            if done and info.get("terminal_reason") == "bankruptcy":
                bankruptcy_this_episode = 1

        training_rewards.append(cumulative_reward)
        episode_loan_durations.append(env.turns_with_loan)
        bankruptcy_flags.append(bankruptcy_this_episode)
        ending_monies.append(env.current_money)
        recent_action_counts.append(episode_action_counts)
        invalid_action_flags.append(invalid_actions_this_episode)

        if episode % 100 == 0:
            recent_rewards = training_rewards[-100:]
            recent_losses = training_losses[-100:] if training_losses else []
            recent_loan_durations = episode_loan_durations[-100:]
            recent_bankruptcies = bankruptcy_flags[-100:]
            recent_ending_monies = ending_monies[-100:]
            recent_invalid_actions = invalid_action_flags[-100:]
            block_action_counts = [0] * num_actions
            for action_counts in recent_action_counts[-100:]:
                for action_id, count in enumerate(action_counts):
                    block_action_counts[action_id] += count

            append_training_log(
                log_path=log_path,
                episode=episode,
                epsilon=epsilon,
                episode_reward=cumulative_reward,
                rolling_average_reward=sum(recent_rewards) / len(recent_rewards),
                mean_recent_loss=(sum(recent_losses) / len(recent_losses)) if recent_losses else 0.0,
                replay_buffer_size=len(agent.replay_buffer),
                average_loan_duration=(sum(recent_loan_durations) / len(recent_loan_durations)) if recent_loan_durations else 0.0,
                bankruptcy_count=sum(recent_bankruptcies),
                average_ending_money=(sum(recent_ending_monies) / len(recent_ending_monies)) if recent_ending_monies else 0.0,
                invalid_action_count=sum(recent_invalid_actions),
                action_counts=block_action_counts,
            )

        if episode % resolved_training_config.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode:06d}.pth"
            save_checkpoint(
                checkpoint_path,
                agent,
                resolved_game_config,
                resolved_training_config,
                extra_metadata={
                    "episode": episode,
                    **(resume_metadata or {}),
                },
            )
            print(f"Checkpoint saved at episode {episode}: {checkpoint_path}")

        if episode % resolved_training_config.evaluation_interval == 0:
            evaluation_metrics = evaluate_dqn_agent(
                agent,
                num_episodes=resolved_training_config.evaluation_episodes,
                seed=resolved_training_config.seed + 100000 + episode,
                game_config=resolved_game_config,
            )
            append_evaluation_log(evaluation_log_path, episode, evaluation_metrics)

            if evaluation_metrics["average_reward"] > best_average_reward:
                best_average_reward = evaluation_metrics["average_reward"]
                save_checkpoint(
                    best_checkpoint_path,
                    agent,
                    resolved_game_config,
                    resolved_training_config,
                    extra_metadata={
                        "episode": episode,
                        "average_reward": best_average_reward,
                        "is_best_checkpoint": True,
                        **(resume_metadata or {}),
                    },
                )
                print(
                    f"Updated best model at episode {episode}: {best_checkpoint_path} "
                    f"(avg reward {best_average_reward:.2f})"
                )

    if not best_checkpoint_path.exists():
        save_checkpoint(
            best_checkpoint_path,
            agent,
            resolved_game_config,
            resolved_training_config,
            extra_metadata={
                "episode": resolved_training_config.episodes,
                "is_best_checkpoint": True,
                **(resume_metadata or {}),
            },
        )

    plot_path = plot_dir / "dqn_training_curve.png"
    save_training_plot(training_rewards, output_path=plot_path)

    best_agent, _, _ = load_agent_from_checkpoint(
        best_checkpoint_path,
        game_config=resolved_game_config,
    )

    final_evaluation = evaluate_dqn_agent(
        best_agent,
        num_episodes=1000,
        seed=resolved_training_config.seed + 1000,
        game_config=resolved_game_config,
    )

    save_metrics_json(
        {
            "model": "Double DQN",
            "training_episodes": resolved_training_config.episodes,
            "evaluation_episodes": len(final_evaluation["rewards"]),
            "average_reward_per_episode": final_evaluation["average_reward"],
            "average_ending_money": final_evaluation["average_ending_money"],
            "bankruptcy_rate": final_evaluation["bankruptcy_rate"],
            "average_loan_duration": final_evaluation["average_loan_duration"],
            "invalid_action_rate": final_evaluation["invalid_action_rate"],
            "learning_rate": resolved_training_config.learning_rate,
            "gamma": resolved_training_config.gamma,
            "state_dim": state_dim,
            "num_actions": num_actions,
            "checkpoint_path": str(best_checkpoint_path),
            "plot_path": str(plot_path),
            "log_path": str(log_path),
            "evaluation_log_path": str(evaluation_log_path),
            "game_config_path": str(game_config_output_path),
            "training_config_path": str(training_config_output_path),
            "rule_signature": compute_rule_signature(resolved_game_config),
            "training_signature": compute_training_signature(resolved_training_config),
            "resume_checkpoint_path": str(resume_from) if resume_from else None,
            "resume_mode": resume_mode if resume_from else None,
            "resume_checkpoint_rule_signature": (resume_metadata or {}).get("resume_checkpoint_rule_signature"),
            "resume_target_rule_signature": (resume_metadata or {}).get("resume_target_rule_signature"),
            "resume_legacy_checkpoint": (resume_metadata or {}).get("resume_legacy_checkpoint"),
            "final_epsilon": epsilon_by_episode(
                resolved_training_config.episodes - 1,
                epsilon_start=resolved_training_config.epsilon_start,
                epsilon_min=resolved_training_config.epsilon_min,
                epsilon_decay_episodes=resolved_training_config.epsilon_decay_episodes,
            ),
            "mean_training_reward": sum(training_rewards) / len(training_rewards),
            "mean_training_loss": (sum(training_losses) / len(training_losses)) if training_losses else None,
            "best_intermediate_evaluation_reward": best_average_reward,
            "run_notes": resolved_training_config.run_notes,
            "run_path": str(run_path) if run_path is not None else None,
        },
        str(report_dir / "dqn_metrics.json"),
    )

    return best_agent, training_rewards, final_evaluation, best_checkpoint_path, plot_path, log_path, evaluation_log_path


def parse_args():
    """Parse CLI options for dashboard-driven and manual runs."""
    parser = argparse.ArgumentParser(description="Train the advanced Double DQN Scrum Game agent.")
    parser.add_argument("--run-dir", default=None, help="Optional timestamped run directory path.")
    parser.add_argument("--game-config", default=None, help="Optional path to a game_config.json file.")
    parser.add_argument("--training-config", default=None, help="Optional path to a training_config.json file.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of training episodes.")
    parser.add_argument("--evaluation-episodes", type=int, default=None, help="Episodes per periodic checkpoint evaluation.")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Checkpoint save interval.")
    parser.add_argument("--evaluation-interval", type=int, default=None, help="Periodic evaluation interval.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optimizer learning rate.")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor.")
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint path for resume or fine-tune training.")
    parser.add_argument(
        "--resume-mode",
        choices=["strict", "fine-tune"],
        default="strict",
        help="Use 'strict' for same-rule resume or 'fine-tune' to allow rule-signature mismatch when shapes match.",
    )
    parser.add_argument("--notes", default="", help="Optional run notes saved with the artifacts.")
    return parser.parse_args()


def main():
    """Train the advanced Double DQN agent with the requested production hyperparameters."""
    args = parse_args()
    _, training_rewards, final_evaluation, checkpoint_path, plot_path, log_path, evaluation_log_path = train_dqn_agent(
        num_episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        checkpoint_interval=args.checkpoint_interval,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        seed=args.seed,
        run_dir=args.run_dir,
        run_notes=args.notes,
        game_config_path=args.game_config,
        training_config_path=args.training_config,
        resume_from=args.resume_from,
        resume_mode=args.resume_mode,
    )

    print(f"Training episodes completed: {len(training_rewards)}")
    print(f"Evaluation episodes completed: {len(final_evaluation['rewards'])}")
    print(f"Average reward per episode during evaluation: {final_evaluation['average_reward']:.2f}")
    print(f"Average ending money during evaluation: {final_evaluation['average_ending_money']:.2f}")
    print(f"Bankruptcy rate during evaluation: {final_evaluation['bankruptcy_rate']:.3f}")
    print(f"Saved best model checkpoint to: {checkpoint_path}")
    print(f"Saved training curve to: {plot_path}")
    print(f"Saved live training log to: {log_path}")
    print(f"Saved periodic evaluation log to: {evaluation_log_path}")


if __name__ == "__main__":
    main()
