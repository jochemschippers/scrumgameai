"""
DQN Demonstration Game Player.

This module is a CLI demo runner. It loads a specified Double DQN checkpoint,
initializes the ScrumGameEnv, and plays one complete game step-by-step under
a chosen difficulty profile, printing a detailed play-by-play history to the console.

Purpose:
  - Demonstrates the agent's decision-making process in real time (e.g. showing why it switches
    or continues products).
  - Validates that a saved checkpoint successfully runs without crashes.

Connections:
  - Direct Entrypoint: Run via command `py -m play.play_best_dqn_game`
  - Utilizes: `rl.checkpoint_utils.load_agent_from_checkpoint`
  - Utilizes: `play.deployment_profiles.choose_profile_action`
"""

import argparse
from pathlib import Path

from config.config_manager import load_game_config
from play.deployment_profiles import choose_profile_action
from rl.checkpoint_utils import load_agent_from_checkpoint
from rl.dqn_agent import encode_state

# The root directory of the v2_deep_rl package
BASE_DIR = Path(__file__).resolve().parents[1]


def load_dqn_checkpoint(
    model_path: str = "artifacts/checkpoints/best_scrum_model.pth",
    game_config_path: str | None = None,
) -> tuple:
    """
    Load a saved model checkpoint and return the restored agent and environment.
    
    Args:
        model_path: Path to the `.pth` file containing the weights.
        game_config_path: Path to the game rules config. If None, uses default template.
        
    Returns:
        tuple: (DQNAgent, ScrumGameEnv, dict) containing model, env, and metadata.
    """
    checkpoint_path = Path(model_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = BASE_DIR / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        # Load weights and resolve configuration structures
        agent, env, metadata = load_agent_from_checkpoint(
            checkpoint_path,
            game_config=load_game_config(game_config_path) if game_config_path else None,
            strict_signature=game_config_path is not None,
        )
    except Exception as error:
        raise RuntimeError(
            "The selected checkpoint is incompatible with the requested Scrum Game config. "
            "Train or load a model for the same ruleset."
        ) from error
    return agent, env, metadata


def product_name(env, product_id: int) -> str:
    """Helper to convert a numeric product ID to its readable string name from the environment."""
    if 1 <= product_id <= len(env.product_names):
        return env.product_names[product_id - 1]
    return f"Product {product_id}"


def play_demo_game(
    model_path: str = "artifacts/checkpoints/best_scrum_model.pth",
    seed: int = 42,
    profile_name: str = "expert",
    game_config_path: str | None = None,
):
    """
    Load a checkpoint, seed the environment, and print a play-by-play game simulation.
    """
    agent, env, metadata = load_dqn_checkpoint(
        model_path=model_path,
        game_config_path=game_config_path,
    )
    checkpoint_path = Path(model_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = BASE_DIR / checkpoint_path

    # Initialize environment and observation state vector
    state = env.reset(seed=seed)
    state_vector = encode_state(state, env)
    done = False
    turn_number = 1

    print("Final DQN Demo Game")
    print(f"Model: {checkpoint_path}")
    print(f"Rule Signature: {metadata.get('current_rule_signature')}")
    print(f"Profile: {profile_name}")
    print("")

    # Main game execution loop
    while not done:
        current_product = state["current_product"]
        # Choose action according to selected difficulty profile (e.g. expert = greedy)
        action = choose_profile_action(agent, state_vector, profile_name=profile_name)
        next_state, reward, done, info = env.step(action)

        # 1. Print action taken and basic turn info
        print(
            f"Turn {turn_number}: {product_name(env, current_product)} - "
            f"Action: {info['action_name']}"
        )
        print(
            f"Outcome: {info['result']} - "
            f"Current Bank: {next_state['current_money']}"
        )
        print(
            f"Win Probability: {state['win_probability']:.3f} - Reward: {reward}"
        )
        print(f"Expected Value: {state['expected_value']:.2f}")

        # 2. Print scrum roll totals if turn resolved a sprint
        if "net_result" in info:
            print(f"Net Scrum Result: {info['net_result']}")

        # 3. Print loan alerts if a loan was forced during the turn
        if info.get("loan_triggered"):
            print(
                f"Mandatory Loan Triggered: Yes - "
                f"Interest Due Next Turn: {next_state['interest_due']}"
            )

        # 4. Print refinement grooming outcomes if applied
        if info.get("refinement_roll") is not None:
            print(
                f"Refinement: roll {info['refinement_roll']} - "
                f"{info['refinement_effect']}"
            )
            changed_sprints = info.get("refinement_future_sprints_changed", [])
            if changed_sprints:
                print(f"Refinement affected future sprints: {changed_sprints}")

        # 5. Print incident cards if drawn
        if info.get("incident_triggered"):
            print(
                f"Incident Card: {info['incident_card_id']} - "
                f"{info['incident_card_name']}"
            )
            if info.get("incident_card_description"):
                print(f"Incident Effect: {info['incident_card_description']}")

        # 6. Check game over triggers
        if done and "terminal_reason" in info:
            print(f"Game Over: {info['terminal_reason']}")

        print("")

        # Advance state
        state = next_state
        state_vector = encode_state(state, env)
        turn_number += 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Play one demo game with a saved DDQN checkpoint.")
    parser.add_argument("--model-path", default="artifacts/checkpoints/best_scrum_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default="expert")
    parser.add_argument("--game-config", default=None)
    args = parser.parse_args()

    play_demo_game(
        model_path=args.model_path,
        seed=args.seed,
        profile_name=args.profile,
        game_config_path=args.game_config,
    )


if __name__ == "__main__":
    main()

