import argparse
import os
import numpy as np
from stable_baselines3 import PPO

from environment.negotiation_env_round import NegotiationEnvRound
from environment.gym_wrapper_round import NegotiationGymRoundEnv
from agents.baseline_agents import RandomAgent, GreedyAgent
from agents.PPOOpponent import PPOOpponent


def evaluate_model(model, opponent, base_env, n_episodes=10):
    """
    Evaluate a PPO model against a fixed opponent.
    Returns mean reward and a list of per-episode totals.
    """
    env = NegotiationGymRoundEnv(base_env, opponent, actor="A")
    rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward

        rewards.append(total)

    return float(np.mean(rewards)), rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PPO negotiation agent.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model (.zip)")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "greedy", "snapshot", "ppo"],
                        help="Which opponent to evaluate against.")
    parser.add_argument("--opponent-model", type=str, default=None,
                        help="If --opponent=ppo, path to opponent PPO checkpoint.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    args = parser.parse_args()

    # Load trained model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = PPO.load(args.model)

    # Base environment for evaluation
    base_env = NegotiationEnvRound(n_rounds=20, seed=999)

    # Select opponent policy
    if args.opponent == "random":
        opponent = RandomAgent("B", base_env.n_items, seed=123)

    elif args.opponent == "greedy":
        opponent = GreedyAgent("B", base_env.n_items, seed=456)

    elif args.opponent == "snapshot":
        snap_path = "models/opponent_snap.zip"
        if not os.path.exists(snap_path):
            raise FileNotFoundError("Snapshot opponent not found. Run train.py to generate opponent_snap.zip.")
        opponent_model = PPO.load(snap_path)
        opponent = PPOOpponent(opponent_model, "B", base_env.n_items)

    elif args.opponent == "ppo":
        if not args.opponent_model:
            raise ValueError("Specify --opponent-model=<path> when using --opponent=ppo.")
        opponent_model = PPO.load(args.opponent_model)
        opponent = PPOOpponent(opponent_model, "B", base_env.n_items)

    # Run evaluation
    mean_reward, per_episode = evaluate_model(model, opponent, base_env, args.episodes)

    # Print formatted results
    print("\n================ Evaluation Complete ================")
    print(f"Evaluating model: {args.model}")
    print(f"Against opponent: {args.opponent}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.3f}\n")

    for i, r in enumerate(per_episode):
        print(f"  Episode {i+1}: reward = {r:.3f}")

    print("====================================================\n")


if __name__ == "__main__":
    main()
