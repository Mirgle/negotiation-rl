import csv
import os, sys
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from environment.negotiation_env_round import NegotiationEnvRound
from environment.gym_wrapper_round import NegotiationGymRoundEnv
from agents.baseline_agents import GreedyAgent, RandomAgent
from agents.PPOOpponent import PPOOpponent

# Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "ppo_negotiation_learning_curve.csv")
SNAPSHOT_PATH = os.path.join(BASE_DIR, "models", "opponent_snap.zip")

# Non-optional Config
N_ROUNDS = 20
TRAIN_CHUNK = 10_000
EVAL_EPISODES = 10

# Helper to evaluate against an opponent
def evaluate(agent_model, opponent, base_env, n_episodes=10):
    temp_env = NegotiationGymRoundEnv(base_env, opponent, actor="A")

    total_rewards = []
    for _ in range(n_episodes):
        obs, info = temp_env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = agent_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = temp_env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
        total_rewards.append(ep_reward)

    return float(np.mean(total_rewards))


# Initilize parser with variables
parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int, default=1_000_000)
parser.add_argument("--save-path", type=str, default="models/ppo_negotiation_agentA.zip")
parser.add_argument("--selfplay-interval", type=int, default=50_000)
parser.add_argument("--eval-interval", type=int, default=25_000)
parser.add_argument("--record-csv", type=bool, default=False)
args = parser.parse_args()

def main():
    # Setup environment
    base_env = NegotiationEnvRound(n_rounds=N_ROUNDS, seed=42)
    opponent = RandomAgent(name="B", n_items=base_env.n_items, seed=1337)

    env = NegotiationGymRoundEnv(base_env, opponent, actor="A")

    check_env(env)  # Ensure Gym interface is correct

    # Load model or initialize
    if os.path.exists(args.save_path):
        print(f"Loading existing model from {args.save_path}")
        model = PPO.load(args.save_path)
        model.set_env(env)
    else:
        print(f"Creating new PPO model: {args.save_path}")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4,
            tensorboard_log="./ppo_negotiation_tensorboard/"
        )
        

    eval_opponents = {
        "random": RandomAgent(name="B", n_items=base_env.n_items, seed=999),
        "greedy": GreedyAgent(name="B", n_items=base_env.n_items, seed=777)
    }

    # Init csv eval files if recording
    if (args.record_csv):
        EVAL_FILES = {}
        for name in eval_opponents:
            path = f"./data/eval_{name}.csv"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            f = open(path, "w", newline="")
            writer = csv.DictWriter(f, fieldnames=["timesteps", "mean_reward"])
            writer.writeheader()
            EVAL_FILES[name] = (f, writer)

    # Train Eval loop
    timesteps_done = 0
    while timesteps_done < args.timesteps:

        # train
        model.learn(total_timesteps=TRAIN_CHUNK)
        timesteps_done += TRAIN_CHUNK

        # Self-play update opponent
        if timesteps_done % args.selfplay_interval == 0:
            print(f"[Self-Play] Updating opponent at {timesteps_done} timesteps...")

            # Save snapshot
            model.save(SNAPSHOT_PATH)
            opponent_model = PPO.load(SNAPSHOT_PATH)

            # Replace opponent in training env
            env.opponent = PPOOpponent(
                opponent_model,
                name="B",
                n_items=base_env.n_items
            )

        # Evaluate against baseline
        if timesteps_done % args.eval_interval == 0:
            print(f"[Evaluation] Running baseline eval at {timesteps_done} timesteps...")

            for name, opp in eval_opponents.items():

                mean_reward = evaluate(
                    model,
                    opp,
                    base_env,
                    n_episodes=10
                )
                
                print(f"    vs {name}: mean_reward={mean_reward:.3f}")

                # Save to CSV if recording
                if args.record_csv:
                    f, writer = EVAL_FILES[name]
                    writer.writerow({"timesteps": timesteps_done, "mean_reward": mean_reward})
                    f.flush()


    # Save final model
    model.save(args.save_path)
    print("Training complete — model saved.")
    
if __name__ == "__main__":
    main()