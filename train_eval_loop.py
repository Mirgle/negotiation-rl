import csv
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environment.negotiation_env_round import NegotiationEnvRound
from environment.gym_wrapper_round import NegotiationGymRoundEnv
from agents.baseline_agents import GreedyAgent, RandomAgent
from agents.PPOOpponent import PPOOpponent

# Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_negotiation_agentA.zip")
CSV_PATH = os.path.join(BASE_DIR, "data", "ppo_negotiation_learning_curve.csv")
SNAPSHOT_PATH = os.path.join(BASE_DIR, "models", "opponent_snap.zip")
BENCHMARK_MODEL = os.path.join(BASE_DIR, "models", "model_250k.zip")

# Config
N_ROUNDS = 20
SELFPLAY_INTERVAL = 50_000
EVAL_INTERVAL = 25_000
TOTAL_TIMESTEPS = 1_000_000
TRAIN_CHUNK = 10_000
EVAL_EPISODES = 10

# ---------------------------
#  Eval Helper
# ---------------------------
def evaluate_against_opponent(agent_model, opponent, base_env, n_rounds, n_episodes=10):
    """
    Runs evaluation episodes against a fixed opponent and returns mean total reward.
    """

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


# ---------------------------
#  Environment setup
# ---------------------------
base_env = NegotiationEnvRound(n_rounds=N_ROUNDS, seed=42)
opponent = RandomAgent(name="B", n_items=base_env.n_items, seed=1337)

env = NegotiationGymRoundEnv(base_env, opponent, actor="A")

check_env(env)  # Ensure Gym interface is correct

# ---------------------------
#  Load or initialize model
# ---------------------------
if os.path.exists(MODEL_PATH):
    print(f"Loading existing model from {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    model.set_env(env)
else:
    print("Creating new PPO model")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        tensorboard_log="./ppo_negotiation_tensorboard/"
    )
    
# ---------------------------
#  Initialize eval opponents
# ---------------------------
# Load saved snapshot model
benchmark_model = PPO.load(BENCHMARK_MODEL)

eval_opponents = {
    "random": RandomAgent(name="B", n_items=base_env.n_items, seed=999),
    "greedy": GreedyAgent(name="B", n_items=base_env.n_items, seed=777),
    "250k_model": PPOOpponent(
        benchmark_model, 
        name="B",
        n_items=base_env.n_items
    ),
}

EVAL_FILES = {}
for name in eval_opponents:
    path = f"./data/eval_{name}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=["timesteps", "mean_reward"])
    writer.writeheader()
    EVAL_FILES[name] = (f, writer)

# ---------------------------
#  Training + Evaluation Loop
# ---------------------------
timesteps_done = 0
while timesteps_done < TOTAL_TIMESTEPS:

    # ---- Train ----
    model.learn(total_timesteps=TRAIN_CHUNK)
    timesteps_done += TRAIN_CHUNK

    # -------- SELF-PLAY OPPONENT UPDATE --------
    if timesteps_done % SELFPLAY_INTERVAL == 0:
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

    # -------- EVALUATE AGAINST BASELINES --------
    if timesteps_done % EVAL_INTERVAL == 0:
        print(f"[Evaluation] Running baseline eval at {timesteps_done} timesteps...")

        for name, opp in eval_opponents.items():

            mean_reward = evaluate_against_opponent(
                model,
                opp,
                base_env,
                n_rounds=N_ROUNDS,
                n_episodes=10
            )
            
            print(f"    vs {name}: mean_reward={mean_reward:.3f}")

            # Save to CSV
            f, writer = EVAL_FILES[name]
            writer.writerow({"timesteps": timesteps_done, "mean_reward": mean_reward})
            f.flush()


# ---------------------------
#  Save final model
# ---------------------------
model.save(MODEL_PATH)
print("Training complete — model saved.")