from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Your base negotiation environment
from environment.environment import NegotiationEnv
from environment.environment_wrapper import NegotiationGymEnv
from agents.baseline_agents import GreedyAgent

base_env = NegotiationEnv(n_rounds=5, seed=42)
opponent = GreedyAgent(name="B")  # Fixed opponent
env = NegotiationGymEnv(base_env, opponent, actor="A")

# Optional: check your custom environment
check_env(env)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, n_steps=64, batch_size=32, learning_rate=1e-3)

# Train
model.learn(total_timesteps=50000)

# Save model
model.save("ppo_negotiation_agentA")
