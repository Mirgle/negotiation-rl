from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Your base negotiation environment
from environment.environment import NegotiationEnv
from environment.environment_wrapper import NegotiationGymEnv
from agents.baseline_agents import RandomAgent

base_env = NegotiationEnv(n_rounds=20, seed=42)
opponent = RandomAgent(name="B")
env = NegotiationGymEnv(base_env, opponent, actor="A")

check_env(env)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, n_steps=64, batch_size=32, learning_rate=1e-3)
model.learn(total_timesteps=50000)
model.save("ppo_negotiation_agentA")
