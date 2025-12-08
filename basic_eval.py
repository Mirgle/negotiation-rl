import numpy as np
from stable_baselines3 import PPO

# Import your environment and baseline agent classes
from environment.environment import NegotiationEnv
from agents.baseline_agents import RandomAgent, GreedyAgent
from environment.environment_wrapper import NegotiationGymEnv

# -----------------------
# Configuration
# -----------------------
NUM_EPISODES = 20
OPPONENT = GreedyAgent(name="B")  # baseline opponent
AGENT_PATH = "ppo_negotiation_agentA.zip"

# -----------------------
# Initialize environment
# -----------------------
base_env = NegotiationEnv(n_rounds=20, seed=1234)
env = NegotiationGymEnv(base_env, OPPONENT, actor="A")

# -----------------------
# Load trained agent
# -----------------------
model = PPO.load(AGENT_PATH)
model.set_env(env)  # ensure environment is set
rewards = [0.0 for _ in range(NUM_EPISODES)]

# -----------------------
# Run sanity check
# -----------------------
for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    print(f"\n=== Episode {ep+1} ===")

    while not done:       
        # Get action from trained agent
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)

        # Print info checking
        print("Action proposed:", action)
        print("Reward received:", reward)
        print("Agent inventories:", {a: base_env.agent_states[a]["inventory"] for a in base_env.agent_states})
        print("Agent values:", {a: base_env.agent_states[a]["self_value"] for a in base_env.agent_states})
        print("Last offer:", base_env.last_offer)
        print("Done:", done)
        print("---")

        total_reward += reward if np.isscalar(reward) else reward[0]  # handle array rewards

    print(f"Episode {ep+1} total reward:", total_reward)
    rewards[ep] = total_reward

print("\n=== Summary ===")
print("Average reward:", np.mean(rewards))
for ep in range(NUM_EPISODES):
    print(f"Episode {ep+1} reward: {rewards[ep]}")