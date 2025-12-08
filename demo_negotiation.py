import numpy as np
from stable_baselines3 import PPO

from environment.negotiation_env_round import NegotiationEnvRound
from environment.gym_wrapper_round import NegotiationGymRoundEnv

from agents.baseline_agents import RandomAgent, GreedyAgent
from agents.PPOOpponent import PPOOpponent


# Colored offer formatter
def pretty_offer(offer):
    if offer is None:
        return "None"

    def color(x):
        if x > 0:
            return f"\033[92m+{x}\033[0m"   # green
        elif x < 0:
            return f"\033[91m{x}\033[0m"   # red
        else:
            return f"\033[93m0\033[0m"     # yellow

    return "[" + ", ".join(color(int(x)) for x in offer) + "]"


# Colored utility + inventory formatter
def print_state(env):
    A = env.agent_states["A"]
    B = env.agent_states["B"]

    utilA = env._utility("A")
    utilB = env._utility("B")

    print("\nCurrent Inventories:")
    print(f"  A inv = {A['inventory']} | util = \033[94;1m{utilA:.3f}\033[0m")
    print(f"  B inv = {B['inventory']} | util = \033[94;1m{utilB:.3f}\033[0m")
    print("-----------------------------------------------------")


def run_demo(model_path, opponent_type="greedy", n_rounds=10):

    print("\n=== NEGOTIATION DEMO ===")
    print(f"Model: {model_path}")
    print(f"Opponent: {opponent_type}")
    print("========================\n")

    # ---- Load main agent model ----
    model = PPO.load(model_path)

    # ---- Choose opponent ----
    if opponent_type == "random":
        opponent = RandomAgent("B", n_items=3, seed=123)
    elif opponent_type == "greedy":
        opponent = GreedyAgent("B", n_items=3, seed=123)
    elif opponent_type.endswith(".zip"):
        opp_model = PPO.load(opponent_type)
        opponent = PPOOpponent(opp_model, name="B", n_items=3)
    else:
        raise ValueError("Invalid opponent type")

    # ---- Create environment ----
    base_env = NegotiationEnvRound(n_items=3, n_rounds=n_rounds, seed=0)
    env = NegotiationGymRoundEnv(base_env, opponent, actor="A")

    obs, info = env.reset()
    print_state(base_env)

    # ---- Demo loop ----
    for round_i in range(n_rounds):
        print(f"\n\033[95;1m----- ROUND {round_i+1} -----\033[0m")

        # Agent A action
        action_A, _ = model.predict(obs, deterministic=True)

        # Interpret action with color
        if action_A == 0:
            a_str = "\033[96;1mACCEPT\033[0m"
        elif action_A == 1:
            a_str = "\033[90mPASS\033[0m"
        else:
            offer_idx = action_A - 2
            offer = env._decode_offer(offer_idx)
            a_str = f"PROPOSE {pretty_offer(offer)}"

        print(f"A (model) action: {a_str}")

        # Step negotiation round
        obs, reward, done, info = env.step(action_A)

        print(f"Reward to A this round: \033[92m{reward:.3f}\033[0m")

        # Show current offer
        last_offer = base_env.last_offer
        print(f"Current offer (if any): {pretty_offer(last_offer)}")

        print_state(base_env)

        if done:
            print("\n\033[92;1mEpisode finished.\033[0m\n")
            break

    print("\n=== DEMO COMPLETE ===")


if __name__ == "__main__":
    run_demo(
        model_path="./models/ppo_negotiation_agentA.zip",
        opponent_type="greedy",
        n_rounds=15
    )
