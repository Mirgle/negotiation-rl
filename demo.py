import argparse
import numpy as np
from stable_baselines3 import PPO

from environment.negotiation_env_round import NegotiationEnvRound
from environment.gym_wrapper_round import NegotiationGymRoundEnv
from agents.baseline_agents import RandomAgent, GreedyAgent
from agents.PPOOpponent import PPOOpponent


# Formatting functions
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


def print_state(env):
    A = env.agent_states["A"]
    B = env.agent_states["B"]

    utilA = env._utility("A")
    utilB = env._utility("B")

    print("\nCurrent Inventories:")
    print(f"  A inv = {A['inventory']} | util = \033[94;1m{utilA:.3f}\033[0m")
    print(f"  B inv = {B['inventory']} | util = \033[94;1m{utilB:.3f}\033[0m")
    print("-----------------------------------------------------")


def run_demo(model_path, opponent_type="greedy", n_rounds=10, n_items=3):

    print("\n=== NEGOTIATION DEMO ===")
    print(f"Model: {model_path}")
    print(f"Opponent: {opponent_type}")
    print(f"Rounds: {n_rounds}")
    print("========================\n")

    # ---- Load model ----
    model = PPO.load(model_path)

    # ---- Select opponent ----
    if opponent_type == "random":
        opponent = RandomAgent("B", n_items=n_items, seed=123)

    elif opponent_type == "greedy":
        opponent = GreedyAgent("B", n_items=n_items, seed=123)

    elif opponent_type.endswith(".zip"):
        opp_model = PPO.load(opponent_type)
        opponent = PPOOpponent(opp_model, name="B", n_items=n_items)

    else:
        raise ValueError(f"Invalid opponent: {opponent_type}")

    # ---- Create environment ----
    base_env = NegotiationEnvRound(n_items=n_items, n_rounds=n_rounds, seed=0)
    env = NegotiationGymRoundEnv(base_env, opponent, actor="A")

    obs, info = env.reset()
    print_state(base_env)

    # ---- Demo loop ----
    for round_i in range(n_rounds):
        print(f"\n\033[95;1m----- ROUND {round_i+1} -----\033[0m")

        # =============================
        # Agent A acts
        # =============================
        action_A, _ = model.predict(obs, deterministic=True)

        # Interpret model action
        if action_A == 0:
            A_str = "\033[96;1mACCEPT\033[0m"
        elif action_A == 1:
            A_str = "\033[90mPASS\033[0m"
        else:
            idx = action_A - 2
            offerA = env._decode_offer(idx)
            A_str = f"PROPOSE {pretty_offer(offerA)}"

        print(f"A → {A_str}")

        # Step environment (A’s action)
        obs, reward_A, terminated, truncated, info = env.step(action_A)
        done = terminated or truncated

        # Capture offer after A’s action
        offer_after_A = base_env.last_offer

        # =============================
        # Opponent (B) acts (automatically in env.step())
        # =============================
        # The opponent action is stored in env.last_offer only if B proposed
        offer_after_B = base_env.last_offer

        # To expose opponent behavior, we can *compute* what they did:
        if offer_after_B is None and offer_after_A is not None:
            # B ACCEPTED A’s offer
            B_str = "\033[96mACCEPT\033[0m"
        elif offer_after_B is None and offer_after_A is None:
            B_str = "\033[90mPASS\033[0m"
        else:
            B_str = f"PROPOSE {pretty_offer(offer_after_B)}"

        print(f"B → {B_str}")

        print(f"Reward to A this round: \033[92m{reward_A:.3f}\033[0m")

        # Show active offer on table
        print(f"Current offer on table: {pretty_offer(base_env.last_offer)}")

        print_state(base_env)

        if done:
            print("\n\033[92;1mEpisode finished.\033[0m\n")
            break

    print("\n=== DEMO COMPLETE ===\n")


# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive negotiation demo for PPO models.")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO model (.zip)")
    parser.add_argument("--opponent", type=str, default="greedy",
                        help='Opponent type: "random", "greedy", or path to PPO model (.zip)')
    parser.add_argument("--rounds", type=int, default=15, help="Number of rounds to simulate")
    parser.add_argument("--items", type=int, default=3, help="Number of items in negotiation")

    args = parser.parse_args()

    run_demo(
        model_path=args.model,
        opponent_type=args.opponent,
        n_rounds=args.rounds,
        n_items=args.items
    )
