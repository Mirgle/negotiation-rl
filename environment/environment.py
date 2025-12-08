# environment.py
import numpy as np
import random
from typing import Dict, Any

class NegotiationEnv:
    def __init__(self, n_items=3, n_rounds=5, seed=0):
        self.n_items = n_items
        self.n_rounds = n_rounds
        self.rng = np.random.RandomState(seed)

        self.reset()

    # Reset the environment to the initial state for new episode
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.round = 0
        self.agent_states = {
            "A": {"inventory": _random_distribution(self, 10, self.rng),
                  "self_value": _normalize_values(self.rng.rand(self.n_items))},   # Random positive values, normed
            "B": {"inventory": _random_distribution(self, 10, self.rng),
                  "self_value": _normalize_values(self.rng.rand(self.n_items))}
        }
        info = {
            "round": self.round,
            "n_items": self.n_items,
            "initial_inventory": {a: s["inventory"].copy() for a, s in self.agent_states.items()}
        }
        
        self.last_offer = None
        self.prev_utility = {a: self._utility(a) for a in self.agent_states}
        return self._obs(), info

    # Gets the observation state for current agent (their inventory, valuation, and last offer)
    def _obs(self):
        return self.agent_states

    def step(self, actor: str, action: Dict[str, Any]):
        # Store previous utilities
        reward = {a: 0.0 for a in self.agent_states}

        # Process the action
        if action["type"] == "propose":
            self.last_offer = action["offer"]

        elif action["type"] == "accept" and self.last_offer is not None:
            proposer = "B" if actor == "A" else "A"
            # Apply the trade
            #print(f"Trade executed between {proposer} and {actor} with offer {self.last_offer}")
            penalty = self._execute_trade(proposer=proposer, accepter=actor, offer=self.last_offer)
            # Penalize both agents if trade is invalid - disabled b/c agent cant see oppoent inventory
            #reward[proposer] -= penalty
            #reward[actor] -= penalty
            self.last_offer = None  # Reset last offer after acceptance

        # Advance round
        self.round += 1
        done = self.round >= self.n_rounds

        # Compute rewards as delta in utility
        for a in self.agent_states:
            current_utility = self._utility(a)
            reward[a] += current_utility - self.prev_utility[a]
            # Update previous utility for next step
            self.prev_utility[a] = current_utility

        # Prepare info dict
        info = {
            "round": self.round,
            "n_items": self.n_items,
            "initial_inventory": {a: s["inventory"].copy() for a, s in self.agent_states.items()}
        }
        
        #print(f"Round: {self.round}, Actor: {actor}, Reward: {reward}")
        return self._obs(), reward, done, info

    
    # Rewards the agent based on their utility function
    def _utility(self, agent):
        self_values = self.agent_states[agent]['self_value']
        inventory = self.agent_states[agent]['inventory']
        return float(np.dot(self_values, inventory))    # Dot product is sum of values times quantities
    
    # Execute the trade between proposer and accepter
    def _execute_trade(self, proposer: str, accepter: str, offer: np.ndarray):
        proposer_inventory = self.agent_states[proposer]["inventory"]
        accepter_inventory = self.agent_states[accepter]["inventory"]
        
        proposer_hypothetical = proposer_inventory - offer
        accepter_hypothetical = accepter_inventory + offer
        
        if np.all(proposer_hypothetical >= 0) and np.all(accepter_hypothetical >= 0):
            self.agent_states[proposer]["inventory"] = proposer_hypothetical
            self.agent_states[accepter]["inventory"] = accepter_hypothetical
            return 0  # Trade executed successfully
        else:
            #print("Invalid trade attempted; inventories unchanged.")
            return 1  # Penalty for invalid trade


def _normalize_values(values):
    return values / values.sum()

def _random_distribution(self, n_total_items, rng):
    if n_total_items <= 0:
        return np.zeros(self.n_items, dtype=int)

    counts = rng.multinomial(n_total_items, [1/self.n_items]*self.n_items)
    return counts

