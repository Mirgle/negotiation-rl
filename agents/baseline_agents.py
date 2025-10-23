import numpy as np
import random
from typing import Dict, Any, Optional

class Agent:
    """Base class for all agents in the negotiation environment."""

    def __init__(self, name: str, n_items: int, seed: int):
        self.name = name
        self.n_items = n_items
        self.rng = np.random.RandomState(seed)

    def act(self, obs: Dict[str, Any], last_offer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action based on the current observation and last offer.
        Should return one of the followng:
          {"type": "propose", "offer": [...]}
          {"type": "accept"}
          {"type": "pass"}
        """
        raise NotImplementedError()
        
    def get_net_value(self, obs: Dict[str, Any]) -> float:
        """Calculate and return the agent's net value based on it's own valuation."""
        state = obs[self.name]
        self_value = np.array(state["self_value"])
        inventory = np.array(state["inventory"])
        return float(np.dot(self_value, inventory))
    
    def print_state(self, obs: Dict[str, Any]):
        """Print the agent's current private state in a readable format."""
        if self.name not in obs:
            print(f"[{self.name}] Observation not found in obs dict.")
            return

        state = obs[self.name]
        self_value = np.round(state["self_value"], 2)
        inventory = state["inventory"]
        print(f"--- Agent {self.name} State ---")
        print(f"inventory: {inventory}")
        print(f"Values:   {self_value}")
        print(f"Utility:  {self.get_net_value(obs):.3f}")
        print("------------------------------")
    
    

    def __repr__(self):
        return f"Agent(name={self.name}, n_items={self.n_items})"

class RandomAgent(Agent):
    """Baseline: acts randomly, sometimes accepts, sometimes proposes random trade."""

    def __init__(self, name="A", n_items=3, accept_prob=0.3, seed=None):
        self.name = name
        self.n_items = n_items
        self.accept_prob = accept_prob
        self.rng = np.random.RandomState(seed)

    # Decides on an action based on observation and last offer
    # It cannot both accept and propose in the same turn, it can either accept or propose
    def act(self, obs, last_offer):
        my_inventory = np.array(obs[self.name]["inventory"])
        
        # Randomly choose whether to accept the last offer
        if last_offer is not None and self.rng.rand() < self.accept_prob:
            hypothetical = my_inventory + np.array(last_offer["offer"])
            if np.any(hypothetical >= 0):  # Ensure I don't accept an invalid offer
                return {"type": "accept"}
        
        # Randomly propose a trade
        offer = self.rng.randint(-1, 2, size=self.n_items)  # allowed values: {-1, 0, 1}
        hypothetical = my_inventory + np.array(offer)
        if np.any(hypothetical < 0):  # Ensure I don't propose an invalid offer
            return {"type": "pass"}
        
        return {"type": "propose", "offer": offer}

class GreedyAgent(Agent):
    """Baseline: proposes trades that strictly increase its own utility."""

    def __init__(self, name="A", n_items=3, seed=None):
        self.name = name
        self.n_items = n_items
        self.rng = np.random.RandomState(seed)

    def act(self, obs, last_offer):
        my_state = obs[self.name]
        my_vals = np.array(my_state["self_value"])
        my_inventory = np.array(my_state["inventory"])

        # Occasionally accept if the last offer increases my utility
        if last_offer is not None and last_offer.get("type") == "propose":
            hypothetical = my_inventory + np.array(last_offer["offer"])
            if np.dot(my_vals, hypothetical) > np.dot(my_vals, my_inventory):
                return {"type": "accept"}

        # Otherwise propose a new trade that helps me
        best_offer = np.zeros(self.n_items, dtype=int)
        best_gain = 0

        # Try a few random offers, until we find the best offer
        for _ in range(20):  
            offer = self.rng.randint(-1, 2, size=self.n_items)
            if np.all(offer == 0):
                continue
            new_inventory = my_inventory + offer
            if np.any(new_inventory < 0):  # Ensure I don't offer more than I have
                continue
            gain = np.dot(my_vals, new_inventory) - np.dot(my_vals, my_inventory)
            if gain > best_gain:
                best_gain = gain
                best_offer = offer

        if best_gain > 0:
            return {"type": "propose", "offer": best_offer}
        else:
            # No improving offer found, pass
            return {"type": "pass"}