import numpy as np
import random

class GreedyAgent:
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