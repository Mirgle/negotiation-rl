import numpy as np
import random

class RandomAgent:
    """Baseline: acts randomly, sometimes accepts, sometimes proposes random trade."""

    def __init__(self, name="A", n_items=3, accept_prob=0.3, seed=None):
        self.name = name
        self.n_items = n_items
        self.accept_prob = accept_prob
        self.rng = np.random.RandomState(seed)

    # Decides on an action based on observation and last offer
    # It cannot both accept and propose in the same turn, it can either accept or propose
    def act(self, obs, last_offer):
        # Randomly choose whether to accept the last offer
        if last_offer is not None and self.rng.rand() < self.accept_prob:
            return {"type": "accept"}
        
        # Randomly propose a trade
        offer = self.rng.randint(-1, 2, size=self.n_items)  # allowed values: {-1, 0, 1}
        if np.all(offer == 0):
            offer[random.randrange(self.n_items)] = random.choice([-1, 1])
        return {"type": "propose", "offer": offer}
