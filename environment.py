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
    def reset(self):
        self.round = 0
        self.agent_states = {
            "agent_1": {"inventory": self.rng.randint(0,3,size=self.n_items),
                  "self_value": self.rng.rand(self.n_items)},
            "agent_2": {"inventory": self.rng.randint(0,3,size=self.n_items),
                  "self_value": self.rng.rand(self.n_items)}
        }
        self.last_offer = None
        return self._obs()

    # Gets the observation state for current agent (their inventory, valuation, and last offer)
    def _obs(self):
        return self.agent_states

    # Processes a single agent's turn in negotiation
    def step(self, actor: str, action: Dict[str, Any]):
        self.round += 1
        done = self.round >= self.n_rounds
        reward = {a: self._utility(a) for a in self.agent_states}
        return self._obs(), reward, done, {}
    
    # Rewards the agent based on their utility function
    def _utility(self, agent):
        self_values = self.agent_states[agent]['self_value']
        inventory = self.agent_states[agent]['inventory']
        return float(np.dot(self_values, inventory))    # Dot product is sum of values times quantities
