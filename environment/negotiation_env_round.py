# negotiation_env_round.py
import numpy as np
from typing import Dict, Any, Tuple

class NegotiationEnvRound:
    """
    New environment: ONE env.step() = agent turn + opponent turn.
    Only after both act do we compute reward.
    """

    def __init__(self, n_items=3, n_rounds=5, seed=0):
        self.n_items = n_items
        self.n_rounds = n_rounds
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.round = 0
        self.agent_states = {
            "A": {
                "inventory": self._random_distribution(10),
                "self_value": self._normalize_values(self.rng.rand(self.n_items))
            },
            "B": {
                "inventory": self._random_distribution(10),
                "self_value": self._normalize_values(self.rng.rand(self.n_items))
            },
        }

        # Previous utilities for reward calculation
        self.prev_utility = {
            a: self._utility(a) for a in ["A", "B"]
        }

        self.last_offer = None

        obs = self._get_obs()
        info = {"round": self.round}
        return obs, info

    # -------------------------
    #  Core step (one full round)
    # -------------------------
    def step(
        self,
        acting_agent: str,
        agent_action: Dict[str, Any],
        opponent_action: Dict[str, Any]
    ) -> Tuple[Dict, float, bool, Dict]:

        """
        acting_agent = RL-controlled agent ("A" or "B")
        agent_action = action dict from RL agent
        opponent_action = action dict from scripted opponent
        """

        reward = 0.0

        # Apply RL agent action
        self._apply_action(acting_agent, agent_action)

        # Apply opponent action
        opponent = "B" if acting_agent == "A" else "A"
        self._apply_action(opponent, opponent_action)

        # Compute utility deltas
        for a in ["A", "B"]:
            u_now = self._utility(a)
            reward_part = u_now - self.prev_utility[a]

            if a == acting_agent:
                reward += reward_part  # Only RL agent receives reward

            self.prev_utility[a] = u_now

        # Advance round counter
        self.round += 1
        done = self.round >= self.n_rounds

        return self._get_obs(), reward, done, {"round": self.round}

    # -------------------------
    #  Helpers
    # -------------------------
    def _apply_action(self, agent: str, action: Dict[str, Any]):
        """
        Executes accept/propose/pass.
        Only accept modifies inventories.
        Propose sets last_offer (visible to both agents).
        """
        if action["type"] == "propose":
            self.last_offer = action["offer"]

        elif action["type"] == "accept" and self.last_offer is not None:
            proposer = "B" if agent == "A" else "A"
            self._execute_trade(proposer, agent, self.last_offer)
            self.last_offer = None

        elif action["type"] == "pass":
            pass

    def _get_obs(self):
        """Return the entire state for both agents"""
        return {
            "A": {
                "inventory": self.agent_states["A"]["inventory"].copy(),
                "self_value": self.agent_states["A"]["self_value"].copy(),
                "last_offer": (
                    self.last_offer.copy() if self.last_offer is not None else None
                )
            },
            "B": {
                "inventory": self.agent_states["B"]["inventory"].copy(),
                "self_value": self.agent_states["B"]["self_value"].copy(),
                "last_offer": (
                    self.last_offer.copy() if self.last_offer is not None else None
                )
            },
        }

    # -------------------------
    #  Utility & Trade Logic
    # -------------------------
    def _utility(self, agent: str) -> float:
        vals = self.agent_states[agent]['self_value']
        inv = self.agent_states[agent]['inventory']
        return float(np.dot(vals, inv))

    def _execute_trade(self, proposer: str, accepter: str, offer: np.ndarray):
        inv_prop = self.agent_states[proposer]["inventory"]
        inv_acc = self.agent_states[accepter]["inventory"]

        p_new = inv_prop - offer
        a_new = inv_acc + offer

        if np.any(p_new < 0) or np.any(a_new < 0):
            return  # invalid → ignore trade

        self.agent_states[proposer]["inventory"] = p_new
        self.agent_states[accepter]["inventory"] = a_new

    # -------------------------
    #  Misc helpers
    # -------------------------
    def _normalize_values(self, v):
        return v / v.sum()

    def _random_distribution(self, total):
        return self.rng.multinomial(total, [1/self.n_items]*self.n_items)
