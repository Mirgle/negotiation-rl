import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NegotiationGymEnv(gym.Env):
    """
    Gym wrapper for your NegotiationEnv to train a single RL agent with PPO.
    """
    def __init__(self, base_env, opponent, actor="A"):
        super().__init__()
        self.env = base_env
        self.actor = actor
        self.opponent = opponent
        self.n_items = self.env.n_items

        # Action space: each item delta is -1, 0, 1 → map to [0,1,2] for MultiDiscrete
        self.action_space = spaces.MultiDiscrete([3]*self.n_items)

        # Observation: inventory + values + last_offer (flattened)
        # Inventory: 0..10, Values: 0..1, Last offer: -1..1
        self.observation_space = spaces.Box(
            low=-1,
            high=10, 
            shape=(self.n_items*3,),
            dtype=np.float32
        )
        
        # Action space
        # 0: accept, 1: pass, 2.. : propose offers
        self.n_propose_actions = 3 ** self.n_items
        self.action_space = spaces.Discrete(2 + self.n_propose_actions)
        self.last_offer = None

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.last_offer = None        
        return self._flatten_obs(obs), info

    def step(self, action_idx):
        # Map action_idx to dict
        if action_idx == 0:
            action_dict = {"type": "accept"}
        elif action_idx == 1:
            action_dict = {"type": "pass"}
        else:
            offer_idx = action_idx - 2
            offer_vector = self._decode_offer(offer_idx)
            action_dict = {"type": "propose", "offer": offer_vector}

        # Step agent
        obs, reward, done, info = self.env.step(self.actor, action_dict)
        
        # Reward if offer proposed would increase utility
        if action_dict["type"] == "propose":
            current_utility = self.env._utility(self.actor)
            hypothetical_inventory = np.array(obs[self.actor]["inventory"]) + np.array(action_dict["offer"])
            if np.all(hypothetical_inventory >= 0):
                hypothetical_utility = np.dot(hypothetical_inventory, obs[self.actor]["self_value"])
                if hypothetical_utility > current_utility:
                    reward[self.actor] += 0.2  # small bonus for beneficial proposal

        # Step opponent
        opponent_action = self.opponent.act(obs, self.last_offer)
        obs, reward_op, done, info = self.env.step(
            "B" if self.actor == "A" else "A", opponent_action
        )
        
        reward[self.actor] += reward_op[self.actor]

        self.last_offer = opponent_action if opponent_action["type"] == "propose" else None
        truncated = False
        return self._flatten_obs(obs), reward[self.actor], done, truncated, info

    def _flatten_obs(self, obs):
        state = obs[self.actor]
        inventory = np.array(state["inventory"], dtype=np.float32)
        values = np.array(state["self_value"], dtype=np.float32)
        last_offer = (
            np.array(self.last_offer["offer"], dtype=np.float32) if self.last_offer else np.zeros(self.n_items)
        )
        return np.concatenate([inventory, values, last_offer]).astype(np.float32)
    
    def _decode_offer(self, idx):
        # Convert 0..3^n_items-1 to vector of -1,0,1
        base3 = np.base_repr(idx, base=3).zfill(self.n_items)
        return np.array([int(c)-1 for c in base3], dtype=int)
