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
        # Inventory: 0..max_items, Values: 0..1, Last offer: -1..1 mapped to 0..2
        self.observation_space = spaces.Box(
            low=0,
            high=10,  # adjust max inventory if needed
            shape=(self.n_items*2 + self.n_items,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.last_offer = None        
        return self._flatten_obs(obs), info

    def step(self, action):
        # Convert MultiDiscrete [0,1,2] → [-1,0,1]
        offer = np.array(action) - 1
        action_dict = {"type": "propose", "offer": offer}

        obs, reward, done, info = self.env.step(self.actor, action_dict)

        # Step opponent (fixed policy)
        opponent_action = self.opponent.act(obs, self.last_offer)
        obs, reward_op, done, info = self.env.step(
            "B" if self.actor == "A" else "A", opponent_action
        )

        self.last_offer = opponent_action if opponent_action["type"] == "propose" else None

        truncated = False  # Not handling truncation in this wrapper
        return self._flatten_obs(obs), reward[self.actor], done, truncated, info

    def _flatten_obs(self, obs):
        state = obs[self.actor]
        inventory = np.array(state["inventory"], dtype=np.float32)
        values = np.array(state["self_value"], dtype=np.float32)
        last_offer = (
            np.array(self.last_offer["offer"], dtype=np.float32) if self.last_offer else np.zeros(self.n_items)
        )
        return np.concatenate([inventory, values, last_offer]).astype(np.float32)
