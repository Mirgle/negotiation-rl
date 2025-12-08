# gym_wrapper_round.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NegotiationGymRoundEnv(gym.Env):

    def __init__(self, base_env, opponent, actor="A"):
        super().__init__()
        self.env = base_env        # instance of NegotiationEnvRound
        self.opponent = opponent   # baseline scripted agent
        self.actor = actor         # RL agent name ("A" or "B")
        self.n_items = base_env.n_items

        # Discrete action: 0=accept, 1=pass, 2.. = propose offer
        self.action_space = spaces.Discrete(2 + 3**self.n_items)

        # Observation: inventory + values + last_offer
        self.observation_space = spaces.Box(
            shape=(self.n_items * 3,),
            low=-1, high=10,
            dtype=np.float32
        )

    def reset(self, seed=None):
        obs, info = self.env.reset(seed)
        return self._flatten_obs(obs), info

    def step(self, action_idx):

        # Decode RL agent action
        if action_idx == 0:
            agent_act = {"type": "accept"}
        elif action_idx == 1:
            agent_act = {"type": "pass"}
        else:
            offer = self._decode_offer(action_idx - 2)
            agent_act = {"type": "propose", "offer": offer}

        # Get opponent action
        obs = self.env._get_obs()
        opp_act = self.opponent.act(obs, self.env.last_offer)

        # One full round happens inside env.step()
        obs, reward, done, info = self.env.step(self.actor, agent_act, opp_act)

        return self._flatten_obs(obs), reward, done, False, info

    # -------------------------
    #  Helpers
    # -------------------------
    def _flatten_obs(self, obs):
        st = obs[self.actor]
        inv = np.array(st["inventory"], dtype=np.float32)
        val = np.array(st["self_value"], dtype=np.float32)
        off = (
            np.array(st["last_offer"], dtype=np.float32)
            if st["last_offer"] is not None
            else np.zeros(self.n_items, dtype=np.float32)
        )
        return np.concatenate([inv, val, off])

    def _decode_offer(self, idx):
        base3 = np.base_repr(idx, base=3).zfill(self.n_items)
        return np.array([int(c) - 1 for c in base3], dtype=int)
