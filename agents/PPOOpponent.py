import numpy as np

class PPOOpponent:
    """
    Wraps a PPO model so it can serve as an opponent.agent.
    It uses the same observation flattening logic as the Gym wrapper.
    """
    def __init__(self, model, name, n_items):
        self.model = model
        self.name = name           # "A" or "B"
        self.n_items = n_items

    def act(self, obs, last_offer):
        # Build opponent's flattened observation
        st = obs[self.name]
        inv = np.array(st["inventory"], dtype=np.float32)
        val = np.array(st["self_value"], dtype=np.float32)
        off = (
            np.array(last_offer, dtype=np.float32)
            if last_offer is not None
            else np.zeros(self.n_items, dtype=np.float32)
        )
        flat_obs = np.concatenate([inv, val, off])

        # Get action from PPO model
        action_idx, _ = self.model.predict(flat_obs, deterministic=True)

        # Decode into your action dictionary
        if action_idx == 0:
            return {"type": "accept"}
        elif action_idx == 1:
            return {"type": "pass"}
        else:
            offer = self._decode_offer(action_idx - 2)
            return {"type": "propose", "offer": offer}

    def _decode_offer(self, idx):
        base3 = np.base_repr(idx, base=3).zfill(self.n_items)
        return np.array([int(c) - 1 for c in base3], dtype=int)
