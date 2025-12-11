# Negotiation-RL: Multi-Agent PPO for Automated Bargaining

This repository contains a custom negotiation environment and PPO-based training pipeline for learning emergent bargaining behavior through reinforcement learning and self-play.

## Project Overview
Agents negotiate over discrete item bundles with private valuations.  
Actions: **propose**, **accept**, **pass**.  
Rewards come from changes in utility after accepted trades.

The project investigates:
- Whether PPO can learn stable negotiation strategies  
- How agents behave under iterative self-play  
- Performance against baseline opponents  
- Emergence of equilibrium-like negotiation dynamics  

## Install

```bash
git clone https://github.com/Mirgle/negotiation-rl.git
cd negotiation-rl
# optionally create vitual environment
pip install -r requirements.txt
```

## Example Usage
Usable scripts are in /scripts.
All commandline parameters are optional.
All commands should be called from negotiation_rl root.

**Train a model:**
```bash
python train.py \
    --timesteps 1500000 \
    --save-path models/agent_final.zip \
    --eval-interval 50000 \
    --selfplay-interval 100000
```

**Evaluate a Model against Baselines:**
```bash
python evaluate.py --model models/ppo_negotiation_agentA.zip --opponent greedy
```

**Run Demo Negotiation with a Model:**
```bash
python demo.py --model models/ppo_negotiation_agentA.zip --opponent greedy
```

Optional Opponent Choices:
- random
- greedy
- snapshot

Snapshot is a very early 250k timestep snapshot model trained against greedy/random baselines.

## Model Checkpoints
Checkpoints are included in /models.
There is a model measured at 250k timesteps, 1 million timesteps, agentA (most up-to-date, ~2 million time steps)
They can be demoed with these commands:
```bash
python evaluate.py --model models/ppo_negotiation_agentA.zip --opponent greedy
python evaluate.py --model models/model_250k.zip --opponent greedy
python evaluate.py --model models/model_1million.zip --opponent greedy
```