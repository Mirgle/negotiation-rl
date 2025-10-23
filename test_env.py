from environment.environment import NegotiationEnv
from agents.baseline_agents import RandomAgent, GreedyAgent

env = NegotiationEnv(n_rounds=10, seed=1002)
agentA = RandomAgent(name="A")
agentB = GreedyAgent(name="B")

obs, info = env.reset()
done = False
actor = "A"
last_offer = None

def print_last_action(action):
    if action is None:
        print("--")
    elif action["type"] == "pass":
        print(actor, "passes.")
    elif action["type"] == "accept":
        print(actor, "accepts the offer.")
    elif action["type"] == "propose":
        print(f"{actor} proposes the trade: {action['offer']}")
        
print("Starting negotiation between RandomAgent (A) and GreedyAgent (B)")
print("Initial States: ")
agentA.print_state(obs)
agentB.print_state(obs)

while not done:
    agent = agentA if actor == "A" else agentB
    action = agent.act(obs, last_offer)
    print_last_action(action)
    
    obs, rewards, done, info = env.step(actor, action)
    last_offer = action if action["type"] == "propose" else None    
    actor = "B" if actor == "A" else "A"

print("Final States: ")
agentA.print_state(obs)
agentB.print_state(obs)
