from pettingzoo.mpe import simple_spread_v3
import numpy as np

env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
env.reset(seed=42)

for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    action = env.action_space(agent).sample()  # random action
    env.step(action)
print("✅ Ran one episode successfully")