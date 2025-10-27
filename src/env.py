"""
This file defines a wrapper for the JAXMarl MPE environments
to handle preprocessing of observations, making them suitable for a VAE.
"""

# env.py

import matplotlib.animation as animation
import jax
from jaxmarl import make

class EnvWrapper:
    """
    A wrapper for the JAXMarl MPE environment that preprocesses 
    """

    def __init__(self, env_name="MPE_simple_tag_v3", max_steps=25):
        key = jax.random.PRNGKey(0)
        key, key_env = jax.random.split(key)
        self.env = make(env_name)
        self.obs, self.state = self.env.reset(key_env)
        self.agents = self.env.agents
        self.action_space = {agent: self.env.action_space(agent) for agent in self.agents}
        self.max_steps = max_steps
        self.current_step = 0

    
    def reset(self, key):
        key, key_env = jax.random.split(key)
        self.obs, self.state = self.env.reset(key_env)
        self.current_step = 0
        return self.obs, self.state

    def step(self, key, actions):
        key, key_step = jax.random.split(key)
        self.obs, self.state, rewards, dones, infos = self.env.step(key_step, self.state, actions)
        self.current_step += 1
        return self.obs, self.state, rewards, dones, infos
    
    