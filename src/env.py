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
        

