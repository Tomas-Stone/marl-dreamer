# Entry point: load config, env, initialize agent, start training
# Can load any agent and env based on config file and train the agent


import jax
from jaxmarl import make

class Trainer:

    def __init__(self, config):
        self.config = config
        self.env = make(config.env.name, config.env)

    
