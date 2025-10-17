"""
This file defines a wrapper for the Gymnasium CarRacing-v2 environment
to handle preprocessing of observations, making them suitable for a VAE.
"""

# env.py

import gymnasium as gym
import torch
from torchvision import transforms

class CarRacingEnv:
    """A wrapper for the CarRacing-v2 environment to handle preprocessing."""
    def __init__(self, device):
        # We use discrete actions for simplicity
        self.env = gym.make("CarRacing-v3", continuous=False)
        self.device = device
        
        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.Grayscale(), # Convert to single channel for simplicity
            transforms.ToTensor(), # Converts to [C, H, W] and scales to [0, 1]
        ])

    def reset(self):
        """Resets the environment and returns the preprocessed initial observation."""
        obs, _ = self.env.reset()
        return self.transform(obs).to(self.device)

    def step(self, action):
        """Takes an action and returns the preprocessed next observation and other info."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self.transform(obs).to(self.device)
        return processed_obs, reward, terminated, truncated, info

    @property
    def action_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()

