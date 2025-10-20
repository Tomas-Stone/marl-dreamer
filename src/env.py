"""
This file defines a wrapper for the Gymnasium CarRacing-v2 environment
to handle preprocessing of observations, making them suitable for a VAE.
"""

# env.py

import gymnasium as gym
import torch
from torchvision import transforms
