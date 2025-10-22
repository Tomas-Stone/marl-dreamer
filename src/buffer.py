"""
buffer.py

A simple replay buffer for storing and sampling trajectory sequences.
Designed for world model training where we need sequences of experience, not single transitions.

For Goal 2: The buffer stores complete trajectories from environment rollouts.
Each trajectory contains: observations, actions, rewards, and done flags.

Scientific Design Principles:
1. Trajectory-based: Store sequences (not individual transitions) for temporal modeling
2. JAX-compatible: Uses jnp.ndarray for seamless integration with JaxMARL
3. Fixed capacity: Overwrites oldest trajectories when full
4. Simple sampling: Random trajectory selection for unbiased learning
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List


class ReplayBuffer:
    """
    A simple replay buffer that stores complete trajectories.
    
    Each trajectory is a sequence of (observation, action, reward, done) tuples.
    For single-agent environments in Phase 1.
    """
    
    def __init__(
        self,
        capacity: int = 1000,  # Number of trajectories to store
        sequence_length: int = 50,  # Length of each trajectory
        obs_shape: Tuple = None,  # Will be set dynamically
        action_dim: int = None,   # Will be set dynamically
    ):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of trajectories to store
            sequence_length: Length of each trajectory sequence
            obs_shape: Shape of observations (e.g., (3, 64, 64) for images)
            action_dim: Dimension of action space
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # Storage will be initialized on first add
        self.observations = None
        self.actions = None
        self.rewards = None
        self.dones = None
        
        # Track current size and position
        self.size = 0  # Number of trajectories stored
        self.position = 0  # Next position to write to
        
    def add_trajectory(
        self,
        observations: np.ndarray,  # Shape: [T, ...]
        actions: np.ndarray,       # Shape: [T,] or [T, action_dim]
        rewards: np.ndarray,       # Shape: [T,]
        dones: np.ndarray,         # Shape: [T,]
    ):
        """
        Add a complete trajectory to the buffer.
        
        Args:
            observations: Sequence of observations [T, ...]
            actions: Sequence of actions [T,] or [T, action_dim]
            rewards: Sequence of rewards [T,]
            dones: Sequence of done flags [T,]
        """
        # Initialize storage on first trajectory
        if self.observations is None:
            self._initialize_storage(observations, actions)
        
        # Ensure correct sequence length
        T = observations.shape[0]
        if T != self.sequence_length:
            # Truncate or pad if necessary
            observations = self._pad_or_truncate(observations, self.sequence_length)
            actions = self._pad_or_truncate(actions, self.sequence_length)
            rewards = self._pad_or_truncate(rewards, self.sequence_length)
            dones = self._pad_or_truncate(dones, self.sequence_length)
        
        # Store the trajectory
        self.observations[self.position] = observations
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards
        self.dones[self.position] = dones
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample_batch(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """
        Sample a batch of trajectories from the buffer.
        
        Args:
            batch_size: Number of trajectories to sample
            
        Returns:
            Dictionary containing batched trajectories:
                'observations': [B, T, ...]
                'actions': [B, T, ...]
                'rewards': [B, T]
                'dones': [B, T]
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Random sampling without replacement
        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        
        # Return as JAX arrays for training
        return {
            'observations': jnp.array(self.observations[indices]),
            'actions': jnp.array(self.actions[indices]),
            'rewards': jnp.array(self.rewards[indices]),
            'dones': jnp.array(self.dones[indices]),
        }
    
    def _initialize_storage(self, obs_sample: np.ndarray, action_sample: np.ndarray):
        """Initialize storage arrays based on first trajectory."""
        self.obs_shape = obs_sample.shape[1:]  # Remove time dimension
        
        if len(action_sample.shape) == 1:
            action_shape = ()
        else:
            action_shape = action_sample.shape[1:]
        
        # Pre-allocate storage
        self.observations = np.zeros(
            (self.capacity, self.sequence_length, *self.obs_shape),
            dtype=np.float32
        )
        self.actions = np.zeros(
            (self.capacity, self.sequence_length, *action_shape),
            dtype=np.int32 if len(action_shape) == 0 else np.float32
        )
        self.rewards = np.zeros(
            (self.capacity, self.sequence_length),
            dtype=np.float32
        )
        self.dones = np.zeros(
            (self.capacity, self.sequence_length),
            dtype=np.bool_
        )
    
    def _pad_or_truncate(self, array: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate array to target length along first dimension."""
        current_length = array.shape[0]
        
        if current_length == target_length:
            return array
        elif current_length > target_length:
            # Truncate
            return array[:target_length]
        else:
            # Pad with zeros
            pad_width = [(0, target_length - current_length)] + [(0, 0)] * (array.ndim - 1)
            return np.pad(array, pad_width, mode='constant', constant_values=0)
    
    def __len__(self) -> int:
        """Return the current number of trajectories in the buffer."""
        return self.size
    
    def is_ready(self, min_trajectories: int = 10) -> bool:
        """Check if buffer has enough data to start training."""
        return self.size >= min_trajectories
