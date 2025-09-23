"""RSSM interface scaffold.

This module defines a backend-agnostic interface for a Recurrent
State-Space Model (RSSM) suitable for Dreamer-style world models.

The goal here is to lock down shapes and method contracts before
filling in an implementation (e.g., in JAX/Flax).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class RSSMState:
    """Container for the RSSM state.

    Attributes
    - h: deterministic hidden state, shape (B, H)
    - z: stochastic latent state, shape (B, Z)
    """

    h: Any
    z: Any


@dataclass
class RSSMConfig:
    """Configuration of the RSSM scaffold.

    Parameters
    - deter_size: size of deterministic hidden state
    - stoch_size: size of stochastic latent state
    - hidden_size: size of MLP/GRU hidden layers (backend-specific)
    """

    deter_size: int = 128
    stoch_size: int = 32
    hidden_size: int = 128


class RSSM:
    """Recurrent State-Space Model scaffold.

    This class is intentionally backend-agnostic. Implementations can be
    provided in JAX/Flax later by filling in `init`, `init_state`, and
    the `apply_*` methods.
    """

    def __init__(self, config: RSSMConfig):
        self.config = config

    # ------------------------- Initialization -------------------------
    def init(self, rng: Any) -> Dict[str, Any]:
        """Initialize model parameters.

        Args
        - rng: backend-specific random key/seed

        Returns
        - params: PyTree/dict of model parameters
        """
        # Placeholder: to be implemented with chosen backend (e.g., Flax)
        return {}

    def init_state(self, batch_size: int, *, device: Optional[Any] = None) -> RSSMState:
        """Create an all-zero initial state.

        Args
        - batch_size: number of parallel sequences
        - device: optional device hint for backend
        """
        H = self.config.deter_size
        Z = self.config.stoch_size
        zeros = _zeros_like_backend
        return RSSMState(h=zeros((batch_size, H), device=device), z=zeros((batch_size, Z), device=device))

    # --------------------------- One step -----------------------------
    def apply_observe_step(
        self,
        params: Dict[str, Any],
        prev: RSSMState,
        action: Any,
        embed: Any,
    ) -> RSSMState:
        """Posterior update: incorporate encoder embedding from observation.

        Shapes
        - prev.h: (B, H), prev.z: (B, Z)
        - action: (B, A)
        - embed: (B, E)
        Returns
        - next RSSMState
        """
        raise NotImplementedError("Implement with chosen backend (e.g., JAX/Flax)")

    def apply_imagine_step(
        self,
        params: Dict[str, Any],
        prev: RSSMState,
        action: Any,
    ) -> RSSMState:
        """Prior rollout step without observations (for imagination)."""
        raise NotImplementedError("Implement with chosen backend (e.g., JAX/Flax)")

    # -------------------------- Sequences -----------------------------
    def apply_observe(
        self,
        params: Dict[str, Any],
        init_state: RSSMState,
        actions: Any,
        embeds: Any,
    ) -> Tuple[RSSMState, RSSMState]:
        """Roll forward over a sequence with observations.

        Args
        - actions: (T, B, A)
        - embeds:  (T, B, E)
        Returns
        - last_state, stacked_states (T, B, ...)
        """
        raise NotImplementedError("Implement scan with chosen backend")

    def apply_imagine(
        self,
        params: Dict[str, Any],
        init_state: RSSMState,
        actions: Any,
    ) -> Tuple[RSSMState, RSSMState]:
        """Roll forward over a sequence without observations.

        Args
        - actions: (T, B, A)
        Returns
        - last_state, stacked_states (T, B, ...)
        """
        raise NotImplementedError("Implement scan with chosen backend")


# ----------------------------- Utilities ------------------------------
def _zeros_like_backend(shape: Tuple[int, ...], *, device: Optional[Any] = None):
    """Backend-agnostic zero allocation placeholder.

    Replace this with jnp.zeros if using JAX or np.zeros for NumPy.
    """
    try:
        import jax
        import jax.numpy as jnp

        if device is not None:
            with jax.default_device(device):
                return jnp.zeros(shape, dtype=jnp.float32)
        return jnp.zeros(shape, dtype=jnp.float32)
    except Exception:
        import numpy as np

        return np.zeros(shape, dtype=np.float32)
