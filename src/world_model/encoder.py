"""Encoder interface scaffold.

Define a simple, backend-agnostic encoder API. Implementations can be
added later using JAX/Flax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EncoderConfig:
    """Configuration for the observation encoder."""

    output_size: int = 128


class Encoder:
    """Observation encoder scaffold.

    Methods follow an init/apply pattern similar to Flax.
    """

    def __init__(self, config: EncoderConfig):
        self.config = config

    def init(self, rng: Any, sample_obs: Any) -> Dict[str, Any]:
        """Initialize parameters.

        Args
        - rng: backend-specific RNG
        - sample_obs: example observation array for shape inference
        """
        return {}

    def apply(self, params: Dict[str, Any], obs: Any) -> Any:
        """Encode an observation into a latent embedding.

        Returns shape (B, E) where E = config.output_size.
        """
        raise NotImplementedError("Implement encoder with chosen backend")
