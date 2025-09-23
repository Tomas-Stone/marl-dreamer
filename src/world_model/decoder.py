"""Decoder interface scaffold.

Define a backend-agnostic decoder that maps latent state to
reconstructed observations (or predictive heads such as reward).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DecoderConfig:
    """Configuration for the decoder."""

    output_dim: int


class Decoder:
    """Observation decoder scaffold.

    Methods follow an init/apply pattern similar to Flax.
    """

    def __init__(self, config: DecoderConfig):
        self.config = config

    def init(self, rng: Any, sample_latent: Any) -> Dict[str, Any]:
        """Initialize parameters from a latent sample (for shape inference)."""
        return {}

    def apply(self, params: Dict[str, Any], latent: Any) -> Any:
        """Decode latent state to observation prediction with shape (B, D).

        D = config.output_dim.
        """
        raise NotImplementedError("Implement decoder with chosen backend")
