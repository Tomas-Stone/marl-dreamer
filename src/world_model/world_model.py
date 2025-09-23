"""World model scaffold combining encoder, RSSM, and decoder.

This provides a thin, backend-agnostic facade with stable interfaces.
Implementations should fill in the underlying modules using JAX/Flax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .encoder import Encoder, EncoderConfig
from .rssm import RSSM, RSSMConfig, RSSMState
from .decoder import Decoder, DecoderConfig


@dataclass
class WorldModelConfig:
    obs_dim: int
    action_dim: int
    encoder_output: int = 128
    deter_size: int = 128
    stoch_size: int = 32
    hidden_size: int = 128


class WorldModel:
    """Backend-agnostic Dreamer-style world model scaffold.

    Exposes stable methods that mirror typical Dreamer components; the
    actual math will be implemented in JAX/Flax in a follow-up pass.
    """

    def __init__(self, config: WorldModelConfig):
        self.config = config

        self.encoder = Encoder(EncoderConfig(output_size=config.encoder_output))
        self.rssm = RSSM(
            RSSMConfig(
                deter_size=config.deter_size,
                stoch_size=config.stoch_size,
                hidden_size=config.hidden_size,
            )
        )
        self.decoder = Decoder(DecoderConfig(output_dim=config.obs_dim))

    # ------------------------- Initialization -------------------------
    def init(self, rng: Any, sample_obs: Any) -> Dict[str, Any]:
        """Initialize all submodule parameters and return a flat dict."""
        params = {
            "encoder": self.encoder.init(rng, sample_obs),
            "rssm": self.rssm.init(rng),
            "decoder": self.decoder.init(rng, sample_latent=None),
        }
        return params

    def init_state(self, batch_size: int) -> RSSMState:
        return self.rssm.init_state(batch_size)

    # --------------------------- One step -----------------------------
    def observe_step(
        self,
        params: Dict[str, Any],
        prev: RSSMState,
        obs: Any,
        action: Any,
    ) -> RSSMState:
        """Posterior update using an observation and previous action."""
        embed = self.encoder.apply(params["encoder"], obs)
        return self.rssm.apply_observe_step(params["rssm"], prev, action, embed)

    def decode(self, params: Dict[str, Any], state: RSSMState) -> Any:
        """Decode latent state (deterministic + stochastic) to observation space."""
        latent = (state.h, state.z)
        return self.decoder.apply(params["decoder"], latent)

    # ------------------------- Sequence helpers -----------------------
    def observe(
        self,
        params: Dict[str, Any],
        init_state: RSSMState,
        obs_seq: Any,
        act_seq: Any,
    ) -> Tuple[RSSMState, RSSMState]:
        """Fold over a sequence of observations and actions.

        obs_seq: (T, B, obs_dim)
        act_seq: (T, B, action_dim)
        """
        raise NotImplementedError("Implement with backend scanning util")

