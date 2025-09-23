"""Imagination rollout scaffold.

Defines a backend-agnostic interface for rolling out the RSSM prior
given a policy function. The actual scan/loop should be implemented in
the chosen backend (e.g., JAX with lax.scan).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from .rssm import RSSM, RSSMState


def rollout_imagine(
    rssm: RSSM,
    params: Dict[str, Any],
    start_state: RSSMState,
    policy_fn: Callable[[Any, RSSMState, Any], Any],
    policy_params: Any,
    horizon: int,
    rng: Any,
) -> Tuple[RSSMState, RSSMState]:
    """Roll out imagined trajectories for `horizon` steps.

    Args
    - rssm: RSSM instance (scaffold)
    - params: RSSM parameters
    - start_state: initial RSSMState
    - policy_fn: function producing actions, signature (params, state, rng) -> action
    - policy_params: parameters passed to policy_fn
    - horizon: number of imagination steps
    - rng: backend RNG

    Returns
    - last_state, stacked_states (T, ...)
    """
    raise NotImplementedError("Implement with chosen backend (e.g., JAX lax.scan)")
