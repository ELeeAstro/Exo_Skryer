"""
sampler_blackjax_MCMC.py
========================

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

# sampler_blackjax.py
from __future__ import annotations
from typing import Dict

import jax
import jax.numpy as jnp
import blackjax

from .build_prepared import Prepared  # your dataclass

__all__ = [
    "run_nuts_blackjax"
]


def _run_blackjax_single_chain(
    prep: Prepared,
    warmup: int,
    draws: int,
    seed: int,
) -> Dict[str, jnp.ndarray]:
    """
    Single-chain BlackJAX NUTS in u-space; returns constrained samples per parameter.

    Uses prep.logprob(u) as the log-posterior in unconstrained space.
    """
    key = jax.random.PRNGKey(int(seed))

    # --- window adaptation to tune step size / mass matrix ---
    wa = blackjax.window_adaptation(blackjax.nuts, prep.logprob)
    (state, parameters), _ = wa.run(key, prep.init_u, num_steps=int(warmup))

    # build the NUTS step kernel with tuned parameters
    step_fn = blackjax.nuts(prep.logprob, **parameters).step
    kernel = jax.jit(step_fn)

    @jax.jit
    def one_step(st, rng):
        st, info = kernel(rng, st)
        return st, st  # carry state, collect state

    # independent keys for each draw
    keys = jax.random.split(key, int(draws))

    # run the chain with lax.scan (fully JITted loop)
    _, states = jax.lax.scan(one_step, state, keys)
    u = states.position  # (draws, dim_free)

    # ---- map back to constrained parameter dict ----
    out: Dict[str, jnp.ndarray] = {}

    # Free params
    for i, name in enumerate(prep.names):
        out[name] = prep.bijectors[i].forward(u[:, i])  # (draws,)

    # Fixed params: broadcast to (draws,)
    for k, v in prep.fixed.items():
        out[k] = jnp.full((int(draws),), v)

    return out


def run_nuts_blackjax(cfg, prep: Prepared, exp_dir) -> Dict[str, jnp.ndarray]:
    """
    High-level BlackJAX NUTS driver, mirroring the NumPyro interface:

        samples_dict = run_blackjax_nuts(cfg, prep, exp_dir)

    Reads hyperparameters from:
        cfg.sampling.nuts.warmup
        cfg.sampling.nuts.draws
        cfg.sampling.nuts.seed
        cfg.sampling.nuts.chains   (currently only chains=1 is supported)

    Returns
    -------
    Dict[str, jnp.ndarray]
        Mapping parameter name -> samples with shape (draws,).
        This matches the NumPyro driver so you can feed it into to_inferencedata().
    """
    nuts_cfg = cfg.sampling.nuts

    warmup = int(nuts_cfg.warmup)
    draws  = int(nuts_cfg.draws)
    seed   = int(nuts_cfg.seed)
    chains = int(getattr(nuts_cfg, "chains", 1))

    if chains != 1:
        # you can extend this later with vmap/pmap; for now be explicit
        raise NotImplementedError("BlackJAX driver currently supports chains=1 only.")

    samples = _run_blackjax_single_chain(prep, warmup, draws, seed)

    # If you’d like it to look exactly like a single-chain NumPyro output to ArviZ,
    # you can leave shapes as (draws,) — to_inferencedata() will treat that as 1 chain.

    return samples
