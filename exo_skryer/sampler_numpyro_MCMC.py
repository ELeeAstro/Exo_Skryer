"""
sampler_numpyro_MCMC.py
=======================

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

# sampler_numpyro.py
from __future__ import annotations
from typing import Dict

import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, HMC

from .build_prepared import Prepared

__all__ = [
    "run_nuts_numpyro"
]


# ---------------------------------------------------------------------
#  Potential function factory
# ---------------------------------------------------------------------

def _make_potential_fn(prep: Prepared):
    """
    Build a NumPyro-compatible potential function from Prepared.logprob.

    We represent all free parameters as a single latent site 'u'
    (the unconstrained parameter vector).
    """
    def potential_fn(z):
        u = z["u"]  # shape (..., dim_free)
        return -prep.logprob(u)  # negative log-posterior
    return potential_fn


# ---------------------------------------------------------------------
#  Shared MCMC runner
# ---------------------------------------------------------------------

def _run_numpyro_generic(
    kernel,
    prep: Prepared,
    warmup: int,
    draws: int,
    seed: int,
    chains: int,
) -> Dict[str, jnp.ndarray]:
    """
    Shared driver for NUTS / HMC kernels using the potential_fn(u).
    """
    rng_key = jax.random.PRNGKey(int(seed))

    mcmc = MCMC(
        kernel,
        num_warmup=int(warmup),
        num_samples=int(draws),
        num_chains=int(chains),
        progress_bar=True,
    )

    # --- init params: broadcast across chains if needed ---
    init_u = prep.init_u  # shape (dim_free,)

    if chains > 1:
        # (chains, dim_free): same starting point in each chain
        init_u = jnp.broadcast_to(init_u, (chains,) + init_u.shape)

    init_params = {"u": init_u}

    # no model; kernel uses potential_fn directly
    mcmc.run(rng_key, init_params=init_params)
    samples = mcmc.get_samples(group_by_chain=True)  # {"u": (chains, draws, dim_free)}

    u_samples = samples["u"]
    # For chains==1, squeeze the leading chain dim to get (draws, dim_free)
    if chains == 1:
        u_samples = u_samples[0]

    # ---- map back to constrained parameter dict ----
    out: Dict[str, jnp.ndarray] = {}

    # Free params
    for i, name in enumerate(prep.names):
        out[name] = prep.bijectors[i].forward(u_samples[..., i])

    # Fixed params
    fixed_shape = u_samples.shape[:-1]  # (chains, draws) or (draws,)
    for k, v in prep.fixed.items():
        out[k] = jnp.broadcast_to(v, fixed_shape)

    return out


# ---------------------------------------------------------------------
#  Public NUTS interface
# ---------------------------------------------------------------------

def run_nuts_numpyro(cfg, prep: Prepared, exp_dir) -> Dict[str, jnp.ndarray]:
    """
    NUTS sampler using NumPyro; returns constrained samples per parameter.

    Arguments
    ---------
    cfg      : SimpleNamespace from your YAML (must have cfg.sampling.nuts.*)
    prep     : Prepared object from build_prepared(...)
    exp_dir  : Path-like for the experiment directory (unused here, but
               kept in the signature for future saving / diagnostics).
    """
    nuts_cfg = cfg.sampling.nuts

    warmup = int(nuts_cfg.warmup)
    draws  = int(nuts_cfg.draws)
    seed   = int(nuts_cfg.seed)
    chains = int(getattr(nuts_cfg, "chains", 1))

    potential_fn = _make_potential_fn(prep)

    # forward_mode_differentiation=True often helps avoid NaNs in complex models
    kernel = NUTS(
        potential_fn=potential_fn,
        dense_mass=True,
        forward_mode_differentiation=True,
    )

    return _run_numpyro_generic(kernel, prep, warmup, draws, seed, chains)
