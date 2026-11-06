"""
retrieval_target.py
===================

u-space view of a retrieval, shared by the SMC engines and evidence_post. The
integral of exp(loglik_u + logprior_u) over u is the Bayesian evidence, because
logprior_u includes the Jacobian of each parameter transform.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import jax
import jax.numpy as jnp

from .limb_asymmetry import jitter_param_name
from .sampler_blackjax_MCMC import (
    LOG_FLOOR,
    _build_bijector,
    _build_loglik_u,
    _build_logprior_u,
    _canonical_prior_params,
    _extract_offset_params,
    _extract_params,
)
from .sampler_nessai_NS import _prior_and_bounds

__all__ = [
    "LOG_FLOOR",
    "RetrievalTarget",
    "batched",
    "build_target",
]


@dataclass(frozen=True)
class RetrievalTarget:
    names: List[str]
    bijectors: List[Any]
    priors: List[Any]  # frozen scipy distributions, in the order of names
    loglik_u: Callable[[jnp.ndarray], jnp.ndarray]
    logprior_u: Callable[[jnp.ndarray], jnp.ndarray]
    fixed: Dict[str, float]  # delta parameters and the optional jitter default

    def logprob_u(self, u: jnp.ndarray) -> jnp.ndarray:
        return self.loglik_u(u) + self.logprior_u(u)

    def theta_to_u(self, theta: np.ndarray) -> np.ndarray:
        theta = jnp.asarray(theta)
        return np.stack(
            [np.asarray(b.inverse(theta[:, i])) for i, b in enumerate(self.bijectors)], axis=1
        )

    def u_to_theta(self, u: np.ndarray) -> np.ndarray:
        u = jnp.asarray(u)
        return np.stack(
            [np.asarray(b.forward(u[:, i])) for i, b in enumerate(self.bijectors)], axis=1
        )

    def sample_prior_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.stack([p.rvs(size=n, random_state=rng) for p in self.priors], axis=1)

    def samples_dict(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        """Return {name: samples}, including the fixed parameters."""
        samples = {name: np.asarray(theta[:, i], dtype=np.float64) for i, name in enumerate(self.names)}
        for name, value in self.fixed.items():
            samples[name] = np.full((theta.shape[0],), value, dtype=np.float64)
        return samples


def build_target(cfg, obs: dict, fm: Callable) -> RetrievalTarget:
    """Build the u-space log-likelihood and log-prior from cfg.params."""
    sampled_params, delta_dict = _extract_params(cfg)
    names = [p.name for p in sampled_params]
    offset_names, offset_idx, has_offsets = _extract_offset_params(cfg, obs)

    jitter_key = jitter_param_name(getattr(getattr(cfg, "physics", None), "rt_scheme", None))
    defaults = {} if jitter_key in {p.name for p in cfg.params} else {jitter_key: -99.0}

    dists = [str(getattr(p, "dist", "")).lower() for p in sampled_params]
    bijectors = [_build_bijector(d, p) for d, p in zip(dists, sampled_params)]
    priors_tuple = tuple((d, _canonical_prior_params(d, p)) for d, p in zip(dists, sampled_params))

    return RetrievalTarget(
        names=names,
        bijectors=bijectors,
        priors=[_prior_and_bounds(p)[0] for p in sampled_params],
        loglik_u=_build_loglik_u(names, bijectors, delta_dict, defaults, obs, fm,
                                 offset_names, offset_idx, has_offsets, jitter_key),
        logprior_u=_build_logprior_u(sampled_params, bijectors, priors_tuple),
        fixed={**delta_dict, **defaults},
    )


def batched(fn: Callable, chunk: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Evaluate fn over rows of an array in fixed-size chunks. The last chunk is
    padded, so the forward model compiles once for any number of rows.
    """
    vectorised = jax.jit(jax.vmap(fn))

    def call(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.empty(x.shape[0], dtype=np.float64)
        for start in range(0, x.shape[0], chunk):
            rows = x[start:start + chunk]
            n = rows.shape[0]
            if n < chunk:
                rows = np.concatenate([rows, np.repeat(rows[-1:], chunk - n, axis=0)])
            out[start:start + n] = np.asarray(vectorised(rows))[:n]
        return out

    return call
