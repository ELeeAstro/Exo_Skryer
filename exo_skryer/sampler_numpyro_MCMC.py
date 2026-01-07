"""
sampler_numpyro_MCMC.py
=======================
Self-contained NumPyro NUTS sampler with internal parameter transforms.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from numpyro.infer import MCMC, NUTS

__all__ = [
    "run_nuts_numpyro"
]


# ---------------------------------------------------------------------
#  Bijector utilities (copied from build_prepared.py)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Bijector:
    forward: Callable[[jnp.ndarray], jnp.ndarray]
    inverse: Callable[[jnp.ndarray], jnp.ndarray]
    log_abs_det_jac: Callable[[jnp.ndarray], jnp.ndarray]


def _identity_bijector() -> Bijector:
    return Bijector(
        forward=lambda u: u,
        inverse=lambda x: x,
        log_abs_det_jac=lambda u: jnp.zeros_like(u),
    )


def _log_bijector() -> Bijector:
    return Bijector(
        forward=lambda u: jnp.exp(u),
        inverse=lambda x: jnp.log(x),
        log_abs_det_jac=lambda u: u,
    )


def _logit_bijector(low: float, high: float) -> Bijector:
    lower = float(low)
    upper = float(high)
    width = upper - lower
    if not width > 0:
        raise ValueError("logit bounds must satisfy high > low")

    def forward(u):
        s = jax.nn.sigmoid(u)
        return lower + width * s

    def inverse(x):
        y = (x - lower) / width
        y = jnp.clip(y, 1e-12, 1 - 1e-12)
        return jnp.log(y) - jnp.log1p(-y)

    def log_det(u):
        s = jax.nn.sigmoid(u)
        s = jnp.clip(s, 1e-12, 1 - 1e-12)
        return jnp.log(width) + jnp.log(s) + jnp.log1p(-s)

    return Bijector(forward=forward, inverse=inverse, log_abs_det_jac=log_det)


def _whitened_log_bijector(low: float, high: float) -> Bijector:
    lower = float(low)
    upper = float(high)
    if not (upper > lower > 0):
        raise ValueError("log-whitened bounds must satisfy 0 < low < high")

    log_low = jnp.log(lower)
    log_high = jnp.log(upper)
    mean = 0.5 * (log_low + log_high)
    scale = 0.5 * (log_high - log_low)

    def forward(u):
        return jnp.exp(mean + scale * u)

    def inverse(theta):
        return (jnp.log(theta) - mean) / scale

    def log_det(u):
        theta = forward(u)
        return jnp.log(jnp.abs(scale)) + jnp.log(theta)

    return Bijector(forward=forward, inverse=inverse, log_abs_det_jac=log_det)


# ---------------------------------------------------------------------
#  Prior PDF utilities (copied from build_prepared.py)
# ---------------------------------------------------------------------

def _normal_logpdf(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5 * (z * z + jnp.log(2 * jnp.pi) + 2 * jnp.log(sigma))


def _lognormal_logpdf(x, mu, sigma):
    return jnp.where(
        x > 0,
        _normal_logpdf(jnp.log(x), mu, sigma) - jnp.log(x),
        -jnp.inf,
    )


def _uniform_logpdf(x, low, high):
    inside = (x >= low) & (x <= high)
    return jnp.where(inside, -jnp.log(high - low), -jnp.inf)


def _log_uniform_logpdf(x, low, high):
    inside = (x >= low) & (x <= high) & (x > 0)
    norm = jnp.log(high) - jnp.log(low)
    return jnp.where(inside, -jnp.log(x) - norm, -jnp.inf)


def _evaluate_prior_logpdf(dist: str, theta, params: dict) -> jnp.ndarray:
    selector = dist.lower()
    if selector in ("gaussian", "normal"):
        return _normal_logpdf(theta, params["mu"], params["sigma"])
    if selector == "lognormal":
        return _lognormal_logpdf(theta, params["mu"], params["sigma"])
    if selector == "uniform":
        return _uniform_logpdf(theta, params["low"], params["high"])
    if selector == "log_uniform":
        return _log_uniform_logpdf(theta, params["low"], params["high"])
    if selector == "delta":
        return jnp.array(0.0)
    return jnp.array(-jnp.inf)


# ---------------------------------------------------------------------
#  Parameter utilities (copied from build_prepared.py)
# ---------------------------------------------------------------------

def _infer_transform(dist: str, param) -> str:
    requested = getattr(param, "transform", None)
    if requested is not None:
        value = str(requested).lower()
        if value in {"identity", "log", "logit"}:
            return value
        raise ValueError(f"Unknown transform '{requested}' for param '{getattr(param,'name','?')}'")
    selector = dist.lower()
    if selector in {"uniform", "truncnormal"}:
        return "logit"
    if selector == "log_uniform":
        return "log"
    return "identity"


def _default_init(dist: str, param) -> float:
    if getattr(param, "init", None) is not None:
        return float(param.init)
    selector = dist.lower()
    if selector in {"gaussian", "normal"}:
        return float(getattr(param, "mu"))
    if selector in {"uniform"}:
        low = float(getattr(param, "low"))
        high = float(getattr(param, "high"))
        return 0.5 * (low + high)
    if selector == "log_uniform":
        low = float(getattr(param, "low"))
        high = float(getattr(param, "high"))
        return float(jnp.sqrt(low * high))
    if selector == "delta":
        value = getattr(param, "value", getattr(param, "init", None))
        if value is None:
            raise ValueError(f"delta param '{getattr(param,'name','?')}' needs 'value' or 'init'")
        return float(value)
    if selector == "lognormal":
        return float(jnp.exp(getattr(param, "mu")))
    raise ValueError(f"Param '{getattr(param,'name','?')}' requires 'init' or an inferable default for dist '{dist}'")


def _canonical_prior_params(dist: str, param) -> dict:
    selector = dist.lower()
    if selector in {"gaussian", "normal"}:
        return {"mu": float(param.mu), "sigma": float(param.sigma)}
    if selector in {"uniform", "log_uniform"}:
        return {"low": float(param.low), "high": float(param.high)}
    if selector == "lognormal":
        return {"mu": float(param.mu), "sigma": float(param.sigma)}
    if selector == "delta":
        return {}
    raise ValueError(f"Unsupported dist '{dist}' for param '{getattr(param,'name','?')}'")


# ---------------------------------------------------------------------
#  Internal helper functions for parameter extraction
# ---------------------------------------------------------------------

def _extract_params(cfg):
    """Separate sampled vs delta parameters."""
    sampled = []
    delta_dict = {}
    for p in cfg.params:
        dist = str(getattr(p, "dist", "")).lower()
        if dist == "delta":
            delta_dict[p.name] = float(getattr(p, "value", p.init))
        else:
            sampled.append(p)
    return sampled, delta_dict


def _build_bijector(dist: str, param) -> Bijector:
    """Build bijector for a parameter based on its distribution."""
    transform = _infer_transform(dist, param)
    if transform == "identity":
        return _identity_bijector()
    elif transform == "log":
        low = getattr(param, "low", None)
        high = getattr(param, "high", None)
        if low is not None and high is not None:
            return _whitened_log_bijector(float(low), float(high))
        else:
            return _log_bijector()
    elif transform == "logit":
        low = float(getattr(param, "low"))
        high = float(getattr(param, "high"))
        return _logit_bijector(low, high)
    else:
        raise ValueError(f"Unknown transform '{transform}' for param '{getattr(param,'name','?')}'")


def _compute_init_u(param, bijector: Bijector) -> float:
    """Compute initial u-space value for a parameter."""
    dist = str(getattr(param, "dist", "")).lower()
    init_theta = _default_init(dist, param)
    init_arr = jnp.asarray(init_theta)
    return float(bijector.inverse(init_arr))


def _build_logprior_u(sampled_params, bijectors, priors_tuple):
    """Build log-prior with Jacobians."""
    def logprior_u(u_vec: jnp.ndarray) -> jnp.ndarray:
        logp = 0.0
        for i, (dist_i, params_i) in enumerate(priors_tuple):
            theta_i = bijectors[i].forward(u_vec[i])
            logp += _evaluate_prior_logpdf(dist_i, theta_i, params_i)
            logp += bijectors[i].log_abs_det_jac(u_vec[i])
        return logp
    return logprior_u


def _build_loglik_u(sampled_names, bijectors, delta_dict, obs, fm):
    """Build log-likelihood in u-space."""
    y_obs = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

    def loglik_u(u_vec: jnp.ndarray) -> jnp.ndarray:
        # Unpack u to theta
        theta_map = {sampled_names[i]: bijectors[i].forward(u_vec[i]) for i in range(len(sampled_names))}
        # Inject delta parameters
        theta_map.update(delta_dict)

        # Compute forward model
        mu = fm(theta_map)
        res = y_obs - mu
        sig = jnp.clip(dy_obs, 1e-300, jnp.inf)
        is_finite = jnp.all(jnp.isfinite(mu))

        def _ok(_):
            r = res / sig
            r = jnp.where(jnp.isfinite(r), r, 0.0)
            logC = -jnp.log(sig) - 0.5 * jnp.log(2.0 * jnp.pi)
            return jnp.sum(logC - 0.5 * (r * r))

        def _bad(_):
            return -jnp.inf

        return lax.cond(is_finite, _ok, _bad, operand=None)

    return loglik_u


# ---------------------------------------------------------------------
#  Public NUTS interface
# ---------------------------------------------------------------------

def run_nuts_numpyro(cfg, obs: dict, fm: Callable, exp_dir) -> Dict[str, jnp.ndarray]:
    """
    NUTS sampler using NumPyro; returns constrained samples per parameter.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration namespace (must provide `cfg.sampling.nuts.*` and `cfg.params`).
    obs : dict
        Observational data dictionary with keys 'y', 'dy', 'wl', 'dwl'.
    fm : Callable
        Forward model function: fm(theta_dict) -> jnp.ndarray
    exp_dir : path-like
        Experiment directory (unused here; kept for API compatibility).

    Returns
    -------
    samples_dict : Dict[str, jnp.ndarray]
        Dictionary mapping parameter names to samples arrays.
        Shape: (draws,) for single chain, (chains, draws) for multiple chains.
    """
    # Extract NUTS configuration
    nuts_cfg = cfg.sampling.nuts
    warmup = int(nuts_cfg.warmup)
    draws = int(nuts_cfg.draws)
    seed = int(nuts_cfg.seed)
    chains = int(getattr(nuts_cfg, "chains", 1))

    # Extract parameters
    sampled_params, delta_dict = _extract_params(cfg)
    sampled_names = [p.name for p in sampled_params]

    # Build bijectors
    bijectors = [_build_bijector(str(getattr(p, "dist", "")).lower(), p) for p in sampled_params]

    # Build prior definitions
    priors_tuple = tuple([
        (str(getattr(p, "dist", "")).lower(), _canonical_prior_params(str(getattr(p, "dist", "")).lower(), p))
        for p in sampled_params
    ])

    # Compute init_u
    init_u = jnp.array([_compute_init_u(p, bij) for p, bij in zip(sampled_params, bijectors)])

    # Build log-functions
    logprior_u = _build_logprior_u(sampled_params, bijectors, priors_tuple)
    loglik_u = _build_loglik_u(sampled_names, bijectors, delta_dict, obs, fm)

    def logprob(u):
        return loglik_u(u) + logprior_u(u)

    # Create potential function for NumPyro (negative log-posterior)
    def potential_fn(z):
        u = z["u"]
        return -logprob(u)

    # Create NUTS kernel
    kernel = NUTS(
        potential_fn=potential_fn,
        dense_mass=True,
        forward_mode_differentiation=True,
    )

    # Setup MCMC
    rng_key = jax.random.PRNGKey(int(seed))
    mcmc = MCMC(
        kernel,
        num_warmup=int(warmup),
        num_samples=int(draws),
        num_chains=int(chains),
        progress_bar=True,
    )

    # Broadcast init_u across chains if needed
    if chains > 1:
        init_u = jnp.broadcast_to(init_u, (chains,) + init_u.shape)

    init_params = {"u": init_u}

    # Run MCMC
    mcmc.run(rng_key, init_params=init_params)
    samples = mcmc.get_samples(group_by_chain=True)  # {"u": (chains, draws, dim_free)}

    u_samples = samples["u"]
    # For chains==1, squeeze the leading chain dim to get (draws, dim_free)
    if chains == 1:
        u_samples = u_samples[0]

    # Transform output: u → physical
    out: Dict[str, jnp.ndarray] = {}
    for i, name in enumerate(sampled_names):
        out[name] = bijectors[i].forward(u_samples[..., i])

    # Inject delta parameters
    fixed_shape = u_samples.shape[:-1]  # (chains, draws) or (draws,)
    for k, v in delta_dict.items():
        out[k] = jnp.broadcast_to(v, fixed_shape)

    return out
