"""
sampler_blackjax_MCMC.py
========================
Self-contained BlackJAX NUTS sampler with internal parameter transforms.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import blackjax

__all__ = [
    "run_nuts_blackjax"
]

LOG_FLOOR = -1e300  # finite invalid logL for numerical stability


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


def _extract_offset_params(cfg, obs: dict) -> Tuple[List[str], jnp.ndarray, bool]:
    """
    Extract offset parameters and build mapping to data points.

    Returns
    -------
    offset_param_names : List[str]
        Names of offset parameters in order (matching offset_group_names).
    offset_group_idx : jnp.ndarray
        Integer array mapping each data point to its offset parameter index.
    has_offsets : bool
        Whether any offset parameters are defined.
    """
    group_names = obs.get("offset_group_names", np.array(["__no_offset__"]))
    group_idx = obs.get("offset_group_idx", np.zeros(len(obs["y"]), dtype=int))

    # Find offset parameters by naming convention "offset_<group_name>"
    param_map: Dict[str, str] = {}  # group_name -> param_name
    for p in cfg.params:
        name = p.name
        if name.startswith("offset_"):
            group_name = name[7:]  # strip "offset_" prefix
            param_map[group_name] = name

    # Check if we have any real offset groups (not __no_offset__)
    real_groups = [g for g in group_names if g != "__no_offset__"]
    has_offsets = len(real_groups) > 0 and len(param_map) > 0

    if has_offsets:
        # Validate all groups have corresponding offset parameters
        for g in real_groups:
            if g not in param_map:
                raise ValueError(
                    f"Offset group '{g}' found in data but no 'offset_{g}' parameter defined in YAML"
                )

        # Build offset_param_names in same order as group_names
        offset_param_names = [param_map[g] for g in group_names if g in param_map]

        # Reindex: map group_idx to offset_param_names order
        name_to_idx = {param_map[g]: i for i, g in enumerate(group_names) if g in param_map}
        reindexed = np.zeros_like(group_idx)
        for i, g in enumerate(group_names):
            if g in param_map:
                reindexed[group_idx == i] = name_to_idx[param_map[g]]
        offset_group_idx = jnp.asarray(reindexed, dtype=jnp.int32)
    else:
        offset_param_names = []
        offset_group_idx = jnp.zeros(len(obs["y"]), dtype=jnp.int32)

    return offset_param_names, offset_group_idx, has_offsets


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


def _build_loglik_u(sampled_names, bijectors, delta_dict, obs, fm,
                    offset_param_names, offset_group_idx, has_offsets):
    """Build log-likelihood in u-space with offset support."""
    y_obs = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

    def loglik_u(u_vec: jnp.ndarray) -> jnp.ndarray:
        # Unpack u to theta
        theta_map = {sampled_names[i]: bijectors[i].forward(u_vec[i]) for i in range(len(sampled_names))}
        # Inject delta parameters
        theta_map.update(delta_dict)

        # Compute forward model
        mu = fm(theta_map)

        # Apply instrument offsets if defined (offset params are in ppm)
        if has_offsets:
            offset_values = jnp.array([theta_map[n] for n in offset_param_names])
            offset_vec = offset_values[offset_group_idx] / 1e6  # ppm -> fractional
            y_shifted = y_obs + offset_vec  # positive offset shifts data UP
        else:
            y_shifted = y_obs

        res = y_shifted - mu
        sig = jnp.clip(dy_obs, 1e-300, jnp.inf)
        is_finite = jnp.all(jnp.isfinite(mu))

        def _ok(_):
            r = res / sig
            r = jnp.where(jnp.isfinite(r), r, 0.0)
            logC = -jnp.log(sig) - 0.5 * jnp.log(2.0 * jnp.pi)
            return jnp.sum(logC - 0.5 * (r * r))

        def _bad(_):
            return jnp.asarray(LOG_FLOOR)

        return lax.cond(is_finite, _ok, _bad, operand=None)

    return loglik_u


# ---------------------------------------------------------------------
#  BlackJAX single-chain runner
# ---------------------------------------------------------------------

def _run_blackjax_single_chain(
    logprob: Callable,
    init_u: jnp.ndarray,
    sampled_names: List[str],
    bijectors: List[Bijector],
    delta_dict: Dict[str, float],
    warmup: int,
    draws: int,
    seed: int,
) -> Dict[str, jnp.ndarray]:
    """
    Single-chain BlackJAX NUTS in u-space; returns constrained samples per parameter.
    """
    key = jax.random.PRNGKey(int(seed))

    # Window adaptation to tune step size / mass matrix
    wa = blackjax.window_adaptation(blackjax.nuts, logprob)
    (state, parameters), _ = wa.run(key, init_u, num_steps=int(warmup))

    # Build the NUTS step kernel with tuned parameters
    step_fn = blackjax.nuts(logprob, **parameters).step
    kernel = jax.jit(step_fn)

    @jax.jit
    def one_step(st, rng):
        st, info = kernel(rng, st)
        return st, st  # carry state, collect state

    # Split key to avoid reusing warmup key for sampling
    key, sample_key = jax.random.split(key)
    # Independent keys for each draw
    keys = jax.random.split(sample_key, int(draws))

    # Run the chain with lax.scan (fully JITted loop)
    _, states = jax.lax.scan(one_step, state, keys)
    u = states.position  # (draws, dim_free)

    # Transform output: u → physical
    out: Dict[str, jnp.ndarray] = {}
    for i, name in enumerate(sampled_names):
        out[name] = bijectors[i].forward(u[:, i])  # (draws,)

    # Inject delta parameters: broadcast to (draws,)
    for k, v in delta_dict.items():
        out[k] = jnp.full((int(draws),), v)

    return out


# ---------------------------------------------------------------------
#  Public NUTS interface
# ---------------------------------------------------------------------

def run_nuts_blackjax(cfg, obs: dict, fm: Callable, exp_dir) -> Dict[str, jnp.ndarray]:
    """
    NUTS sampler using BlackJAX; returns constrained samples per parameter.

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
        Shape: (draws,) for single chain.
    """
    # Extract NUTS configuration
    nuts_cfg = cfg.sampling.nuts
    warmup = int(nuts_cfg.warmup)
    draws = int(nuts_cfg.draws)
    seed = int(nuts_cfg.seed)
    chains = int(getattr(nuts_cfg, "chains", 1))

    if chains != 1:
        raise NotImplementedError("BlackJAX driver currently supports chains=1 only.")

    # Extract parameters
    sampled_params, delta_dict = _extract_params(cfg)
    sampled_names = [p.name for p in sampled_params]

    # Extract offset parameters
    offset_param_names, offset_group_idx, has_offsets = _extract_offset_params(cfg, obs)

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
    loglik_u = _build_loglik_u(sampled_names, bijectors, delta_dict, obs, fm,
                               offset_param_names, offset_group_idx, has_offsets)

    def logprob(u):
        return loglik_u(u) + logprior_u(u)

    # Run single chain
    samples = _run_blackjax_single_chain(
        logprob=logprob,
        init_u=init_u,
        sampled_names=sampled_names,
        bijectors=bijectors,
        delta_dict=delta_dict,
        warmup=warmup,
        draws=draws,
        seed=seed,
    )

    return samples
