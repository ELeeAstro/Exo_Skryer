"""
sampler_dynesty_NS.py
=====================

Dynesty nested sampling driver with JAX forward model integration.

Key points:
- Prior transform: unit cube -> physical params (no redundant "logit" handling)
- Gaussian likelihood + jitter (consistent with other samplers)
- Log-likelihood is JAX-jitted on-device; single host sync per call
- Delta parameters and optional defaults injected consistently
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Callable
from pathlib import Path
import pickle

import numpy as np
import jax
import jax.numpy as jnp

try:
    import dynesty
    from dynesty import NestedSampler, DynamicNestedSampler
    from dynesty import utils as dyutils
    DYNESTY_AVAILABLE = True
except ImportError:
    DYNESTY_AVAILABLE = False
    dynesty = None
    NestedSampler = None
    DynamicNestedSampler = None


__all__ = [
    "build_prior_transform_dynesty",
    "build_loglikelihood_dynesty",
    "run_nested_dynesty",
]


LOG_FLOOR = -1e300  # finite "invalid" logL for dynesty stability


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


def _prior_center_theta0(cfg, param_names: List[str]) -> np.ndarray:
    """Compute prior-centered initial point for warmup compilation."""
    theta0 = np.zeros((len(param_names),), dtype=np.float64)
    name_to_param = {p.name: p for p in cfg.params}
    for i, name in enumerate(param_names):
        p = name_to_param[name]
        dist = str(getattr(p, "dist", "")).lower()
        if dist == "uniform":
            lo, hi = float(p.low), float(p.high)
            theta0[i] = 0.5 * (lo + hi)
        elif dist in ("gaussian", "normal"):
            theta0[i] = float(p.mu)
        elif dist == "lognormal":
            theta0[i] = float(np.exp(p.mu))
        else:
            raise ValueError(f"Unsupported distribution '{dist}' for warmup")
    return theta0


def build_prior_transform_dynesty(cfg) -> Tuple[Callable[[np.ndarray], np.ndarray], List[str]]:
    """
    Build Dynesty prior transform from cfg.params.

    Supports:
      - uniform(low, high)
      - normal(mu, sigma)
      - lognormal(mu, sigma) where underlying normal is N(mu, sigma)
      - delta parameters are excluded (handled separately)

    Returns:
      prior_transform(u)->theta, param_names
    """
    from scipy.special import ndtri  # inverse standard normal CDF

    params_cfg = [p for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]
    param_names = [p.name for p in params_cfg]

    def prior_transform(u: np.ndarray) -> np.ndarray:
        theta = np.empty_like(u, dtype=np.float64)

        # Avoid ndtri(0/1) -> +/- inf
        eps = 1e-300
        u = np.clip(u, eps, 1.0 - eps)

        for i, p in enumerate(params_cfg):
            dist_name = str(getattr(p, "dist", "")).lower()

            if dist_name == "uniform":
                low = float(getattr(p, "low"))
                high = float(getattr(p, "high"))
                # NOTE: dynesty already samples u ~ Uniform(0,1). Any "logit" field is irrelevant here.
                theta[i] = low + u[i] * (high - low)

            elif dist_name in ("gaussian", "normal"):
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                theta[i] = mu + sigma * ndtri(u[i])

            elif dist_name == "lognormal":
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                theta[i] = np.exp(mu + sigma * ndtri(u[i]))

            else:
                raise ValueError(f"Unsupported distribution '{dist_name}' for parameter '{p.name}'")

        return theta

    return prior_transform, param_names


def build_loglikelihood_dynesty(cfg, obs: dict, fm: Callable, param_names: List[str]) -> Callable[[np.ndarray], float]:
    """
    Build Dynesty log-likelihood function that wraps a JAX-jitted loglike.

    Implements Gaussian + jitter model:
      - residual r = y_obs - mu
      - inflate via sigma_jit^2 = 10^(2c) (c in log10 space)
      - reject NaNs/Infs via -inf on device, and return LOG_FLOOR on host
      - supports instrument offset parameters (offset_<group> in ppm)
    """
    # Observations to device (closed over)
    y_obs    = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

    # Extract offset parameters
    offset_param_names, offset_group_idx, has_offsets = _extract_offset_params(cfg, obs)

    # Collect delta params once
    delta_dict: Dict[str, float] = {}
    for p in cfg.params:
        if str(getattr(p, "dist", "")).lower() == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is not None:
                delta_dict[p.name] = float(val)

    # Silent defaults that only apply if the parameter is NOT present in YAML at all
    OPTIONAL_DEFAULTS: Dict[str, float] = {
        "c": -99.0,  # log10(sigma_jit): "effectively zero jitter"
    }
    cfg_names = {p.name for p in cfg.params}
    optional_defaults_active = {k: v for k, v in OPTIONAL_DEFAULTS.items() if k not in cfg_names}

    # Make static-key dict builder for JIT: keys are fixed at trace time.
    # theta_vec is (D,) JAX array.
    def _vec_to_theta_dict(theta_vec: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        d = {name: theta_vec[i] for i, name in enumerate(param_names)}
        # Inject deltas as JAX scalars (static values)
        for k, v in delta_dict.items():
            d[k] = jnp.asarray(v, dtype=theta_vec.dtype)
        # Inject optional defaults only if not explicitly configured in YAML
        for k, v in optional_defaults_active.items():
            if k not in d:
                d[k] = jnp.asarray(v, dtype=theta_vec.dtype)
        return d

    @jax.jit
    def loglike_jax(theta_vec: jnp.ndarray) -> jnp.ndarray:
        params = _vec_to_theta_dict(theta_vec)

        mu = fm(params)  # (N,)
        valid_mu = jnp.all(jnp.isfinite(mu))

        def valid_ll(_):
            # Apply instrument offsets if defined (offset params are in ppm)
            if has_offsets:
                offset_values = jnp.array([params[n] for n in offset_param_names])
                offset_vec = offset_values[offset_group_idx] / 1e6  # ppm -> fractional
                y_shifted = y_obs + offset_vec  # positive offset shifts data UP
            else:
                y_shifted = y_obs

            r = y_shifted - mu

            # 'c' is guaranteed present: either in YAML (sampled/delta) or injected default
            c = params["c"]  # log10(sigma_jit)
            sig_jit2 = 10.0 ** (2.0 * c)  # 10^(2c)

            sig_eff = jnp.sqrt(dy_obs**2 + sig_jit2)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = -jnp.log(sig_eff) - 0.5 * jnp.log(2.0 * jnp.pi)
            ll = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)

            return jnp.where(jnp.isfinite(ll), ll, jnp.asarray(LOG_FLOOR))

        return jax.lax.cond(valid_mu, valid_ll, lambda _: jnp.asarray(LOG_FLOOR), operand=None)

    # Warmup compilation with prior-centered theta
    theta0 = _prior_center_theta0(cfg, param_names)
    _ = float(loglike_jax(jnp.asarray(theta0)))

    def loglikelihood(theta: np.ndarray) -> float:
        # dynesty passes numpy float array
        theta_vec = jnp.asarray(theta, dtype=jnp.float64)
        ll = loglike_jax(theta_vec)
        val = float(ll)  # single device sync per call
        if not np.isfinite(val):
            return LOG_FLOOR
        return val

    return loglikelihood


def run_nested_dynesty(
    cfg,
    obs: dict,
    fm: Callable,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run Dynesty nested sampling and return (samples_dict, evidence_info).
    """
    if not DYNESTY_AVAILABLE:
        raise ImportError("Dynesty is not installed. Install with: pip install dynesty")

    dy_cfg = cfg.sampling.dynesty

    nlive = int(getattr(dy_cfg, "nlive", 500))
    bound = str(getattr(dy_cfg, "bound", "multi"))
    sample = str(getattr(dy_cfg, "sample", "auto"))
    dlogz = float(getattr(dy_cfg, "dlogz", 0.5))
    maxiter = getattr(dy_cfg, "maxiter", None)
    maxcall = getattr(dy_cfg, "maxcall", None)
    bootstrap = int(getattr(dy_cfg, "bootstrap", 0))
    enlarge = getattr(dy_cfg, "enlarge", None)
    update_interval = getattr(dy_cfg, "update_interval", None)
    dynamic = bool(getattr(dy_cfg, "dynamic", False))
    print_progress = bool(getattr(dy_cfg, "print_progress", True))
    seed = int(getattr(dy_cfg, "seed", 42))

    exp_dir.mkdir(parents=True, exist_ok=True)

    prior_fn, param_names = build_prior_transform_dynesty(cfg)
    loglike_fn = build_loglikelihood_dynesty(cfg, obs, fm, param_names)
    ndim = len(param_names)

    print(f"[Dynesty] Running nested sampling...")
    print(f"[Dynesty] Free parameters: {ndim}")
    print(f"[Dynesty] Parameter names: {param_names}")
    print(f"[Dynesty] nlive: {nlive}")
    print(f"[Dynesty] bound: {bound}")
    print(f"[Dynesty] sample: {sample}")
    print(f"[Dynesty] dlogz: {dlogz}")
    print(f"[Dynesty] dynamic: {dynamic}")

    rstate = np.random.default_rng(seed)

    if dynamic:
        sampler = DynamicNestedSampler(
            loglike_fn,
            prior_fn,
            ndim,
            nlive=nlive,
            bound=bound,
            sample=sample,
            bootstrap=bootstrap,
            enlarge=enlarge,
            update_interval=update_interval,
            rstate=rstate,
        )
        sampler.run_nested(dlogz_init=dlogz, print_progress=print_progress)
    else:
        sampler = NestedSampler(
            loglike_fn,
            prior_fn,
            ndim,
            nlive=nlive,
            bound=bound,
            sample=sample,
            bootstrap=bootstrap,
            enlarge=enlarge,
            update_interval=update_interval,
            rstate=rstate,
        )
        sampler.run_nested(
            dlogz=dlogz,
            maxiter=maxiter,
            maxcall=maxcall,
            print_progress=print_progress,
        )

    results = sampler.results

    # Dynesty stores ncall per iteration; sum it for total likelihood calls.
    n_like = int(np.sum(results.ncall)) if hasattr(results, "ncall") else None

    evidence_info: Dict[str, Any] = {
        "logZ": float(results.logz[-1]),
        "logZ_err": float(results.logzerr[-1]),
        "ESS": float(getattr(results, "ess", np.nan)),
        "H": float(results.h[-1]) if hasattr(results, "h") else np.nan,
        "n_like": n_like,
        "sampler": "dynesty",
        "n_live": nlive,
        "dynamic": dynamic,
    }

    print(f"[Dynesty] Evidence: {evidence_info['logZ']:.3f} ± {evidence_info['logZ_err']:.3f}")
    if n_like is not None:
        print(f"[Dynesty] Likelihood evaluations: {n_like}")
    print(f"[Dynesty] ESS: {evidence_info['ESS']}")

    # Save full results object
    results_path = exp_dir / "dynesty_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    evidence_info["results_file"] = str(results_path)

    # Equal-weight posterior samples (numerically stable weights)
    logw = results.logwt - results.logz[-1]
    logw = logw - np.max(logw)
    w = np.exp(logw)
    w = w / np.sum(w)
    samples = dyutils.resample_equal(results.samples, w)
    n_samples = samples.shape[0]

    # Add n_samples to evidence_info
    evidence_info["n_samples"] = n_samples

    samples_dict: Dict[str, np.ndarray] = {}

    # Free parameters
    for i, name in enumerate(param_names):
        samples_dict[name] = samples[:, i]

    # Delta parameters
    for p in cfg.params:
        name = p.name
        if name not in samples_dict:
            if str(getattr(p, "dist", "")).lower() == "delta":
                val = getattr(p, "value", getattr(p, "init", None))
                if val is not None:
                    samples_dict[name] = np.full((n_samples,), float(val), dtype=np.float64)

    return samples_dict, evidence_info
