"""
sampler_ultranest_NS.py
=======================

UltraNest nested sampler driver with internal prior transform and likelihood construction.

- Prior transform: unit cube -> physical params (no redundant "logit" handling)
- Uses scipy.special.ndtri for Normal / LogNormal transforms
- Injects delta params and optional defaults (e.g. c=-99) into the forward-model dict
- Log-likelihood computed on-device (JAX), single host sync per call (float(logL))
- Avoids noisy per-call exception printing
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Callable
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

try:
    from ultranest import ReactiveNestedSampler
    ULTRANEST_AVAILABLE = True
except ImportError:
    ULTRANEST_AVAILABLE = False
    ReactiveNestedSampler = None

__all__ = [
    "build_prior_transform_ultranest",
    "build_loglikelihood_ultranest",
    "run_nested_ultranest",
]

LOG_FLOOR = -1e100  # finite invalid logL (robust for nested samplers)


def build_prior_transform_ultranest(cfg) -> Tuple[Callable[[np.ndarray], np.ndarray], List[str]]:
    """
    Build UltraNest prior transform from cfg.params.

    Note: UltraNest supplies u ~ Uniform(0,1)^D already, so YAML 'transform: logit'
    should NOT change prior_transform. (Keep 'transform' for JAX samplers only.)
    """
    from scipy.special import ndtri  # inverse standard normal CDF

    params_cfg = [p for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]
    param_names = [p.name for p in params_cfg]

    # Pre-extract all parameter info to avoid closure issues and for safety
    param_info = []
    for p in params_cfg:
        dist_name = str(getattr(p, "dist", "")).lower()
        info = {"name": p.name, "dist": dist_name}

        if dist_name == "uniform":
            info["low"] = float(getattr(p, "low"))
            info["high"] = float(getattr(p, "high"))
        elif dist_name in ("gaussian", "normal"):
            info["mu"] = float(getattr(p, "mu"))
            info["sigma"] = float(getattr(p, "sigma"))
        elif dist_name == "lognormal":
            info["mu"] = float(getattr(p, "mu"))
            info["sigma"] = float(getattr(p, "sigma"))
        else:
            raise ValueError(f"Unsupported distribution '{dist_name}' for parameter '{p.name}'")

        param_info.append(info)

    def prior_transform(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)

        # Handle both vectorized (2D) and non-vectorized (1D) inputs
        # When vectorized=True, u has shape (n_samples, n_params)
        # When vectorized=False, u has shape (n_params,)
        is_batch = u.ndim == 2

        if is_batch:
            n_samples, n_params = u.shape
            if n_params != len(param_info):
                raise ValueError(
                    f"Dimension mismatch: got {n_params} parameters, expected {len(param_info)}. "
                    f"Parameter names: {[info['name'] for info in param_info]}"
                )
        else:
            n_params = len(u)
            if n_params != len(param_info):
                raise ValueError(
                    f"Dimension mismatch: got {n_params} parameters, expected {len(param_info)}. "
                    f"Parameter names: {[info['name'] for info in param_info]}"
                )

        theta = np.empty_like(u, dtype=np.float64)

        eps = 1e-12
        u_clipped = np.clip(u, eps, 1.0 - eps)

        for i, info in enumerate(param_info):
            dist_name = info["dist"]

            if is_batch:
                # Vectorized: u_clipped[:, i] has shape (n_samples,)
                u_i = u_clipped[:, i]
            else:
                # Non-vectorized: u_clipped[i] is a scalar
                u_i = u_clipped[i]

            if dist_name == "uniform":
                theta_i = info["low"] + u_i * (info["high"] - info["low"])

            elif dist_name in ("gaussian", "normal"):
                theta_i = info["mu"] + info["sigma"] * ndtri(u_i)

            elif dist_name == "lognormal":
                theta_i = np.exp(info["mu"] + info["sigma"] * ndtri(u_i))

            # Assign to appropriate location
            if is_batch:
                theta[:, i] = theta_i
            else:
                theta[i] = theta_i

        return theta

    return prior_transform, param_names


def build_loglikelihood_ultranest(cfg, obs: dict, fm: Callable, param_names: List[str]) -> Callable[[np.ndarray], float]:
    """
    Build UltraNest log-likelihood function. The heavy lifting is done in JAX.

    Implements Gaussian + jitter likelihood:
      - jitter uses c = log10(sigma_jit), sigma_jit^2 = 10^(2c)
      - invalid mu / invalid ll -> LOG_FLOOR on host
    """
    y_obs    = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

    # Delta params (fixed) injected into theta_dict for consistency
    delta_dict: Dict[str, float] = {}
    for p in cfg.params:
        if str(getattr(p, "dist", "")).lower() == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is not None:
                delta_dict[p.name] = float(val)

    # Optional defaults only if parameter name not present anywhere in YAML
    OPTIONAL_DEFAULTS: Dict[str, float] = {"c": -99.0}
    cfg_names = {p.name for p in cfg.params}
    optional_defaults_active = {k: v for k, v in OPTIONAL_DEFAULTS.items() if k not in cfg_names}

    # Build a fixed-key dict inside jit
    def _vec_to_theta_dict(theta_vec: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        d = {name: theta_vec[i] for i, name in enumerate(param_names)}
        for k, v in delta_dict.items():
            d[k] = jnp.asarray(v, dtype=theta_vec.dtype)
        for k, v in optional_defaults_active.items():
            if k not in d:
                d[k] = jnp.asarray(v, dtype=theta_vec.dtype)
        return d

    @jax.jit
    def loglike_jax(theta_vec: jnp.ndarray) -> jnp.ndarray:
        theta_map = _vec_to_theta_dict(theta_vec)

        mu = fm(theta_map)
        valid_mu = jnp.all(jnp.isfinite(mu))

        def invalid_ll(_):
            return -jnp.inf

        def valid_ll(_):
            r = y_obs - mu

            c = theta_map["c"]                # always present (YAML or default)
            sig_jit2 = 10.0 ** (2.0 * c)      # 10^(2c)

            sig_eff = jnp.sqrt(dy_obs**2 + sig_jit2)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = -jnp.log(sig_eff) - 0.5 * jnp.log(2.0 * jnp.pi)
            ll = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)

            return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

        return jax.lax.cond(valid_mu, valid_ll, invalid_ll, operand=None)

    # Warm-up compilation once (helps avoid "first call compiles during sampling")
    _ = float(loglike_jax(jnp.zeros((len(param_names),), dtype=jnp.float64)))

    # Host callback UltraNest expects (non-vectorized)
    def loglikelihood(theta: np.ndarray) -> float:
        theta_vec = jnp.asarray(theta, dtype=jnp.float64)
        val = float(loglike_jax(theta_vec))  # one device sync
        if not np.isfinite(val):
            return LOG_FLOOR
        return val

    return loglikelihood


def run_nested_ultranest(
    cfg,
    obs: dict,
    fm: Callable,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run UltraNest nested sampling.
    """
    if not ULTRANEST_AVAILABLE:
        raise ImportError("UltraNest is not installed. Install with: pip install ultranest")

    un_cfg = getattr(cfg.sampling, "ultranest", None)
    if un_cfg is None:
        raise ValueError("Missing cfg.sampling.ultranest configuration.")

    n_live = int(getattr(un_cfg, "num_live_points", 500))
    dlogz = float(getattr(un_cfg, "dlogz", 0.5))
    max_iters = int(getattr(un_cfg, "max_iters", 0))
    min_num_live_points = int(getattr(un_cfg, "min_num_live_points", n_live))
    show_status = bool(getattr(un_cfg, "show_status", True))
    verbose = bool(getattr(un_cfg, "verbose", True))

    exp_dir.mkdir(parents=True, exist_ok=True)

    prior_fn, param_names = build_prior_transform_ultranest(cfg)
    loglike_fn = build_loglikelihood_ultranest(cfg, obs, fm, param_names)

    # Construct sampler (non-vectorized for simplicity and compatibility)
    sampler = ReactiveNestedSampler(
        param_names,
        loglike_fn,
        prior_fn,
        vectorized=False
    )

    if verbose:
        print(f"[UltraNest] Running nested sampling...")
        print(f"[UltraNest] Free parameters: {len(param_names)}")
        print(f"[UltraNest] Parameter names: {param_names}")
        print(f"[UltraNest] min_num_live_points: {min_num_live_points}")
        print(f"[UltraNest] dlogz: {dlogz}")

    run_kwargs = dict(
        min_num_live_points=min_num_live_points,
        dlogz=dlogz,
        show_status=show_status,
        viz_callback=None,
    )
    if max_iters > 0:
        run_kwargs["max_iters"] = max_iters

    results = sampler.run(**run_kwargs)
    sampler.print_results()

    # Save full results object
    import pickle
    results_path = exp_dir / "ultranest_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    if verbose:
        print(f"[UltraNest] Results saved to {results_path}")

    logz = float(results.get("logz", np.nan))
    logzerr = float(results.get("logzerr", np.nan))

    evidence_info: Dict[str, Any] = {
        "logZ": logz,
        "logZ_err": logzerr,
        "ESS": float(results.get("ess", np.nan)),
        "H": float(results.get("H", np.nan)),
        "n_like": int(results.get("ncall", 0)),
        "sampler": "ultranest",
        "results_file": str(results_path),
        "n_live": n_live,
        "min_num_live_points": min_num_live_points,
        "dlogz": dlogz,
    }

    # Extract posterior samples
    if "samples" in results:
        samples = np.asarray(results["samples"], dtype=np.float64)
    elif "weighted_samples" in results and "points" in results["weighted_samples"]:
        samples = np.asarray(results["weighted_samples"]["points"], dtype=np.float64)
    else:
        raise RuntimeError("UltraNest results did not include posterior samples.")

    n_samples = samples.shape[0]

    # Add n_samples to evidence_info
    evidence_info["n_samples"] = n_samples

    if verbose:
        print(f"[UltraNest] Posterior samples: {n_samples}")

    # Build samples_dict for consistency with other samplers
    samples_dict: Dict[str, np.ndarray] = {name: samples[:, i] for i, name in enumerate(param_names)}

    # Add fixed/delta parameters
    for param in cfg.params:
        name = param.name
        if name not in samples_dict:
            if str(getattr(param, "dist", "")).lower() == "delta":
                val = getattr(param, "value", getattr(param, "init", None))
                if val is not None:
                    samples_dict[name] = np.full((n_samples,), float(val), dtype=np.float64)

    return samples_dict, evidence_info
