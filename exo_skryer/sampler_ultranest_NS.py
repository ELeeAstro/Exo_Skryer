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

LOG_FLOOR = -1e300  # finite invalid logL (robust for nested samplers)


def _extract_offset_params(cfg, obs: dict) -> Tuple[List[str], jnp.ndarray, bool]:
    """
    Extract offset parameters and build mapping to data points.

    Returns
    -------
    offset_param_names : List[str]
        Names of offset parameters in order (matching offset_group_names).
    offset_group_idx : jnp.ndarray
        Integer array mapping each data point to its offset parameter index.
        Uses -1 for points with no offset applied.
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

        # Reindex: map group_idx to offset_param_names order (-1 => no offset)
        name_to_idx = {param_map[g]: i for i, g in enumerate(group_names) if g in param_map}
        reindexed = np.full_like(group_idx, fill_value=-1)
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

        eps = 1e-300
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
      - supports instrument offset parameters (offset_<group> in ppm)
    """
    y_obs    = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

    # Extract offset parameters
    offset_param_names, offset_group_idx, has_offsets = _extract_offset_params(cfg, obs)

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
            return jnp.asarray(LOG_FLOOR)

        def valid_ll(_):
            # Apply instrument offsets if defined (offset params are in ppm)
            if has_offsets:
                offset_values = jnp.array([theta_map[n] for n in offset_param_names])
                idx_safe = jnp.clip(offset_group_idx, 0, offset_values.shape[0] - 1)
                mask = (offset_group_idx >= 0).astype(y_obs.dtype)
                offset_vec = (offset_values[idx_safe] / 1e6) * mask  # ppm -> fractional
                y_shifted = y_obs + offset_vec  # positive offset shifts data UP
            else:
                y_shifted = y_obs

            r = y_shifted - mu

            c = theta_map["c"]                # always present (YAML or default)
            sig_jit2 = 10.0 ** (2.0 * c)      # 10^(2c)

            sig_eff = jnp.sqrt(dy_obs**2 + sig_jit2)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = -jnp.log(sig_eff) - 0.5 * jnp.log(2.0 * jnp.pi)
            ll = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)

            return jnp.where(jnp.isfinite(ll), ll, jnp.asarray(LOG_FLOOR))

        return jax.lax.cond(valid_mu, valid_ll, invalid_ll, operand=None)

    # Warmup compilation with prior-centered theta
    theta0 = _prior_center_theta0(cfg, param_names)
    _ = float(loglike_jax(jnp.asarray(theta0)))

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
    frac_remain = getattr(un_cfg, "frac_remain", None)
    show_status = bool(getattr(un_cfg, "show_status", True))
    verbose = bool(getattr(un_cfg, "verbose", True))
    use_stepsampler = bool(getattr(un_cfg, "use_stepsampler", False))
    stepsampler_nsteps = int(getattr(un_cfg, "stepsampler_nsteps", 30))
    stepsampler_direction = str(getattr(un_cfg, "stepsampler_direction", "mixture")).lower()

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

    print(f"[UltraNest] Running nested sampling...")
    print(f"[UltraNest] Free parameters: {len(param_names)}")
    print(f"[UltraNest] Parameter names: {param_names}")
    print(f"[UltraNest] num_live_points: {n_live}")
    print(f"[UltraNest] min_num_live_points: {min_num_live_points}")
    print(f"[UltraNest] dlogz: {dlogz}")
    print(f"[UltraNest] frac_remain: {frac_remain}")
    print(f"[UltraNest] use_stepsampler: {use_stepsampler}")

    run_kwargs = dict(
        min_num_live_points=min_num_live_points,
        dlogz=dlogz,
        show_status=show_status,
        viz_callback=None,
    )
    if frac_remain is not None:
        run_kwargs["frac_remain"] = float(frac_remain)
    if max_iters > 0:
        run_kwargs["max_iters"] = max_iters

    if use_stepsampler:
        try:
            from ultranest.stepsampler import (
                SliceSampler,
                generate_mixture_random_direction,
                generate_random_direction,
            )
        except ImportError as e:
            raise ImportError(
                "UltraNest stepsampler support is not available in this installation."
            ) from e

        if stepsampler_direction == "random":
            direction_fn = generate_random_direction
        elif stepsampler_direction == "mixture":
            direction_fn = generate_mixture_random_direction
        else:
            raise ValueError(
                f"Unknown stepsampler_direction={stepsampler_direction!r}. "
                f"Use 'mixture' or 'random'."
            )

        try:
            sampler.stepsampler = SliceSampler(
                nsteps=stepsampler_nsteps,
                generate_direction=direction_fn,
            )
        except TypeError:
            # Compatibility fallback for older UltraNest APIs.
            sampler.stepsampler = SliceSampler(stepsampler_nsteps, direction_fn)

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
        "n_live": min_num_live_points,
        "num_live_points_cfg": n_live,
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
