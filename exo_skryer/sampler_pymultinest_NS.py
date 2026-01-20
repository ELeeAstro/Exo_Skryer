"""
sampler_pymultinest_NS.py
==========================

PyMultiNest nested sampler driver with JAX forward model integration.

Key points:
- Prior transform: unit cube -> physical params (in-place modification)
- Identical likelihood to Dynesty/UltraNest (Gaussian + jitter)
- File-based output (PyMultiNest requirement)
- Results read via pymultinest.Analyzer

PyMultiNest is a Python wrapper around the MultiNest Fortran code.
Installation: pip install pymultinest (requires MultiNest library)
See: https://johannesbuchner.github.io/PyMultiNest/
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Callable
from pathlib import Path
import pickle

import numpy as np
import jax
import jax.numpy as jnp

try:
    import pymultinest
    PYMULTINEST_AVAILABLE = True
except ImportError:
    PYMULTINEST_AVAILABLE = False
    pymultinest = None


__all__ = [
    "build_prior_transform_pymultinest",
    "build_loglikelihood_pymultinest",
    "run_nested_pymultinest",
]


LOG_FLOOR = -1e300  # finite "invalid" logL for PyMultiNest stability


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


def build_prior_transform_pymultinest(cfg) -> Tuple[Callable, List[str]]:
    """
    Build PyMultiNest prior transform from cfg.params.

    PyMultiNest expects in-place transformation:
      - Input: cube (mutable np.ndarray), ndim (int), nparams (int)
      - Action: Transform cube[i] from [0,1] to physical parameter in-place
      - Return: None (modifies cube in-place)

    Supports:
      - uniform(low, high)
      - normal(mu, sigma)
      - lognormal(mu, sigma) where underlying normal is N(mu, sigma)
      - delta parameters are excluded (handled separately)

    Returns:
      prior_transform(cube, ndim, nparams), param_names
    """
    from scipy.special import ndtri  # inverse standard normal CDF

    params_cfg = [p for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]
    param_names = [p.name for p in params_cfg]

    # Pre-extract parameter info to avoid closure issues and for efficiency
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

    def prior_transform(cube, ndim, nparams):
        """PyMultiNest prior callback (in-place transformation)."""
        # Avoid ndtri(0/1) -> +/- inf
        eps = 1e-12

        for i, info in enumerate(param_info):
            u = np.clip(cube[i], eps, 1.0 - eps)
            dist_name = info["dist"]

            if dist_name == "uniform":
                cube[i] = info["low"] + u * (info["high"] - info["low"])

            elif dist_name in ("gaussian", "normal"):
                cube[i] = info["mu"] + info["sigma"] * ndtri(u)

            elif dist_name == "lognormal":
                cube[i] = np.exp(info["mu"] + info["sigma"] * ndtri(u))

    return prior_transform, param_names


def build_loglikelihood_pymultinest(cfg, obs: dict, fm: Callable, param_names: List[str]) -> Callable:
    """
    Build PyMultiNest log-likelihood function that wraps a JAX-jitted loglike.

    PyMultiNest expects:
      - Input: cube (np.ndarray of transformed physical params), ndim, nparams
      - Return: float (log-likelihood value)

    Implements Gaussian + jitter model:
      - residual r = y_obs - mu
      - inflate via sigma_jit^2 = 10^(2c) (c in log10 space)
      - reject NaNs/Infs via -inf on device, and return LOG_FLOOR on host
    """
    # Observations to device (closed over)
    y_obs = jnp.asarray(obs["y"])
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
                y_shifted = y_obs - offset_vec
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

    def loglikelihood(cube, ndim, nparams):
        """PyMultiNest log-likelihood callback.

        Note: cube already contains transformed physical parameters (after prior_transform).
        """
        # PyMultiNest passes numpy float array; cube[:ndim] to handle potential oversized array
        theta_vec = jnp.asarray(cube[:ndim], dtype=jnp.float64)
        ll = loglike_jax(theta_vec)
        val = float(ll)  # single device sync per call
        if not np.isfinite(val):
            return LOG_FLOOR
        return val

    return loglikelihood


def run_nested_pymultinest(
    cfg,
    obs: dict,
    fm: Callable,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run PyMultiNest nested sampling and return (samples_dict, evidence_info).

    PyMultiNest creates many output files with basename prefix.
    Results are read using pymultinest.Analyzer after sampling completes.
    """
    if not PYMULTINEST_AVAILABLE:
        raise ImportError(
            "PyMultiNest is not installed. Install with:\n"
            "  pip install pymultinest\n"
            "Note: Requires MultiNest Fortran library to be installed separately.\n"
            "See: https://johannesbuchner.github.io/PyMultiNest/install.html"
        )

    pmn_cfg = getattr(cfg.sampling, "pymultinest", None)
    if pmn_cfg is None:
        raise ValueError("Missing cfg.sampling.pymultinest configuration.")

    # Extract configuration parameters
    n_live_points = int(getattr(pmn_cfg, "n_live_points", 400))
    evidence_tolerance = float(getattr(pmn_cfg, "evidence_tolerance", 0.5))
    sampling_efficiency = float(getattr(pmn_cfg, "sampling_efficiency", 0.8))
    n_iter_before_update = int(getattr(pmn_cfg, "n_iter_before_update", 100))
    null_log_evidence = float(getattr(pmn_cfg, "null_log_evidence", -1e90))
    max_modes = int(getattr(pmn_cfg, "max_modes", 100))
    mode_tolerance = float(getattr(pmn_cfg, "mode_tolerance", -1e90))
    seed = int(getattr(pmn_cfg, "seed", -1))
    verbose = bool(getattr(pmn_cfg, "verbose", True))
    resume = bool(getattr(pmn_cfg, "resume", True))
    importance_nested_sampling = bool(getattr(pmn_cfg, "importance_nested_sampling", False))
    multimodal = bool(getattr(pmn_cfg, "multimodal", True))
    const_efficiency_mode = bool(getattr(pmn_cfg, "const_efficiency_mode", False))

    # Ensure output directory exists (don't create subdirectories)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Build prior and likelihood
    prior_fn, param_names = build_prior_transform_pymultinest(cfg)
    loglike_fn = build_loglikelihood_pymultinest(cfg, obs, fm, param_names)

    ndim = len(param_names)

    print(f"[PyMultiNest] Running nested sampling...")
    print(f"[PyMultiNest] Free parameters: {ndim}")
    print(f"[PyMultiNest] Parameter names: {param_names}")
    print(f"[PyMultiNest] n_live_points: {n_live_points}")
    print(f"[PyMultiNest] evidence_tolerance: {evidence_tolerance}")
    print(f"[PyMultiNest] sampling_efficiency: {sampling_efficiency}")
    print(f"[PyMultiNest] multimodal: {multimodal}")

    # Set output files basename (all files go directly to exp_dir)
    outputfiles_basename = str(exp_dir / "pymultinest_")

    # Run PyMultiNest
    pymultinest.run(
        LogLikelihood=loglike_fn,
        Prior=prior_fn,
        n_dims=ndim,
        n_params=ndim,  # same as n_dims for our case
        outputfiles_basename=outputfiles_basename,
        verbose=verbose,
        resume=resume,
        n_live_points=n_live_points,
        evidence_tolerance=evidence_tolerance,
        sampling_efficiency=sampling_efficiency,
        n_iter_before_update=n_iter_before_update,
        null_log_evidence=null_log_evidence,
        max_modes=max_modes,
        mode_tolerance=mode_tolerance,
        seed=seed,
        importance_nested_sampling=importance_nested_sampling,
        multimodal=multimodal,
        const_efficiency_mode=const_efficiency_mode,
    )

    if verbose:
        print(f"[PyMultiNest] Sampling complete. Reading results...")

    # Read results using Analyzer
    analyzer = pymultinest.Analyzer(
        n_params=ndim,
        outputfiles_basename=outputfiles_basename,
    )

    # Get statistics
    stats = analyzer.get_stats()

    # Extract evidence information
    logZ = float(stats['nested sampling global log-evidence'])
    logZ_err = float(stats['nested sampling global log-evidence error'])

    # Extract posterior samples (equal-weighted)
    posterior = analyzer.get_equal_weighted_posterior()
    # posterior is array with shape (n_samples, ndim + 2)
    # columns: [params..., -2*logL, logL]
    samples = posterior[:, :ndim]  # extract parameter columns
    n_samples = samples.shape[0]

    # Get best fit
    best_fit = analyzer.get_best_fit()
    best_logL = float(best_fit['log_likelihood'])

    if verbose:
        print(f"[PyMultiNest] Evidence: {logZ:.3f} ± {logZ_err:.3f}")
        print(f"[PyMultiNest] Posterior samples: {n_samples}")
        print(f"[PyMultiNest] Best-fit log-likelihood: {best_logL:.3f}")

    # Build evidence_info dict
    evidence_info: Dict[str, Any] = {
        "logZ": logZ,
        "logZ_err": logZ_err,
        "ESS": float(n_samples),  # PyMultiNest gives equal-weight samples
        "H": np.nan,  # MultiNest doesn't report H directly
        "n_like": np.nan,  # Not easily accessible from PyMultiNest
        "n_samples": n_samples,
        "best_logL": best_logL,
        "sampler": "pymultinest",
        "n_live": n_live_points,
        "evidence_tolerance": evidence_tolerance,
    }

    # Save stats to pickle for future reference
    stats_path = exp_dir / "pymultinest_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    evidence_info["stats_file"] = str(stats_path)

    if verbose:
        print(f"[PyMultiNest] Stats saved to {stats_path}")

    # Build samples_dict
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
