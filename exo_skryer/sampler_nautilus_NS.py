"""
sampler_nautilus_NS.py
======================

Nautilus nested sampler driver with JAX forward-model integration.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Callable
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats as sps

try:
    from nautilus import Prior, Sampler
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    Prior = None
    Sampler = None


__all__ = [
    "build_prior_nautilus",
    "build_loglikelihood_nautilus",
    "run_nested_nautilus",
]


LOG_FLOOR = -1e300  # finite invalid logL for numerical stability


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

    param_map: Dict[str, str] = {}
    for p in cfg.params:
        name = p.name
        if name.startswith("offset_"):
            group_name = name[7:]
            param_map[group_name] = name

    real_groups = [g for g in group_names if g != "__no_offset__"]
    has_offsets = len(real_groups) > 0 and len(param_map) > 0

    if has_offsets:
        for g in real_groups:
            if g not in param_map:
                raise ValueError(
                    f"Offset group '{g}' found in data but no 'offset_{g}' parameter defined in YAML"
                )

        offset_param_names = [param_map[g] for g in group_names if g in param_map]

        name_to_idx = {param_map[g]: i for i, g in enumerate(group_names) if g in param_map}
        reindexed = np.full_like(group_idx, fill_value=-1)
        for i, g in enumerate(group_names):
            if g in param_map:
                reindexed[group_idx == i] = name_to_idx[param_map[g]]
        offset_group_idx = jnp.asarray(reindexed, dtype=jnp.int32)
    else:
        offset_param_names = []
        offset_group_idx = jnp.full(len(obs["y"]), -1, dtype=jnp.int32)

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
        elif dist == "log_uniform":
            lo, hi = float(p.low), float(p.high)
            theta0[i] = float(np.sqrt(lo * hi))
        else:
            raise ValueError(f"Unsupported distribution '{dist}' for warmup")
    return theta0


def build_prior_nautilus(cfg) -> Tuple[Prior, List[str]]:
    """
    Build a Nautilus Prior from cfg.params and return (prior, free_param_names).
    """
    prior = Prior()
    param_names: List[str] = []

    for p in cfg.params:
        name = p.name
        dist = str(getattr(p, "dist", "")).lower()
        if dist == "delta":
            continue

        param_names.append(name)

        if dist == "uniform":
            prior.add_parameter(name, dist=(float(p.low), float(p.high)))
        elif dist in ("gaussian", "normal"):
            prior.add_parameter(name, dist=sps.norm(loc=float(p.mu), scale=float(p.sigma)))
        elif dist == "lognormal":
            prior.add_parameter(
                name,
                dist=sps.lognorm(s=float(p.sigma), scale=float(np.exp(float(p.mu)))),
            )
        elif dist == "log_uniform":
            prior.add_parameter(name, dist=sps.reciprocal(float(p.low), float(p.high)))
        else:
            raise ValueError(f"Unsupported distribution '{dist}' for parameter '{name}'")

    return prior, param_names


def build_loglikelihood_nautilus(cfg, obs: dict, fm: Callable, param_names: List[str]) -> Callable:
    """
    Build Nautilus log-likelihood callback: likelihood(theta_dict) -> float.
    """
    y_obs = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

    offset_param_names, offset_group_idx, has_offsets = _extract_offset_params(cfg, obs)

    delta_dict: Dict[str, float] = {}
    for p in cfg.params:
        if str(getattr(p, "dist", "")).lower() == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is not None:
                delta_dict[p.name] = float(val)

    OPTIONAL_DEFAULTS: Dict[str, float] = {"c": -99.0}
    cfg_names = {p.name for p in cfg.params}
    optional_defaults_active = {k: v for k, v in OPTIONAL_DEFAULTS.items() if k not in cfg_names}

    def _dict_to_theta_dict(theta_in: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        dtype = y_obs.dtype
        d = {name: jnp.asarray(theta_in[name], dtype=dtype) for name in param_names}
        for k, v in delta_dict.items():
            d[k] = jnp.asarray(v, dtype=dtype)
        for k, v in optional_defaults_active.items():
            if k not in d:
                d[k] = jnp.asarray(v, dtype=dtype)
        return d

    @jax.jit
    def loglike_jax(theta_map: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        mu = fm(theta_map)
        valid_mu = jnp.all(jnp.isfinite(mu))

        def valid_ll(_):
            if has_offsets:
                offset_values = jnp.array([theta_map[n] for n in offset_param_names])
                idx_safe = jnp.clip(offset_group_idx, 0, offset_values.shape[0] - 1)
                mask = (offset_group_idx >= 0).astype(y_obs.dtype)
                offset_vec = (offset_values[idx_safe] / 1e6) * mask
                y_shifted = y_obs + offset_vec
            else:
                y_shifted = y_obs

            r = y_shifted - mu
            c = theta_map["c"]
            sig_jit2 = 10.0 ** (2.0 * c)

            sig_eff = jnp.sqrt(dy_obs**2 + sig_jit2)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = -jnp.log(sig_eff) - 0.5 * jnp.log(2.0 * jnp.pi)
            ll = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)
            return jnp.where(jnp.isfinite(ll), ll, jnp.asarray(LOG_FLOOR))

        return jax.lax.cond(valid_mu, valid_ll, lambda _: jnp.asarray(LOG_FLOOR), operand=None)

    theta0 = _prior_center_theta0(cfg, param_names)
    theta0_map = {name: jnp.asarray(theta0[i], dtype=y_obs.dtype) for i, name in enumerate(param_names)}
    theta0_map.update({k: jnp.asarray(v, dtype=y_obs.dtype) for k, v in delta_dict.items()})
    theta0_map.update({k: jnp.asarray(v, dtype=y_obs.dtype) for k, v in optional_defaults_active.items()})
    _ = float(loglike_jax(theta0_map))

    def loglikelihood(theta_dict: Dict[str, Any]) -> float:
        theta_map = _dict_to_theta_dict(theta_dict)
        val = float(loglike_jax(theta_map))
        if not np.isfinite(val):
            return LOG_FLOOR
        return val

    return loglikelihood


def run_nested_nautilus(
    cfg,
    obs: dict,
    fm: Callable,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run Nautilus nested sampling and return (samples_dict, evidence_info).
    """
    if not NAUTILUS_AVAILABLE:
        raise ImportError("Nautilus is not installed. Install with: pip install nautilus-sampler")

    ncfg = getattr(cfg.sampling, "nautilus", None)
    if ncfg is None:
        raise ValueError("Missing cfg.sampling.nautilus configuration.")

    exp_dir.mkdir(parents=True, exist_ok=True)

    prior, param_names = build_prior_nautilus(cfg)
    loglike_fn = build_loglikelihood_nautilus(cfg, obs, fm, param_names)

    n_live = int(getattr(ncfg, "n_live", 1000))
    n_update = getattr(ncfg, "n_update", None)
    if n_update is not None:
        n_update = int(n_update)
    split_threshold = int(getattr(ncfg, "split_threshold", 100))
    n_networks = int(getattr(ncfg, "n_networks", 4))
    n_batch = getattr(ncfg, "n_batch", None)
    if n_batch is not None:
        n_batch = int(n_batch)
    n_like_new_bound = getattr(ncfg, "n_like_new_bound", None)
    if n_like_new_bound is not None:
        n_like_new_bound = int(n_like_new_bound)
    seed = getattr(ncfg, "seed", None)
    if seed is not None:
        seed = int(seed)
    filepath = getattr(ncfg, "filepath", None)
    resume = bool(getattr(ncfg, "resume", True))
    verbose = bool(getattr(ncfg, "verbose", True))

    f_live = float(getattr(ncfg, "f_live", 0.01))
    n_shell = int(getattr(ncfg, "n_shell", 1))
    n_eff = getattr(ncfg, "n_eff", None)
    if n_eff is not None:
        n_eff = float(n_eff)
    n_like_max = getattr(ncfg, "n_like_max", None)
    if n_like_max is not None:
        n_like_max = int(n_like_max)
    discard_exploration = bool(getattr(ncfg, "discard_exploration", False))
    timeout = getattr(ncfg, "timeout", None)
    if timeout is not None:
        timeout = float(timeout)
    equal_weight = bool(getattr(ncfg, "equal_weight", True))

    if filepath is None:
        filepath_use = str(exp_dir / "nautilus_checkpoint.hdf5")
    else:
        filepath_path = Path(filepath)
        if not filepath_path.is_absolute():
            filepath_path = (exp_dir / filepath_path)
        filepath_use = str(filepath_path)

    print(f"[Nautilus] Running nested sampling...")
    print(f"[Nautilus] Free parameters: {len(param_names)}")
    print(f"[Nautilus] Parameter names: {param_names}")
    print(f"[Nautilus] n_live: {n_live}")
    print(f"[Nautilus] f_live: {f_live}")
    print(f"[Nautilus] n_shell: {n_shell}")

    sampler_kwargs: Dict[str, Any] = dict(
        prior=prior,
        likelihood=loglike_fn,
        n_live=n_live,
        n_update=n_update,
        split_threshold=split_threshold,
        n_networks=n_networks,
        n_batch=n_batch,
        n_like_new_bound=n_like_new_bound,
        vectorized=False,
        pass_dict=True,
        seed=seed,
        filepath=filepath_use,
        resume=resume,
    )
    try:
        sampler = Sampler(**sampler_kwargs)
    except TypeError:
        # Compatibility fallback for versions without pass_dict.
        sampler_kwargs.pop("pass_dict", None)
        sampler = Sampler(**sampler_kwargs)

    run_kwargs: Dict[str, Any] = dict(
        f_live=f_live,
        n_shell=n_shell,
        discard_exploration=discard_exploration,
        verbose=verbose,
    )
    if n_eff is not None:
        run_kwargs["n_eff"] = n_eff
    if n_like_max is not None:
        run_kwargs["n_like_max"] = n_like_max
    if timeout is not None:
        run_kwargs["timeout"] = timeout

    _ = sampler.run(**run_kwargs)

    # Extract posterior samples
    try:
        points, log_w, log_l = sampler.posterior(return_as_dict=True, equal_weight=equal_weight)
    except TypeError:
        # Compatibility fallback for older versions lacking return_as_dict.
        points, log_w, log_l = sampler.posterior(equal_weight=equal_weight)
        if not isinstance(points, dict):
            arr = np.asarray(points, dtype=np.float64)
            points = {name: arr[:, i] for i, name in enumerate(param_names)}

    n_samples = len(next(iter(points.values()))) if len(points) > 0 else 0

    samples_dict: Dict[str, np.ndarray] = {}
    for name in param_names:
        samples_dict[name] = np.asarray(points[name], dtype=np.float64)

    # Add fixed/delta parameters
    for p in cfg.params:
        if str(getattr(p, "dist", "")).lower() == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is not None:
                samples_dict[p.name] = np.full((n_samples,), float(val), dtype=np.float64)

    # Add optional defaults if active
    cfg_names = {p.name for p in cfg.params}
    if "c" not in cfg_names and "c" not in samples_dict:
        samples_dict["c"] = np.full((n_samples,), -99.0, dtype=np.float64)

    evidence_info: Dict[str, Any] = {
        "logZ": float(getattr(sampler, "log_z", np.nan)),
        "logZ_err": np.nan,
        "ESS": float(getattr(sampler, "n_eff", np.nan)),
        "n_like": int(getattr(sampler, "n_like", 0)) if hasattr(sampler, "n_like") else np.nan,
        "n_samples": int(n_samples),
        "sampler": "nautilus",
        "n_live": n_live,
        "f_live": f_live,
        "n_shell": n_shell,
        "checkpoint_file": filepath_use,
    }

    if verbose:
        print(f"[Nautilus] Evidence: {evidence_info['logZ']:.3f}")
        print(f"[Nautilus] Posterior samples: {n_samples}")

    return samples_dict, evidence_info
