"""
sampler_polychord_NS.py
========================

PolyChord nested sampler driver with JAX forward model integration.

Updates in this version
-----------------------
- FIX: correct parsing of *_equal_weights.txt (no off-by-one; uses last nDims columns)
- FIX: correct offset-group mapping (supports "__no_offset__" without mis-assigning to first offset)
- SAFETY: avoid importing mpi4py (macOS segfaults during MPI.Initialize are common when mismatched)
- BETTER DEFAULTS: num_repeats defaults to ~5*nDims (instead of nlive)
- ROBUST: derived-array handling (phi may be None)
- COMPAT: prior transform writes in-place (safe across PolyChordLite variants)
- NOTE: does NOT set JAX float64; assumes you enable x64 elsewhere as requested.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Callable
from pathlib import Path
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp

try:
    import pypolychord
    from pypolychord.settings import PolyChordSettings

    POLYCHORD_AVAILABLE = True
except ImportError:
    POLYCHORD_AVAILABLE = False
    pypolychord = None
    PolyChordSettings = None


__all__ = [
    "build_prior_transform_polychord",
    "build_loglikelihood_polychord",
    "run_nested_polychord",
]


LOG_FLOOR = -1e300  # finite "invalid" logL for PolyChord stability


def _mpi_rank_size_from_env() -> Tuple[int, int, bool]:
    """Detect MPI rank/size from environment without importing mpi4py."""
    # OpenMPI
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
        return rank, size, size > 1

    # MPICH / PMI-based launchers
    if "PMI_SIZE" in os.environ:
        size = int(os.environ["PMI_SIZE"])
        rank = int(os.environ.get("PMI_RANK", "0"))
        return rank, size, size > 1

    # Slurm
    if "SLURM_NTASKS" in os.environ:
        size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        return rank, size, size > 1

    return 0, 1, False


def _extract_offset_params(cfg, obs: dict) -> Tuple[List[str], jnp.ndarray, bool]:
    """
    Extract offset parameters and build mapping to data points.

    Supports an explicit "__no_offset__" group that maps to "no correction".

    Returns
    -------
    offset_param_names : List[str]
        Names of offset parameters, ordered by appearance in obs offset groups
        (excluding "__no_offset__").
    offset_point_idx : jnp.ndarray
        Integer array mapping each data point -> index into offset_param_names,
        with -1 meaning "no offset applied".
    has_offsets : bool
        True if at least one real offset group exists and has a matching parameter.
    """
    group_names = list(obs.get("offset_group_names", ["__no_offset__"]))
    group_idx = np.asarray(obs.get("offset_group_idx", np.zeros(len(obs["y"]), dtype=int)))

    # group_name -> parameter name (offset_<group>)
    group_to_param: Dict[str, str] = {}
    for p in cfg.params:
        name = p.name
        if name.startswith("offset_"):
            group_to_param[name[7:]] = name

    real_groups = [g for g in group_names if g != "__no_offset__"]
    if len(real_groups) == 0:
        return [], jnp.full(len(obs["y"]), -1, dtype=jnp.int32), False

    # Validate: every real group in data must have a corresponding offset parameter
    for g in real_groups:
        if g not in group_to_param:
            raise ValueError(
                f"Offset group '{g}' found in data but no 'offset_{g}' parameter defined."
            )

    # Keep a stable order for offset parameters: order of real_groups in group_names
    offset_param_names = [group_to_param[g] for g in real_groups]
    group_to_param_idx = {g: i for i, g in enumerate(real_groups)}

    # Build point -> param index (-1 means no offset)
    point_idx = np.full_like(group_idx, fill_value=-1)
    for gi, g in enumerate(group_names):
        if g == "__no_offset__":
            continue
        point_idx[group_idx == gi] = group_to_param_idx[g]

    return offset_param_names, jnp.asarray(point_idx, dtype=jnp.int32), True


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


def build_prior_transform_polychord(cfg, errlog_path: Path | None = None):
    from scipy.special import ndtri

    params_cfg = [p for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]
    param_names = [p.name for p in params_cfg]

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

    printed = {"done": False}

    def prior_transform(cube: np.ndarray) -> np.ndarray:
        # MUST NEVER throw.
        try:
            eps = 1e-12
            theta = np.empty_like(cube, dtype=np.float64)

            for i, info in enumerate(param_info):
                u = float(np.clip(cube[i], eps, 1.0 - eps))
                dist_name = info["dist"]

                if dist_name == "uniform":
                    lo = float(info["low"]); hi = float(info["high"])
                    theta[i] = lo + u * (hi - lo)
                elif dist_name in ("gaussian", "normal"):
                    mu = float(info["mu"]); sig = float(info["sigma"])
                    theta[i] = mu + sig * ndtri(u)
                elif dist_name == "lognormal":
                    mu = float(info["mu"]); sig = float(info["sigma"])
                    theta[i] = np.exp(mu + sig * ndtri(u))

            cube[:] = theta
            return cube

        except Exception as e:
            # Log once (to file), then fall back to mid-prior / safe values
            if (not printed["done"]) and (errlog_path is not None):
                printed["done"] = True
                _log_first_exception_to_file(
                    errlog_path,
                    header="[PolyChord prior] Exception in prior_transform (first occurrence)",
                    exc=e,
                )

            # Safe fallback: map to 0.5 in unit cube -> midpoints
            # (This keeps PolyChord alive and gets you a traceback to inspect.)
            try:
                cube[:] = 0.5
            except Exception:
                pass
            return cube

    return prior_transform, param_names

import traceback
from pathlib import Path

def _log_first_exception_to_file(path: Path, header: str, exc: BaseException) -> None:
    """
    Append a traceback to a file (once per process). Safe to call from callbacks.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(header + "\n")
            f.write(f"{type(exc).__name__}: {exc}\n")
            f.write("Traceback:\n")
            f.write("".join(traceback.format_exc()))
            f.flush()
    except Exception:
        # Never raise from the logger
        pass


def build_loglikelihood_polychord(cfg, obs: dict, fm, param_names):
    """
    Robust PolyChord log-likelihood wrapper:
    - Never lets Python exceptions escape into PolyChord (prevents abort traps)
    - Safe offset handling with -1 indices
    - Keeps float dtype controlled by your global JAX config
    """

    y_obs = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

    # Offsets (uses your earlier _extract_offset_params that returns -1 for no-offset)
    offset_param_names, offset_point_idx, has_offsets = _extract_offset_params(cfg, obs)

    # Delta params: require a value/init (fail early with a clear error)
    delta_dict = {}
    for p in cfg.params:
        if str(getattr(p, "dist", "")).lower() == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is None:
                raise ValueError(f"Delta parameter '{p.name}' has no value/init.")
            delta_dict[p.name] = float(val)

    OPTIONAL_DEFAULTS = {"c": -99.0}
    cfg_names = {p.name for p in cfg.params}
    optional_defaults_active = {k: v for k, v in OPTIONAL_DEFAULTS.items() if k not in cfg_names}

    def _vec_to_theta_dict(theta_vec: jnp.ndarray):
        d = {name: theta_vec[i] for i, name in enumerate(param_names)}
        for k, v in delta_dict.items():
            d[k] = jnp.asarray(v, dtype=theta_vec.dtype)
        for k, v in optional_defaults_active.items():
            if k not in d:
                d[k] = jnp.asarray(v, dtype=theta_vec.dtype)
        return d

    def _safe_apply_offsets(params, y):
        """Apply per-point offsets with idx=-1 meaning no offset. Offsets are in ppm."""
        if not has_offsets:
            return y

        # (n_off,)
        offset_values = jnp.array([params[n] for n in offset_param_names], dtype=y.dtype)

        # idx in [-1, n_off-1]
        idx = offset_point_idx  # (N,)

        # Avoid negative indexing / eager evaluation:
        # clamp indices into [0, n_off-1], then mask out where idx<0
        n_off = offset_values.shape[0]
        idx_safe = jnp.clip(idx, 0, n_off - 1)
        mask = (idx >= 0).astype(y.dtype)

        offset_vec = (offset_values[idx_safe] / 1e6) * mask
        return y - offset_vec

    @jax.jit
    def loglike_jax(theta_vec: jnp.ndarray) -> jnp.ndarray:
        params = _vec_to_theta_dict(theta_vec)

        mu = fm(params)  # must be (N,)
        # If fm returns wrong shape, this can trigger errors later; catch via valid checks.
        valid_mu = jnp.all(jnp.isfinite(mu)) & (mu.shape[0] == y_obs.shape[0])

        def valid_ll(_):
            y_shifted = _safe_apply_offsets(params, y_obs)
            r = y_shifted - mu

            c = params["c"]  # log10(sigma_jit)
            sig_jit2 = 10.0 ** (2.0 * c)

            sig_eff = jnp.sqrt(dy_obs**2 + sig_jit2)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = -jnp.log(sig_eff) - 0.5 * jnp.log(2.0 * jnp.pi)
            ll = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)

            return jnp.where(jnp.isfinite(ll), ll, jnp.asarray(LOG_FLOOR, dtype=ll.dtype))

        return jax.lax.cond(valid_mu, valid_ll, lambda _: jnp.asarray(LOG_FLOOR, dtype=y_obs.dtype), operand=None)

    # Warmup compile with prior-centered theta
    theta0 = _prior_center_theta0(cfg, param_names)
    _ = float(loglike_jax(jnp.asarray(theta0)))

    # Print the first exception only (prevents spam)
    printed_error = {"done": False}

    def loglikelihood(theta: np.ndarray, phi):
        if phi is None:
            phi = np.empty((0,), dtype=np.float64)

        try:
            theta_vec = jnp.asarray(theta)  # respect your global dtype policy
            ll = loglike_jax(theta_vec)
            val = float(ll)  # sync
            if not np.isfinite(val):
                return LOG_FLOOR, phi
            return val, phi

        except Exception as e:
            # NEVER allow Python exceptions to escape into PolyChord.
            if not printed_error["done"]:
                printed_error["done"] = True
                print("\n[PolyChord loglikelihood] EXCEPTION (showing first occurrence):")
                print(f"  type: {type(e).__name__}")
                print(f"  msg : {e}")
                print("  traceback:")
                print("".join(traceback.format_exc()))
                print("\n[PolyChord loglikelihood] Returning LOG_FLOOR for this and future exceptions.\n")
            return LOG_FLOOR, phi

    return loglikelihood


def _read_polychord_equal_weights(equal_weights_file: Path, nDims: int) -> np.ndarray:
    """
    Read PolyChord *_equal_weights.txt robustly.

    Many PolyChord variants produce a file with some metadata columns then parameters.
    The most robust approach is: parameters are the *last nDims columns*.
    """
    data = np.loadtxt(equal_weights_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < nDims:
        raise RuntimeError(
            f"Equal-weights file has {data.shape[1]} columns, but nDims={nDims}."
        )

    samples = data[:, -nDims:]
    return samples


def _parse_stats_for_logZ(stats_file: Path) -> Tuple[float, float]:
    logZ = np.nan
    logZ_err = np.nan
    if not stats_file.exists():
        return logZ, logZ_err

    with open(stats_file, "r") as f:
        for line in f:
            if "log(Z)" in line and "=" in line and "+/-" in line:
                # example: "log(Z)       =  -123.456 +/-   0.789"
                rhs = line.split("=", 1)[1]
                parts = rhs.split("+/-")
                if len(parts) == 2:
                    try:
                        logZ = float(parts[0].strip())
                        logZ_err = float(parts[1].strip())
                        break
                    except ValueError:
                        pass
    return logZ, logZ_err


def run_nested_polychord(
    cfg,
    obs: dict,
    fm: Callable,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run PolyChord nested sampling and return (samples_dict, evidence_info).
    """
    if not POLYCHORD_AVAILABLE:
        raise ImportError(
            "PolyChord is not installed. Install with:\n"
            "  pip install pypolychord\n"
            "Or install with Exo_Skryer:\n"
            "  pip install -e .[polychord]\n"
            "See: https://github.com/PolyChord/PolyChordLite"
        )

    pc_cfg = getattr(cfg.sampling, "polychord", None)
    if pc_cfg is None:
        raise ValueError("Missing cfg.sampling.polychord configuration.")

    # Build prior and likelihood first to know nDims
    prior_fn, param_names = build_prior_transform_polychord(cfg)
    nDims = len(param_names)
    nDerived = 0

    loglike_fn = build_loglikelihood_polychord(cfg, obs, fm, param_names)

    # Config parameters with defaults
    nlive = int(getattr(pc_cfg, "nlive", 500))

    # IMPORTANT: num_repeats should scale with dimensionality, not nlive
    num_repeats = getattr(pc_cfg, "num_repeats", None)
    if num_repeats is None:
        # Typical default: O(few * nDims)
        num_repeats = int(getattr(pc_cfg, "num_repeats_mult", 5) * nDims)
    else:
        num_repeats = int(num_repeats)

    nprior = int(getattr(pc_cfg, "nprior", -1))
    do_clustering = bool(getattr(pc_cfg, "do_clustering", True))
    feedback = int(getattr(pc_cfg, "feedback", 1))
    precision_criterion = float(getattr(pc_cfg, "precision_criterion", 0.001))
    max_ndead = int(getattr(pc_cfg, "max_ndead", -1))
    boost_posterior = float(getattr(pc_cfg, "boost_posterior", 0.0))
    read_resume = bool(getattr(pc_cfg, "read_resume", False))
    write_resume = bool(getattr(pc_cfg, "write_resume", True))
    write_live = bool(getattr(pc_cfg, "write_live", True))
    write_dead = bool(getattr(pc_cfg, "write_dead", True))
    write_stats = bool(getattr(pc_cfg, "write_stats", True))
    equals = bool(getattr(pc_cfg, "equals", True))
    compression_factor = float(getattr(pc_cfg, "compression_factor", np.exp(-1)))
    seed = int(getattr(pc_cfg, "seed", -1))

    # Ensure output directory exists
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Detect MPI without mpi4py (avoids macOS segfaults)
    rank, size, using_mpi = _mpi_rank_size_from_env()

    if rank == 0:
        print("[PolyChord] Running nested sampling...")
        if using_mpi:
            print(f"[PolyChord] MPI processes: {size}")
            print(f"[PolyChord] JAX backend: {jax.default_backend()}")
        print(f"[PolyChord] Free parameters: {nDims}")
        print(f"[PolyChord] Parameter names: {param_names}")
        print(f"[PolyChord] nlive: {nlive}")
        print(f"[PolyChord] num_repeats: {num_repeats}")
        print(f"[PolyChord] precision_criterion: {precision_criterion}")
        print(f"[PolyChord] do_clustering: {do_clustering}")

    # Create PolyChord settings
    settings = PolyChordSettings(nDims, nDerived)
    settings.base_dir = str(exp_dir)
    settings.file_root = "polychord"
    settings.nlive = nlive
    settings.num_repeats = num_repeats
    settings.nprior = nprior
    settings.do_clustering = do_clustering
    settings.feedback = feedback
    settings.precision_criterion = precision_criterion
    settings.max_ndead = max_ndead
    settings.boost_posterior = boost_posterior
    settings.read_resume = read_resume
    settings.write_resume = write_resume
    settings.write_live = write_live
    settings.write_dead = write_dead
    settings.write_stats = write_stats
    settings.equals = equals
    settings.compression_factor = compression_factor

    if seed >= 0:
        settings.seed = seed

    # Run PolyChord
    output = pypolychord.run_polychord(
        loglikelihood=loglike_fn,
        nDims=nDims,
        nDerived=nDerived,
        settings=settings,
        prior=prior_fn,
    )

    # Non-master ranks (when using true MPI runs) should exit quietly
    # PolyChord itself typically handles parallelism; we keep this safe.
    if rank != 0:
        return {}, {}

    print("[PolyChord] Sampling complete. Reading results...")

    stats_file = exp_dir / f"{settings.file_root}.stats"
    equal_weights_file = exp_dir / f"{settings.file_root}_equal_weights.txt"

    # Evidence
    logZ, logZ_err = _parse_stats_for_logZ(stats_file)

    # Samples
    if not equal_weights_file.exists():
        raise RuntimeError(f"PolyChord equal weights file not found: {equal_weights_file}")

    samples = _read_polychord_equal_weights(equal_weights_file, nDims=nDims)
    n_samples = samples.shape[0]

    print(f"[PolyChord] Evidence: {logZ:.6f} ± {logZ_err:.6f}")
    print(f"[PolyChord] Posterior samples: {n_samples}")

    evidence_info: Dict[str, Any] = {
        "logZ": logZ,
        "logZ_err": logZ_err,
        "ESS": float(n_samples),  # equal-weighted samples count
        "H": np.nan,
        "n_like": np.nan,
        "n_samples": int(n_samples),
        "sampler": "polychord",
        "nlive": int(nlive),
        "num_repeats": int(num_repeats),
        "precision_criterion": float(precision_criterion),
        "do_clustering": bool(do_clustering),
        "stats_file": str(stats_file),
        "equal_weights_file": str(equal_weights_file),
    }

    # Save full output object (useful for debugging / future parsing)
    output_path = exp_dir / "polychord_output.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    evidence_info["output_file"] = str(output_path)

    # Build samples dict
    samples_dict: Dict[str, np.ndarray] = {name: samples[:, i] for i, name in enumerate(param_names)}

    # Delta parameters (fixed values) -> replicate across samples
    for p in cfg.params:
        name = p.name
        if name not in samples_dict and str(getattr(p, "dist", "")).lower() == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is not None:
                samples_dict[name] = np.full((n_samples,), float(val), dtype=np.float64)

    return samples_dict, evidence_info
