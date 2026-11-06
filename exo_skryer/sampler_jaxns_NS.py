"""
sampler_jaxns_NS.py
===================

JAXNS nested sampler driver with:
- Stable theta_map keys (sampled + deltas + optional defaults injected)
- Optional "latent logit" parameterisation for bounded uniforms (future gradient-guided friendly)
- Silent defaults (e.g. c=-99) that only apply if the parameter is NOT present in YAML at all
- Same Gaussian likelihood + jitter as before
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
from pathlib import Path
import math

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.model import Model
from jaxns.priors import Prior
from .limb_asymmetry import jitter_param_name, parse_offset_group_name

tfpd = tfp.distributions

# jaxns 2 options with no jaxns 3 equivalent; reported and ignored if set.
_V2_ONLY_OPTIONS = (
    "difficult_model",
    "parameter_estimation",
    "shell_fraction",
    "gradient_guided",
    "init_efficiency_threshold",
)

__all__ = [
    "make_jaxns_model",
    "run_nested_jaxns",
]

LOG_FLOOR = -1e300  # finite invalid logL for numerical stability

# jaxns pickles the model into its checkpoints, but the forward-model closure
# cannot be pickled. Checkpoints store this module-level function by name, and
# make_jaxns_model() points it at the model rebuilt in the current process.
_active_prior_model = None


def _prior_model() -> jnp.ndarray:
    return _active_prior_model()
OFFSET_VECTOR_KEY = "__offset_values__"


def _extract_offset_params(cfg, obs: dict) -> Tuple[list, jnp.ndarray, bool]:
    """
    Extract offset parameters and build mapping to data points.

    Returns
    -------
    offset_param_names : list
        Names of offset parameters in order (matching offset_group_names).
    offset_group_idx : jnp.ndarray
        Integer array mapping each data point to its offset parameter index.
        Uses -1 for points with no offset applied.
    has_offsets : bool
        Whether any offset parameters are defined.
    """
    from typing import Dict
    group_names = obs.get("offset_group_names", np.array(["__no_offset__"]))
    group_idx = obs.get("offset_group_idx", np.zeros(len(obs["y"]), dtype=int))

    # Find offset parameters by naming convention "offset_<group_name>"
    param_map: Dict[str, str] = {}  # group_name -> param_name
    for p in cfg.params:
        name = p.name
        group_name = parse_offset_group_name(name, getattr(getattr(cfg, "physics", None), "rt_scheme", None))
        if group_name is not None:
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


def _sigmoid(x):
    """Custom sigmoid: 1 / (1 + exp(-x)), avoids jax.nn import overhead."""
    return 1.0 / (1.0 + jnp.exp(-x))


def make_jaxns_model(cfg, obs: dict, fm) -> Model:
    """
    Build a JAXNS Model from cfg + obs + fm.

    Improvements vs prior version:
      - delta params injected into theta_map inside prior_model (stable keys)
      - optional defaults injected only if not explicitly specified in cfg.params (stable keys)
      - bounded uniforms can use "transform: logit" -> latent logistic z, mapped by sigmoid
        Note: if z ~ Logistic(0,1), then sigmoid(z) ~ Uniform(0,1) exactly (CDF transform),
        so this preserves the intended uniform prior while living in an unconstrained latent.
      - supports instrument offset parameters (offset_<group> in ppm)
    """

    # --- observed data closed over in the likelihood ---
    y_obs = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

    # --- extract offset parameters ---
    offset_param_names, offset_group_idx, has_offsets = _extract_offset_params(cfg, obs)

    # ----------------------------
    # Build delta injection dict
    # ----------------------------
    delta_dict: Dict[str, float] = {}
    for p in cfg.params:
        dist_name = str(getattr(p, "dist", "")).lower()
        if dist_name == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is None:
                raise ValueError(f"Delta parameter '{p.name}' needs 'value' or 'init'.")
            delta_dict[p.name] = float(val)

    # ----------------------------
    # Select sampled parameters
    # ----------------------------
    params_cfg = [p for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]

    # Silent defaults that only apply if the name is NOT present in YAML at all.
    jitter_key = jitter_param_name(getattr(getattr(cfg, "physics", None), "rt_scheme", None))
    OPTIONAL_DEFAULTS: Dict[str, float] = {
        jitter_key: -99.0,
    }
    cfg_names = {p.name for p in cfg.params}
    optional_defaults_active = {k: v for k, v in OPTIONAL_DEFAULTS.items() if k not in cfg_names}

    # Guard against name collisions: cannot be both sampled and delta
    sampled_names = {p.name for p in params_cfg}
    overlap = sampled_names.intersection(delta_dict.keys())
    if overlap:
        raise ValueError(f"Parameter(s) appear both as sampled and delta: {sorted(overlap)}")

    # ----- realise_params: sample jaxns.Prior objects into a parameter dict -----
    def realise_params() -> Dict[str, Any]:
        """
        Realise the sampled priors and return the dict of parameters passed
        to the likelihood.

        Returned dict ALWAYS contains:
          - all sampled params (physical names)
          - all delta params (physical names)
          - all optional defaults not specified in YAML
        """
        params: Dict[str, Any] = {}

        for p in params_cfg:
            name = getattr(p, "name", None)
            if not name:
                raise ValueError("Each param in cfg.params needs a 'name'.")

            dist_name = str(getattr(p, "dist", "")).lower()
            if not dist_name:
                raise ValueError(f"Parameter '{name}' needs a 'dist' field.")

            # ----- sampled priors -----
            if dist_name == "uniform":
                low = float(getattr(p, "low"))
                high = float(getattr(p, "high"))
                transform = str(getattr(p, "transform", "identity")).lower()

                if transform == "logit":
                    # Unconstrained latent + smooth map to (low, high)
                    # z ~ Logistic(0,1) => sigmoid(z) ~ Uniform(0,1) exactly
                    z = Prior(tfpd.Logistic(loc=0.0, scale=1.0), name=f"{name}__z").realise()
                    t = _sigmoid(z)  # sigmoid output is strictly in (0,1) for finite inputs
                    theta = low + (high - low) * t
                else:
                    theta = Prior(tfpd.Uniform(low=low, high=high), name=name).realise()

            elif dist_name in ("gaussian", "normal"):
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                theta = Prior(tfpd.Normal(loc=mu, scale=sigma), name=name).realise()

            elif dist_name == "lognormal":
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                theta = Prior(tfpd.LogNormal(loc=mu, scale=sigma), name=name).realise()

            else:
                raise ValueError(f"Unsupported dist '{dist_name}' for param '{name}' in JAXNS model.")

            params[name] = theta

        # Inject delta params (fixed)
        for k, v in delta_dict.items():
            params[k] = jnp.asarray(v)

        # Inject optional defaults only if not explicitly configured in YAML
        for k, v in optional_defaults_active.items():
            if k not in params:
                params[k] = jnp.asarray(v)

        if has_offsets:
            params[OFFSET_VECTOR_KEY] = jnp.stack([params[name] for name in offset_param_names])

        return params

    # ----- Gaussian (symmetric) log-likelihood with offset support -----
    @jax.jit
    def log_likelihood(theta_map: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        mu = fm(theta_map)  # (N,)
        valid_mu = jnp.all(jnp.isfinite(mu))

        def invalid_ll(_):
            return jnp.asarray(LOG_FLOOR)

        def valid_ll(_):
            # Apply instrument offsets if defined (offset params are in ppm)
            if has_offsets:
                offset_values = theta_map[OFFSET_VECTOR_KEY]
                idx_safe = jnp.clip(offset_group_idx, 0, offset_values.shape[0] - 1)
                mask = (offset_group_idx >= 0).astype(y_obs.dtype)
                offset_vec = (offset_values[idx_safe] / 1e6) * mask  # ppm -> fractional
                y_shifted = y_obs + offset_vec  # positive offset shifts data UP
            else:
                y_shifted = y_obs

            r = y_shifted - mu  # (N,)

            # 'c' is guaranteed present: either in YAML (sampled/delta) or injected default
            c = theta_map[jitter_key]  # log10(sigma_jit)

            # sigma_jit^2 = 10^(2c)
            sig_jit2 = 10.0 ** (2.0 * c)

            sig_eff = jnp.sqrt(dy_obs**2 + sig_jit2)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = -jnp.log(sig_eff) - 0.5 * jnp.log(2.0 * jnp.pi)
            ll = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)

            return jnp.where(jnp.isfinite(ll), ll, jnp.asarray(LOG_FLOOR))

        return jax.lax.cond(valid_mu, valid_ll, invalid_ll, operand=None)

    # jaxns 3: the prior model realises the priors and returns the log-likelihood.
    def prior_model() -> jnp.ndarray:
        return log_likelihood(realise_params())

    global _active_prior_model
    _active_prior_model = prior_model
    return Model(prior_model=_prior_model)


def run_nested_jaxns(
    cfg,
    obs: dict,
    fm,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Full-featured JAXNS driver.

    Called as:
        samples_dict, evidence_info = run_nested_jaxns(cfg, obs, fm, exp_dir)
    """
    jcfg = cfg.sampling.jaxns

    # ---- core NS configuration ----
    max_samples = getattr(jcfg, "max_samples", None)
    if max_samples is not None:
        max_samples = int(max_samples)
    num_live_points = getattr(jcfg, "num_live_points", None)
    s = getattr(jcfg, "s", None)
    k = int(getattr(jcfg, "k", None) or 0)
    c = getattr(jcfg, "c", None)
    verbose = bool(getattr(jcfg, "verbose", False))
    allocation_target = str(getattr(jcfg, "allocation_target", None) or "uniform")
    replacement_width = getattr(jcfg, "replacement_width", None)
    if replacement_width is not None:
        replacement_width = int(replacement_width)

    # A checkpoint found in checkpoint_dir is resumed automatically.
    checkpoint_dir = getattr(jcfg, "checkpoint_dir", None)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = exp_dir / checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cadence = float(getattr(jcfg, "checkpoint_cadence", None) or 3600.0)

    posterior_samples = int(getattr(jcfg, "posterior_samples", 5000))
    seed = int(getattr(jcfg, "seed", 0))

    ignored = [n for n in _V2_ONLY_OPTIONS if getattr(jcfg, n, None) is not None]
    if ignored:
        print(f"[JAXNS] Ignoring options removed in jaxns 3: {ignored}")

    # jaxns 2's c (number of parallel Markov chains) is jaxns 3's
    # root_allocation_degree; jaxns 2 derived c = num_live_points // (k + 1).
    if c is not None:
        root_degree = int(c)
    elif num_live_points is not None:
        root_degree = max(1, int(num_live_points) // (k + 1))
    else:
        root_degree = None  # jaxns default: 30 * D

    # ---- build JAXNS model from cfg + obs + fm ----
    model = make_jaxns_model(cfg, obs, fm)

    # Get parameter names for printing
    param_names = [p.name for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]

    # s = slices per dimension (jaxns default 5 per dimension)
    sampler = None
    if s is not None:
        sampler = UniDimSliceSampler(num_slices=max(1, int(s) * len(param_names)))

    # ---- termination condition ----
    term_cfg = getattr(jcfg, "termination", None)

    def _term(name):
        return getattr(term_cfg, name, None) if term_cfg is not None else None

    # jaxns 2 dlogZ = log(1 + Z_remaining / Z_current); jaxns 3 takes the
    # remaining-evidence fraction Z_remaining / (Z_current + Z_remaining).
    depth_condition = None
    if _term("dlogZ") is not None:
        depth_condition = DepthCondition(dlogZ=-math.expm1(-float(_term("dlogZ"))))
    if _term("max_samples") is not None:
        term_max = int(_term("max_samples"))
        max_samples = term_max if max_samples is None else min(max_samples, term_max)
    ess_target = _term("ess")
    uncert_target = _term("evidence_uncert")
    max_like = _term("max_num_likelihood_evaluations")
    unsupported = [n for n in ("rtol", "atol") if _term(n) is not None]
    if unsupported:
        print(f"[JAXNS] Ignoring termination options removed in jaxns 3: {unsupported}")

    print(f"[JAXNS] Running nested sampling...")
    print(f"[JAXNS] Free parameters: {len(param_names)}")
    print(f"[JAXNS] Parameter names: {param_names}")
    print(f"[JAXNS] root_allocation_degree: {root_degree if root_degree is not None else 'default (30*D)'}")
    print(f"[JAXNS] max_samples: {max_samples}")
    print(f"[JAXNS] phantom samples: {k > 0}")
    print(f"[JAXNS] allocation_target: {allocation_target}")
    print(f"[JAXNS] replacement_width: {replacement_width if replacement_width is not None else 'default (10*D)'}")
    print(f"[JAXNS] checkpoint_dir: {checkpoint_dir}")

    ns = NestedSampler(
        model=model,
        root_allocation_degree=root_degree,
        max_samples=max_samples,
        sampler=sampler,
        depth_condition=depth_condition,
        collect_phantom_samples=k > 0,
        replacement_width=replacement_width,
        allocation_target=allocation_target,
        verbose=verbose,
    )
    checkpoint = {"checkpoint_dir": checkpoint_dir, "checkpoint_cadence": checkpoint_cadence}

    def report(st) -> Any:
        res = st.to_result()
        if verbose:
            print(
                f"[JAXNS] samples={int(st.num_samples)} "
                f"likelihood_evals={int(st.total_num_likelihood_evaluations)} "
                f"logZ={float(res.log_Z_mean):.3f}+-{float(res.log_Z_uncert):.3f} "
                f"ESS={float(res.ess):.0f}",
                flush=True,
            )
        return res

    # ---- run nested sampler ----
    key = jax.random.key(seed)
    run_key, post_key = jax.random.split(key)
    state = ns.run(key=run_key, **checkpoint)

    # ESS / evidence-uncertainty / likelihood-budget targets are a Python goal
    # in jaxns 3: keep extending the run until they are met.
    if ess_target is not None or uncert_target is not None or max_like is not None:

        def goal(st) -> bool:
            if int(st.termination_reason) != 0:
                return True
            res = report(st)
            if max_like is not None and int(st.total_num_likelihood_evaluations) >= int(max_like):
                return True
            if ess_target is not None and float(res.ess) < float(ess_target):
                return False
            if uncert_target is not None and float(res.log_Z_uncert) > float(uncert_target):
                return False
            return True

        if not goal(state):
            state = ns.resume_until_goal(state, goal, **checkpoint)
    else:
        report(state)

    results = state.to_result().trim()
    term_reason = int(results.termination_reason)

    print(f"JAXNS termination_reason: {term_reason}")
    if term_reason != 0:
        print(f"[JAXNS] Warning: stopped by max_samples ({max_samples}), before the termination targets were met.")

    # Save full results to experiment directory
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_file = exp_dir / "jaxns_results.json"
    results.save(str(results_file))
    results.summary()

    # ---- evidence info ----
    evidence_info: Dict[str, Any] = {
        "logZ": float(results.log_Z_mean),
        "logZ_err": float(results.log_Z_uncert),
        "ESS": float(results.ess),
        "H_mean": float(results.H_mean),
        "n_samples": int(results.total_num_samples),
        "n_like": int(results.total_num_likelihood_evaluations),
        "termination_reason": str(term_reason),
        "results_file": str(results_file),
    }

    # ---- posterior resampling to equal-weight samples ----
    eq_samples = results.resample(posterior_samples, key=post_key).X_samples

    # ---- convert to samples_dict format ----
    samples_dict: Dict[str, np.ndarray] = {}

    # Bayesian parameters: handle logit-transformed and regular parameters
    for p in cfg.params:
        name = p.name
        dist_name = str(getattr(p, "dist", "")).lower()
        transform = str(getattr(p, "transform", "identity")).lower()

        # Skip delta parameters (handled separately below)
        if dist_name == "delta":
            continue

        # Handle logit-transformed uniform parameters
        if transform == "logit" and dist_name == "uniform":
            low = float(getattr(p, "low"))
            high = float(getattr(p, "high"))
            latent_name = f"{name}__z"  # Matches the prior_model naming

            # Check if latent variable is in samples
            if latent_name in eq_samples:
                # Retransform from latent to physical parameter
                z = np.asarray(eq_samples[latent_name])
                t = 1.0 / (1.0 + np.exp(-z))
                t = np.clip(t, 1e-12, 1.0 - 1e-12)
                samples_dict[name] = low + (high - low) * t
            elif name in eq_samples:
                # JAXNS might store transformed values directly
                samples_dict[name] = np.asarray(eq_samples[name])
            else:
                print(f"[Warning] Parameter '{name}' not found in samples (checked '{name}' and '{latent_name}')")
        else:
            # Regular parameters (no transform)
            if name in eq_samples:
                samples_dict[name] = np.asarray(eq_samples[name])
            else:
                print(f"[Warning] Parameter '{name}' not found in samples")

    # Fixed / delta parameters
    for p in cfg.params:
        name = p.name
        if name not in samples_dict:
            dist_name = str(getattr(p, "dist", "")).lower()
            if dist_name == "delta":
                val = getattr(p, "value", getattr(p, "init", None))
                if val is not None:
                    samples_dict[name] = np.full(
                        (posterior_samples,),
                        float(val),
                        dtype=np.float64,
                    )

    return samples_dict, evidence_info
