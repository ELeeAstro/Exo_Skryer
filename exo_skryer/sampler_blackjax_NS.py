"""
sampler_blackjax_NS.py
======================

BlackJAX nested sampling driver with:
- Particles hold only the sampled parameters; deltas and defaults join in the likelihood
- Optional logit latent parameterisation for bounded uniforms (via distrax.Transformed)
- Same Gaussian likelihood + jitter as JAXNS/dynesty/ultranest
"""

from __future__ import annotations
from functools import partial
from typing import Dict, Any, Tuple, List
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
from blackjax.ns.base import init_state_strategy
from blackjax.ns.nss import init as ns_init, live_covariance_factor
from anesthetic import NestedSamples
import distrax
from .build_common import bind_runtime, runtime_args_of
from .limb_asymmetry import jitter_param_name, parse_offset_group_name

__all__ = [
    "build_joint_prior_distrax",
    "run_nested_blackjax",
]

LOG_FLOOR = -1e300  # finite invalid logL for numerical stability
OFFSET_VECTOR_KEY = "__offset_values__"


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


def _collect_delta_and_defaults(cfg) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Delta params
    delta_dict: Dict[str, float] = {}
    for p in cfg.params:
        if str(getattr(p, "dist", "")).lower() == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is None:
                raise ValueError(f"Delta parameter '{p.name}' needs 'value' or 'init'.")
            delta_dict[p.name] = float(val)

    # Optional defaults only if NOT present anywhere in YAML
    jitter_key = jitter_param_name(getattr(getattr(cfg, "physics", None), "rt_scheme", None))
    OPTIONAL_DEFAULTS: Dict[str, float] = {jitter_key: -99.0}
    cfg_names = {p.name for p in cfg.params}
    defaults_active = {k: v for k, v in OPTIONAL_DEFAULTS.items() if k not in cfg_names}

    return delta_dict, defaults_active


def build_joint_prior_distrax(cfg) -> Tuple[distrax.Distribution, List[str]]:
    """
    Build a joint Distrax distribution over all non-delta parameters.

    NOTE: If transform == "logit" for a bounded uniform, we sample z ~ Logistic(0,1)
    and map with sigmoid -> affine to get a (near-)uniform variable in (low, high).
    """
    distributions: Dict[str, distrax.Distribution] = {}
    param_names: List[str] = []

    for param in cfg.params:
        dist = str(getattr(param, "dist", "")).lower()
        if dist == "delta":
            continue

        name = param.name
        param_names.append(name)

        if dist == "uniform":
            low = float(param.low)
            high = float(param.high)
            transform = str(getattr(param, "transform", "identity")).lower()

            if transform == "logit":
                # z ~ Logistic, then sigmoid(z) in (0,1), then affine -> (low, high)
                # distrax.Chain applies bijectors right-to-left
                bijector = distrax.Chain([
                    distrax.ScalarAffine(shift=low, scale=(high - low)),  # last
                    distrax.Sigmoid(),                                    # first
                ])
                distributions[name] = distrax.Transformed(
                    distrax.Logistic(loc=0.0, scale=1.0),
                    bijector,
                )
            else:
                distributions[name] = distrax.Uniform(low=low, high=high)

        elif dist in ("normal", "gaussian"):
            distributions[name] = distrax.Normal(loc=float(param.mu), scale=float(param.sigma))

        elif dist == "lognormal":
            # x = exp(N(mu, sigma))
            distributions[name] = distrax.Transformed(
                distrax.Normal(loc=float(param.mu), scale=float(param.sigma)),
                distrax.Exp(),  # use built-in bijector when available
            )

        elif dist == "beta":
            distributions[name] = distrax.Beta(
                concentration1=float(param.alpha),
                concentration0=float(param.beta),
            )

        elif dist == "gamma":
            distributions[name] = distrax.Gamma(
                concentration=float(param.concentration),
                rate=float(param.rate),
            )

        else:
            raise ValueError(f"Unknown distribution: {dist} for parameter {name}")

    prior = distrax.Joint(distributions)
    return prior, param_names


def _chunked_update(constrained_mcmc_step_fn, num_mcmc_steps, num_delete, batch_size):
    """
    blackjax.ns.from_mcmc.update_with_mcmc_take_last, with its vmap over the
    replaced particles swapped for a batched lax.map to bound peak memory.
    """

    def update_function(rng_key, state, loglikelihood_0, **step_parameters):
        choice_key, sample_key = jax.random.split(rng_key)
        particles = state.particles

        # Start from survivors
        weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
        weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
        start_idx = jax.random.choice(
            choice_key,
            len(weights),
            shape=(num_delete,),
            p=weights / weights.sum(),
            replace=True,
        )
        start_state = jax.tree.map(lambda x: x[start_idx], particles)

        def step(particle, key):
            return constrained_mcmc_step_fn(
                key, particle, loglikelihood_0=loglikelihood_0, **step_parameters
            )

        def mcmc_kernel(inputs):
            key, particle = inputs
            return jax.lax.scan(step, particle, jax.random.split(key, num_mcmc_steps))

        sample_keys = jax.random.split(sample_key, num_delete)
        new_particles, infos = jax.lax.map(mcmc_kernel, (sample_keys, start_state), batch_size=batch_size)
        new_particles = new_particles._replace(
            loglikelihood_birth=jnp.full_like(new_particles.loglikelihood_birth, loglikelihood_0)
        )
        return new_particles, infos

    return update_function


def run_nested_blackjax(cfg, obs: dict, fm, exp_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    ns_cfg = getattr(cfg.sampling, "blackjax_ns", None)
    if ns_cfg is None:
        raise ValueError("Missing cfg.sampling.blackjax_ns configuration.")

    num_live = int(getattr(ns_cfg, "num_live_points", 500))
    num_inner_steps = int(getattr(ns_cfg, "num_inner_steps", 20))
    num_delete = int(getattr(ns_cfg, "num_delete", max(1, num_live // 2)))
    seed = int(getattr(ns_cfg, "seed", 0))
    dlogz_stop = float(getattr(ns_cfg, "dlogz_stop", 0.0))
    jit_enable = bool(getattr(ns_cfg, "jit", True))
    # Particles evaluated together; 0 = all at once. Sets peak memory, not the result.
    batch_size = int(getattr(ns_cfg, "batch_size", 32))
    posterior_samples = int(getattr(ns_cfg, "posterior_samples", 5000))

    exp_dir.mkdir(parents=True, exist_ok=True)

    delta_dict, defaults_active = _collect_delta_and_defaults(cfg)
    jitter_key = jitter_param_name(getattr(getattr(cfg, "physics", None), "rt_scheme", None))

    # Build joint prior
    prior, param_names = build_joint_prior_distrax(cfg)

    print(f"[blackjax_ns] Running nested sampling...")
    print(f"[blackjax_ns] Free parameters: {len(param_names)}")
    print(f"[blackjax_ns] Parameter names: {param_names}")
    print(f"[blackjax_ns] num_live_points: {num_live}")
    print(f"[blackjax_ns] num_inner_steps: {num_inner_steps}")
    print(f"[blackjax_ns] num_delete: {num_delete}")
    print(f"[blackjax_ns] dlogz_stop: {dlogz_stop}")
    print(f"[blackjax_ns] jit: {jit_enable}")
    print(f"[blackjax_ns] batch_size: {batch_size}")

    if bool(getattr(cfg.opac, "full_grid", False)):
        print(
            "[blackjax_ns][warning] cfg.opac.full_grid=True can make BlackJAX NS very memory-hungry, "
            "because likelihoods are vectorised over the live points; consider full_grid=False if you hit OOM."
        )

    # RNG and initial particles
    rng_key = jax.random.key(seed)
    rng_key, prior_key = jax.random.split(rng_key)

    particles = prior.sample(seed=prior_key, sample_shape=(num_live,))

    # Extract offset parameters (with_fixed needs these)
    offset_param_names, offset_group_idx, has_offsets = _extract_offset_params(cfg, obs)

    # Fixed values are added inside the likelihood, never to the particles: NSS
    # takes a Cholesky of the live-point covariance, and a zero-variance
    # coordinate makes that factor (so every slice direction) NaN.
    def with_fixed(params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        out = dict(params)
        for k, v in delta_dict.items():
            out[k] = jnp.asarray(v)
        for k, v in defaults_active.items():
            out.setdefault(k, jnp.asarray(v))
        if has_offsets:
            out[OFFSET_VECTOR_KEY] = jnp.stack([out[name] for name in offset_param_names])
        return out

    # Observations
    y_obs = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

    def loglikelihood_fn_impl(particle: Dict[str, jnp.ndarray], runtime) -> jnp.ndarray:
        """Log-likelihood of one particle; BlackJAX vectorises over the live points."""
        params = with_fixed(particle)
        mu = bind_runtime(fm, runtime)(params)
        valid_mu = jnp.all(jnp.isfinite(mu))

        def invalid_ll(_):
            return jnp.asarray(LOG_FLOOR)

        def valid_ll(_):
            # Apply instrument offsets if defined (offset params are in ppm)
            if has_offsets:
                offset_values = params[OFFSET_VECTOR_KEY]
                idx_safe = jnp.clip(offset_group_idx, 0, offset_values.shape[0] - 1)
                mask = (offset_group_idx >= 0).astype(y_obs.dtype)
                offset_vec = (offset_values[idx_safe] / 1e6) * mask  # ppm -> fractional
                y_shifted = y_obs + offset_vec  # positive offset shifts data UP
            else:
                y_shifted = y_obs

            r = y_shifted - mu

            # c is always present via with_fixed (or sampled from the YAML)
            c = params[jitter_key]
            sig_jit2 = 10.0 ** (2.0 * c)

            sig_eff = jnp.sqrt(dy_obs**2 + sig_jit2)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = -jnp.log(sig_eff) - 0.5 * jnp.log(2.0 * jnp.pi)
            ll = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)

            return jnp.where(jnp.isfinite(ll), ll, jnp.asarray(LOG_FLOOR))

        return jax.lax.cond(valid_mu, valid_ll, invalid_ll, operand=None)

    def logprior_fn_impl(particle: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        return prior.log_prob(particle)

    # The sampler is built inside the jitted functions so that the likelihood
    # sees runtime as a jit argument, not as captured arrays.
    def build_sampler(runtime):
        update = {"update_strategy": partial(_chunked_update, batch_size=batch_size)} if batch_size else {}
        return blackjax.nss(
            logprior_fn=logprior_fn_impl,
            loglikelihood_fn=partial(loglikelihood_fn_impl, runtime=runtime),
            num_delete=num_delete,
            num_inner_steps=num_inner_steps,
            **update,
        )

    def init_impl(positions, key, runtime):
        if not batch_size:
            return build_sampler(runtime).init(positions, key)
        init_state = partial(
            init_state_strategy,
            logprior_fn=logprior_fn_impl,
            loglikelihood_fn=partial(loglikelihood_fn_impl, runtime=runtime),
        )
        return ns_init(
            positions,
            init_state_fn=lambda p: jax.lax.map(init_state, p, batch_size=batch_size),
            update_inner_kernel_params_fn=live_covariance_factor,
            rng_key=key,
        )

    def step_impl(key, state, runtime):
        return build_sampler(runtime).step(key, state)

    init_fn = jax.jit(init_impl) if jit_enable else init_impl
    step_fn = jax.jit(step_impl) if jit_enable else step_impl
    runtime = runtime_args_of(fm)

    print(f"[blackjax_ns] Initializing with {num_live} live points...")
    rng_key, init_key = jax.random.split(rng_key)
    live = init_fn(particles, init_key, runtime)

    dead = []
    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while True:
            # dlogz = log(1 + Z_live / Z), the same criterion as dynesty and nautilus.
            dlogz = float(np.logaddexp(0.0, jax.device_get(live.integrator.logZ_live - live.integrator.logZ)))

            if np.isnan(dlogz):
                raise RuntimeError("[blackjax_ns] Evidence became NaN; stopping the run.")
            if dlogz <= dlogz_stop:
                break

            rng_key, subkey = jax.random.split(rng_key, 2)
            live, dead_info = step_fn(subkey, live, runtime)
            dead.append(dead_info)
            pbar.update(num_delete)

    dead = blackjax.ns.utils.finalise(live, dead)

    rng_key, weight_key, sample_key = jax.random.split(rng_key, 3)
    re_samples = blackjax.ns.utils.sample(sample_key, dead, shape=posterior_samples).position
    log_w = blackjax.ns.utils.log_weights(weight_key, dead, shape=100)
    ns_ess = blackjax.ns.utils.ess(sample_key, dead)
    logzs = jax.scipy.special.logsumexp(log_w, axis=0)

    logZ_mean = float(logzs.mean())
    logZ_std = float(logzs.std())
    ess_val = float(ns_ess)

    print(f"[blackjax_ns] ESS: {int(ess_val)}")
    print(f"[blackjax_ns] logZ estimate: {logZ_mean:.2f} ± {logZ_std:.2f}")

    # Labels
    def make_latex_label(name: str) -> str:
        parts = name.split("_")
        if len(parts) == 1:
            return rf"${name}$"
        elif len(parts) == 2:
            return rf"${parts[0]}_{{{parts[1]}}}$"
        base = parts[0]
        subscript = ",".join(parts[1:])
        return rf"${base}_{{{subscript}}}$"

    labels = {name: make_latex_label(name) for name in param_names}

    # blackjax >= 1.7: dead.particles is a StateWithLogLikelihood. Initial
    # particles have a NaN birth contour; anesthetic expects -inf for them.
    logL_birth = np.asarray(dead.particles.loglikelihood_birth)
    logL_birth = np.where(np.isnan(logL_birth), -np.inf, logL_birth)
    samples = NestedSamples(
        dead.particles.position,
        logL=dead.particles.loglikelihood,
        logL_birth=logL_birth,
        labels=labels,
    )

    csv_path = exp_dir / "nested_samples.csv"
    samples.to_csv(csv_path)
    print(f"[blackjax_ns] Saved nested samples to {csv_path}")

    evidence_info: Dict[str, Any] = {
        "logZ": logZ_mean,
        "logZ_err": logZ_std,
        "ESS": ess_val,
        "sampler": "blackjax_ns",
        "n_live": num_live,
        "num_inner_steps": num_inner_steps,
        "num_delete": num_delete,
        "dlogz_stop": dlogz_stop,
    }

    # Build samples_dict (physical + deltas)
    samples_dict: Dict[str, np.ndarray] = {}
    for name in param_names:
        samples_dict[name] = np.asarray(re_samples[name])

    # Add deltas
    n_samples = len(next(iter(re_samples.values())))
    for p in cfg.params:
        if str(getattr(p, "dist", "")).lower() == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is not None:
                samples_dict[p.name] = np.full((n_samples,), float(val), dtype=np.float64)

    # Optionally add default parameters to outputs
    for k, v in defaults_active.items():
        if k not in samples_dict:
            samples_dict[k] = np.full((n_samples,), float(v), dtype=np.float64)

    return samples_dict, evidence_info
