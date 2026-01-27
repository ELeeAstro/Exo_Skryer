"""
sampler_blackjax_NS.py
======================

BlackJAX nested sampling driver with:
- Stable parameter keys (sampled + deltas + optional defaults injected)
- Optional logit latent parameterisation for bounded uniforms (via distrax.Transformed)
- Same Gaussian likelihood + jitter as JAXNS/dynesty/ultranest
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import blackjax
from anesthetic import NestedSamples
import distrax

__all__ = [
    "build_joint_prior_distrax",
    "run_nested_blackjax",
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
    OPTIONAL_DEFAULTS: Dict[str, float] = {"c": -99.0}
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


def run_nested_blackjax(cfg, obs: dict, fm, exp_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    ns_cfg = getattr(cfg.sampling, "blackjax_ns", None)
    if ns_cfg is None:
        raise ValueError("Missing cfg.sampling.blackjax_ns configuration.")

    num_live = int(getattr(ns_cfg, "num_live_points", 500))
    num_inner_steps = int(getattr(ns_cfg, "num_inner_steps", 20))
    num_delete = int(getattr(ns_cfg, "num_delete", max(1, num_live // 2)))
    seed = int(getattr(ns_cfg, "seed", 0))
    dlogz_stop = float(getattr(ns_cfg, "dlogz_stop", 0.0))
    ll_batch_size = int(getattr(ns_cfg, "likelihood_batch_size", min(num_live, 16)))
    ll_batch_size = max(1, min(num_live, ll_batch_size))
    jit_enable = bool(getattr(ns_cfg, "jit", True))

    exp_dir.mkdir(parents=True, exist_ok=True)

    delta_dict, defaults_active = _collect_delta_and_defaults(cfg)

    # Build joint prior
    prior, param_names = build_joint_prior_distrax(cfg)

    print(f"[blackjax_ns] Running nested sampling...")
    print(f"[blackjax_ns] Free parameters: {len(param_names)}")
    print(f"[blackjax_ns] Parameter names: {param_names}")
    print(f"[blackjax_ns] num_live_points: {num_live}")
    print(f"[blackjax_ns] num_inner_steps: {num_inner_steps}")
    print(f"[blackjax_ns] num_delete: {num_delete}")
    print(f"[blackjax_ns] dlogz_stop: {dlogz_stop}")
    print(f"[blackjax_ns] likelihood_batch_size: {ll_batch_size}")
    print(f"[blackjax_ns] jit: {jit_enable}")

    if bool(getattr(cfg.opac, "full_grid", False)):
        print(
            "[blackjax_ns][warning] cfg.opac.full_grid=True can make BlackJAX NS very memory-hungry, "
            "because likelihood evaluations may be batched; consider full_grid=False if you hit OOM."
        )

    # RNG and initial particles
    rng_key = jax.random.PRNGKey(seed)
    rng_key, prior_key = jax.random.split(rng_key)

    particles = prior.sample(seed=prior_key, sample_shape=(num_live,))

    # Inject delta params + defaults into the particle pytree (stable keys for fm)
    def inject_fixed(pytree: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        out = dict(pytree)
        base_dtype = out[param_names[0]].dtype if len(param_names) else jnp.float64
        for k, v in delta_dict.items():
            out[k] = jnp.full((num_live,), v, dtype=base_dtype)
        for k, v in defaults_active.items():
            if k not in out:
                out[k] = jnp.full((num_live,), v, dtype=base_dtype)
        return out

    particles = inject_fixed(particles)

    # Observations
    y_obs = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

    # Extract offset parameters
    offset_param_names, offset_group_idx, has_offsets = _extract_offset_params(cfg, obs)

    def _loglikelihood_single(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        mu = fm(params)
        valid_mu = jnp.all(jnp.isfinite(mu))

        def invalid_ll(_):
            return jnp.asarray(LOG_FLOOR)

        def valid_ll(_):
            # Apply instrument offsets if defined (offset params are in ppm)
            if has_offsets:
                offset_values = jnp.array([params[n] for n in offset_param_names])
                offset_vec = offset_values[offset_group_idx] / 1e6  # ppm -> fractional
                y_shifted = y_obs - offset_vec
            else:
                y_shifted = y_obs

            r = y_shifted - mu

            # c is always present due to injection (or present in YAML sampling)
            c = params["c"]
            sig_jit2 = 10.0 ** (2.0 * c)

            sig_eff = jnp.sqrt(dy_obs**2 + sig_jit2)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = -jnp.log(sig_eff) - 0.5 * jnp.log(2.0 * jnp.pi)
            ll = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)

            return jnp.where(jnp.isfinite(ll), ll, jnp.asarray(LOG_FLOOR))

        return jax.lax.cond(valid_mu, valid_ll, invalid_ll, operand=None)

    def _pad_to_multiple(pytree: Dict[str, jnp.ndarray], multiple: int) -> Tuple[Dict[str, jnp.ndarray], int]:
        n = int(jax.tree_util.tree_leaves(pytree)[0].shape[0])
        pad = (-n) % multiple
        if pad == 0:
            return pytree, n

        def pad_leaf(x: jnp.ndarray) -> jnp.ndarray:
            tail = jnp.repeat(x[-1:], pad, axis=0)
            return jnp.concatenate([x, tail], axis=0)

        return jax.tree.map(pad_leaf, pytree), n

    def loglikelihood_fn_impl(particles_batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        BlackJAX NS calls `loglikelihood_fn` with a *batch* of particles.

        We evaluate the forward model in a memory-bounded way to avoid
        vectorizing the full radiative-transfer calculation over `num_live`.
        """
        # BlackJAX also calls this on a *single* particle inside the inner kernel.
        # Detect that case (scalar leaves) and return a scalar log-likelihood.
        first_leaf = jax.tree_util.tree_leaves(particles_batch)[0]
        if first_leaf.ndim == 0:
            return _loglikelihood_single(particles_batch)

        n = first_leaf.shape[0]
        if ll_batch_size >= n:
            return jax.vmap(_loglikelihood_single)(particles_batch)

        padded, n = _pad_to_multiple(particles_batch, ll_batch_size)
        n_full = int(jax.tree_util.tree_leaves(padded)[0].shape[0])
        n_blocks = n_full // ll_batch_size

        blocks = jax.tree.map(
            lambda x: x.reshape((n_blocks, ll_batch_size) + x.shape[1:]),
            padded,
        )

        def body(carry, idx):
            block_params = jax.tree.map(lambda x: x[idx], blocks)
            ll_block = jax.vmap(_loglikelihood_single)(block_params)
            return carry, ll_block

        _, ll_blocks = jax.lax.scan(body, None, jnp.arange(n_blocks))  # (n_blocks, ll_batch_size)
        return ll_blocks.reshape((n_full,))[:n]

    # For logprior, we must evaluate only over the sampled parameters (exclude injected deltas/defaults).
    sampled_only = {name: None for name in param_names}

    def strip_fixed(pytree: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        return {k: pytree[k] for k in sampled_only.keys()}

    def logprior_fn_impl(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        return prior.log_prob(strip_fixed(params))

    loglikelihood_fn = jax.jit(loglikelihood_fn_impl) if jit_enable else loglikelihood_fn_impl
    logprior_fn = jax.jit(logprior_fn_impl) if jit_enable else logprior_fn_impl

    nested_sampler = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )

    init_fn = jax.jit(nested_sampler.init) if jit_enable else nested_sampler.init
    step_fn = jax.jit(nested_sampler.step) if jit_enable else nested_sampler.step

    print(f"[blackjax_ns] Initializing with {num_live} live points...")
    live = init_fn(particles)

    dead = []
    with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
        while True:
            # Remaining evidence estimate (should decrease toward 0 as the run converges)
            dlogz = float(jax.device_get(live.logZ_live - live.logZ))

            if dlogz <= dlogz_stop:
                break

            rng_key, subkey = jax.random.split(rng_key, 2)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            pbar.update(num_delete)

    dead = blackjax.ns.utils.finalise(live, dead)

    rng_key, weight_key, sample_key = jax.random.split(rng_key, 3)
    re_samples = blackjax.ns.utils.sample(sample_key, dead, shape=num_live)
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

    samples = NestedSamples(
        dead.particles,
        logL=dead.loglikelihood,
        logL_birth=dead.loglikelihood_birth,
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
