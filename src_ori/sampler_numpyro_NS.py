"""Nested sampling driver using NumPyro's wrapper around JAXNS."""
from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.distributions import transforms

from build_prepared import Prepared


def _coerce_numeric(value):
    """
    Convert YAML scalars like '1e-8' to floats so numpyro sees numeric values.
    Recursively handles sequences to support vector-valued fixed parameters.
    """
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(value, (list, tuple)):
        converted = [_coerce_numeric(v) for v in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    return value


def _make_numpyro_model(cfg, prep: Prepared):
    """Build a NumPyro model consistent with the YAML configuration."""
    params_cfg = getattr(cfg, "params", None)
    if not params_cfg:
        raise ValueError("cfg.params must be defined to run nested sampling.")

    lam = jnp.asarray(prep.lam)
    dlam = jnp.asarray(prep.dlam)
    y_obs = jnp.asarray(prep.y)
    dy_obs = jnp.asarray(prep.dy)
    predict_fn = prep.fm

    def model():
        params: Dict[str, Any] = {}

        for p in params_cfg:
            name = getattr(p, "name", None)
            if not name:
                raise ValueError("Each parameter in cfg.params needs a 'name'.")

            dist_name = str(getattr(p, "dist", "")).lower()
            if not dist_name:
                raise ValueError(f"Parameter '{name}' needs a 'dist'.")

            if dist_name == "delta":
                raw_val = getattr(p, "value", getattr(p, "init", None))
                val = _coerce_numeric(raw_val)
                if val is None:
                    raise ValueError(
                        f"delta param '{name}' needs 'value' or 'init' in YAML."
                    )
                params[name] = jnp.asarray(val)
                continue

            if dist_name == "uniform":
                low = float(getattr(p, "low"))
                high = float(getattr(p, "high"))
                theta = numpyro.sample(name, dist.Uniform(low, high))
            elif dist_name in ("gaussian", "normal"):
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                theta = numpyro.sample(name, dist.Normal(mu, sigma))
            elif dist_name == "lognormal":
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                theta = numpyro.sample(name, dist.LogNormal(mu, sigma))
            else:
                raise ValueError(
                    f"Unsupported dist '{dist_name}' for param '{name}'."
                )

            params[name] = theta

        depth_model = predict_fn(params)
        sigma = jnp.clip(dy_obs, 1e-12, jnp.inf)
        numpyro.sample("obs", dist.Normal(depth_model, sigma), obs=y_obs)

    return model


def _build_constructor_kwargs(cfg):
    ns_cfg = getattr(getattr(cfg, "sampling", None), "numpyro_ns", None)
    if ns_cfg is None:
        return {}

    ctor = {}
    base_ctor = getattr(ns_cfg, "constructor_kwargs", None)
    if isinstance(base_ctor, dict):
        ctor.update(base_ctor)

    for key in ("num_live_points", "max_samples", "sampler", "sampler_name"):
        val = getattr(ns_cfg, key, None)
        if val is None:
            continue
        if key == "sampler_name":
            ctor["sampler"] = val
        else:
            ctor[key] = val

    sampler_kwargs = getattr(ns_cfg, "sampler_kwargs", None)
    if isinstance(sampler_kwargs, dict):
        ctor["sampler_kwargs"] = dict(sampler_kwargs)

    if not ctor:
        return {}

    return ctor


def _build_termination_kwargs(cfg):
    ns_cfg = getattr(getattr(cfg, "sampling", None), "numpyro_ns", None)
    if ns_cfg is None:
        return {}

    term_kwargs: Dict[str, Any] = {}

    term = getattr(ns_cfg, "termination", None)
    if term is None:
        pass
    else:
        term_kwargs.update({k: v for k, v in vars(term).items() if v is not None})

    extra = getattr(ns_cfg, "termination_kwargs", None)
    if isinstance(extra, dict):
        term_kwargs.update(extra)

    return term_kwargs


def run_nested_numpyro(cfg, prep: Prepared, exp_dir):
    """
    Run nested sampling via NumPyro and return posterior samples + evidence info.
    """
    ns_cfg = getattr(getattr(cfg, "sampling", None), "numpyro_ns", None)
    seed = int(getattr(ns_cfg, "seed", 0)) if ns_cfg is not None else 0
    posterior_samples = (
        int(getattr(ns_cfg, "posterior_samples", 2000)) if ns_cfg else 2000
    )

    rng_key = jax.random.PRNGKey(seed)

    model = _make_numpyro_model(cfg, prep)

    constructor_kwargs = _build_constructor_kwargs(cfg)
    termination_kwargs = _build_termination_kwargs(cfg)

    ns = NestedSampler(
        model,
        constructor_kwargs=constructor_kwargs or None,
        termination_kwargs=termination_kwargs or None,
    )
    ns.run(rng_key)
    ns.print_summary()

    _, sample_key = jax.random.split(rng_key)
    raw_samples = ns.get_samples(
        sample_key, num_samples=posterior_samples, group_by_chain=False
    )

    samples_dict: Dict[str, np.ndarray] = {}
    for p in cfg.params:
        name = p.name
        if name in raw_samples:
            samples_dict[name] = np.asarray(raw_samples[name])

    if ns._results is not None:
        evidence_info = {
            "logZ": float(ns._results.log_Z_mean),
            "logZ_err": float(ns._results.log_Z_uncert),
            "ESS": float(ns._results.ESS),
            "H_mean": float(ns._results.H_mean),
            "n_samples": int(ns._results.total_num_samples),
            "n_like": int(ns._results.total_num_likelihood_evaluations),
            "termination_reason": str(ns._results.termination_reason),
        }
    else:
        evidence_info = {}

    # include fixed params not sampled
    draws = next(iter(samples_dict.values())).shape[0] if samples_dict else 1
    for p in cfg.params:
        if p.name in samples_dict:
            continue
        dist_name = str(getattr(p, "dist", "")).lower()
        if dist_name == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is not None:
                samples_dict[p.name] = np.full((draws,), float(val))

    return samples_dict, evidence_info
