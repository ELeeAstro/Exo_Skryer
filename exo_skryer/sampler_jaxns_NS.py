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

import jax
import jax.numpy as jnp
from jax.nn import sigmoid as _sigmoid
import numpy as np

import tensorflow_probability.substrates.jax as tfp
import jaxns
from jaxns import NestedSampler, TerminationCondition, resample, Model, Prior, summary
from jaxns.utils import save_results

tfpd = tfp.distributions

__all__ = [
    "make_jaxns_model",
    "run_nested_jaxns",
]


def make_jaxns_model(cfg, obs: dict, fm) -> Model:
    """
    Build a JAXNS Model from cfg + obs + fm.

    Improvements vs prior version:
      - delta params injected into theta_map inside prior_model (stable keys)
      - optional defaults injected only if not explicitly specified in cfg.params (stable keys)
      - bounded uniforms can use "transform: logit" -> latent logistic z, mapped by sigmoid
        Note: if z ~ Logistic(0,1), then sigmoid(z) ~ Uniform(0,1) exactly (CDF transform),
        so this preserves the intended uniform prior while living in an unconstrained latent.
    """

    # --- observed data closed over in the likelihood ---
    y_obs = jnp.asarray(obs["y"])
    dy_obs = jnp.asarray(obs["dy"])

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
    OPTIONAL_DEFAULTS: Dict[str, float] = {
        "c": -99.0,  # log10(sigma_jit): "effectively zero jitter"
    }
    cfg_names = {p.name for p in cfg.params}
    optional_defaults_active = {k: v for k, v in OPTIONAL_DEFAULTS.items() if k not in cfg_names}

    # Guard against name collisions: cannot be both sampled and delta
    sampled_names = {p.name for p in params_cfg}
    overlap = sampled_names.intersection(delta_dict.keys())
    if overlap:
        raise ValueError(f"Parameter(s) appear both as sampled and delta: {sorted(overlap)}")

    # ----- prior_model: generator of jaxns.Prior objects -----
    def prior_model():
        """
        Generator that yields Prior(...) objects and finally returns the
        dict of parameters passed to the likelihood.

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
                    z = yield Prior(tfpd.Logistic(loc=0.0, scale=1.0), name=f"{name}__z")
                    t = jnp.clip(_sigmoid(z), 1e-12, 1.0 - 1e-12)
                    theta = low + (high - low) * t
                else:
                    theta = yield Prior(tfpd.Uniform(low=low, high=high), name=name)

            elif dist_name in ("gaussian", "normal"):
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                theta = yield Prior(tfpd.Normal(loc=mu, scale=sigma), name=name)

            elif dist_name == "lognormal":
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                theta = yield Prior(tfpd.LogNormal(loc=mu, scale=sigma), name=name)

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

        return params

    # ----- Gaussian (symmetric) log-likelihood -----
    @jax.jit
    def log_likelihood(theta_map: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        mu = fm(theta_map)  # (N,)
        valid_mu = jnp.all(jnp.isfinite(mu))

        def invalid_ll(_):
            return -jnp.inf

        def valid_ll(_):
            r = y_obs - mu  # (N,)

            # 'c' is guaranteed present: either in YAML (sampled/delta) or injected default
            c = theta_map["c"]  # log10(sigma_jit)

            # sigma_jit^2 = 10^(2c)
            sig_jit2 = 10.0 ** (2.0 * c)

            sig_eff = jnp.sqrt(dy_obs**2 + sig_jit2)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = -jnp.log(sig_eff) - 0.5 * jnp.log(2.0 * jnp.pi)
            ll = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)

            return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

        return jax.lax.cond(valid_mu, valid_ll, invalid_ll, operand=None)

    return Model(prior_model=prior_model, log_likelihood=log_likelihood)


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
    max_samples = int(getattr(jcfg, "max_samples", 100_000))
    num_live_points = getattr(jcfg, "num_live_points", None)
    if num_live_points is not None:
        num_live_points = int(num_live_points)

    s = getattr(jcfg, "s", None)
    k = getattr(jcfg, "k", None)
    c = getattr(jcfg, "c", None)
    shell_fraction = getattr(jcfg, "shell_fraction", 0.5)

    difficult_model = bool(getattr(jcfg, "difficult_model", False))
    parameter_estimation = bool(getattr(jcfg, "parameter_estimation", True))
    gradient_guided = bool(getattr(jcfg, "gradient_guided", False))
    init_eff_thr = float(getattr(jcfg, "init_efficiency_threshold", 0.1))
    verbose = bool(getattr(jcfg, "verbose", False))

    posterior_samples = int(getattr(jcfg, "posterior_samples", 5000))
    seed = int(getattr(jcfg, "seed", 0))

    key = jax.random.PRNGKey(seed)

    # ---- build JAXNS model from cfg + obs + fm ----
    model = make_jaxns_model(cfg, obs, fm)

    ns = NestedSampler(
        model=model,
        max_samples=max_samples,
        num_live_points=num_live_points,
        s=s,
        k=k,
        c=c,
        difficult_model=difficult_model,
        parameter_estimation=parameter_estimation,
        shell_fraction=shell_fraction,
        gradient_guided=gradient_guided,
        init_efficiency_threshold=init_eff_thr,
        verbose=verbose,
    )

    # ---- termination condition ----
    term_cfg = getattr(jcfg, "termination", None)
    if term_cfg is not None:
        term_cond = TerminationCondition(
            ess=getattr(term_cfg, "ess", None),
            evidence_uncert=getattr(term_cfg, "evidence_uncert", None),
            dlogZ=getattr(term_cfg, "dlogZ", None),
            max_samples=getattr(term_cfg, "max_samples", None),
            max_num_likelihood_evaluations=getattr(term_cfg, "max_num_likelihood_evaluations", None),
            rtol=getattr(term_cfg, "rtol", None),
            atol=getattr(term_cfg, "atol", None),
        )
    else:
        term_cond = TerminationCondition(ess=None, max_samples=max_samples)

    # ---- run nested sampler ----
    term_reason, state = ns(key, term_cond=term_cond)
    results = ns.to_results(termination_reason=term_reason, state=state)

    print(f"JAXNS termination_reason: {term_reason}")

    # Save full results to experiment directory
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_file = exp_dir / "jaxns_results.json"
    save_results(results, str(results_file))
    summary(results)

    # ---- evidence info ----
    evidence_info: Dict[str, Any] = {
        "logZ": float(results.log_Z_mean),
        "logZ_err": float(results.log_Z_uncert),
        "ESS": float(results.ESS),
        "H_mean": float(results.H_mean),
        "n_samples": int(results.total_num_samples),
        "n_like": int(results.total_num_likelihood_evaluations),
        "termination_reason": str(term_reason),
        "results_file": str(results_file),
    }

    # ---- posterior resampling to equal-weight samples ----
    post_key = jax.random.split(key, 2)[1]
    eq_samples = resample(
        key=post_key,
        samples=results.samples,
        log_weights=results.log_dp_mean,
        S=posterior_samples,
        replace=True,
    )

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
