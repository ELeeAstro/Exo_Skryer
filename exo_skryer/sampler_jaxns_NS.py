"""
sampler_jaxns_NS.py
===================

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

# nested_jaxns.py
from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability.substrates.jax as tfp
import jaxns
from jaxns import NestedSampler, TerminationCondition, resample, Model, Prior, summary
from jaxns.utils import save_results

from .build_prepared import Prepared

tfpd = tfp.distributions


__all__ = [
    "make_jaxns_model",
    "run_nested_jaxns"
]


def make_jaxns_model(cfg, prep: Prepared) -> Model:
    """
    Build a JAXNS Model from the YAML config (cfg) and Prepared object (prep).

    - Priors are taken from cfg.params (your YAML 'params:' list).
    - Likelihood uses prep.predict_fn and observed data (prep.lam, prep.dlam, prep.y, prep.dy).
    """

    # --- observed data closed over in the likelihood ---
    lam   = jnp.asarray(prep.lam)
    dlam  = jnp.asarray(prep.dlam)
    y_obs = jnp.asarray(prep.y)
    dy_obs_p = jnp.asarray(prep.dy_p)
    dy_obs_m = jnp.asarray(prep.dy_m)

    # Only sample non-delta parameters; fixed values are injected inside the forward model.
    params_cfg = [
        p for p in cfg.params
        if str(getattr(p, "dist", "")).lower() != "delta"
    ]

    # ----- prior_model: generator of jaxns.Prior objects -----
    def prior_model():
        """
        Generator that yields Prior(...) objects and finally returns the
        dict of parameters passed to the likelihood.
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
            if dist_name in ("uniform",):
                low = float(getattr(p, "low"))
                high = float(getattr(p, "high"))
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

        # Whatever we return here is what the likelihood gets as `params`
        return params

    # ----- Split-normal (asymmetric Gaussian) log-likelihood -----
    @jax.jit
    def log_likelihood(theta_map: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        mu = prep.fm(theta_map)  # (N_obs,)

        res = y_obs - mu

        # choose sigma depending on which side y_obs lies relative to mu
        sig = jnp.where(res >= 0.0, dy_obs_p, dy_obs_m)

        # split-normal normalization: sqrt(2/pi) / (dy_minus + dy_plus)
        sig = jnp.clip(sig, 1e-300, jnp.inf)
        norm = jnp.clip(dy_obs_m + dy_obs_p, 1e-300, jnp.inf)
        logC = 0.5 * jnp.log(2.0 / jnp.pi) - jnp.log(norm)

        return jnp.sum(logC - 0.5 * (res / sig) ** 2)

    return Model(prior_model=prior_model, log_likelihood=log_likelihood)


def run_nested_jaxns(
    cfg,
    prep: Prepared,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Full-featured JAXNS driver.

    Called as:
        samples_dict, evidence_info = run_nested_jaxns(cfg, prep, exp_dir)

    Uses:
      - cfg.params          for prior definitions
      - cfg.sampling.jaxns  for JAXNS settings
      - prep.*              for data and forward model
    """

    jcfg = cfg.sampling.jaxns

    # ---- core NS configuration ----
    max_samples     = int(getattr(jcfg, "max_samples", 100_000))
    num_live_points = getattr(jcfg, "num_live_points", None)
    if num_live_points is not None:
        num_live_points = int(num_live_points)

    s        = getattr(jcfg, "s", None)
    k        = getattr(jcfg, "k", None)
    c        = getattr(jcfg, "c", None)
    shell_fraction = getattr(jcfg, "shell_fraction", 0.5)

    difficult_model      = bool(getattr(jcfg, "difficult_model", False))
    parameter_estimation = bool(getattr(jcfg, "parameter_estimation", True))
    gradient_guided      = bool(getattr(jcfg, "gradient_guided", False))
    init_eff_thr         = float(getattr(jcfg, "init_efficiency_threshold", 0.1))
    verbose              = bool(getattr(jcfg, "verbose", False))

    posterior_samples = int(getattr(jcfg, "posterior_samples", 5000))
    seed              = int(getattr(jcfg, "seed", 0))

    key = jax.random.PRNGKey(seed)

    # ---- build JAXNS model from cfg + prep ----
    model = make_jaxns_model(cfg, prep)

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
            max_num_likelihood_evaluations=getattr(
                term_cfg, "max_num_likelihood_evaluations", None
            ),
            rtol=getattr(term_cfg, "rtol", None),
            atol=getattr(term_cfg, "atol", None),
        )
    else:
        term_cond = TerminationCondition(
            ess=None,
            max_samples=max_samples,
        )

    # ---- run nested sampler ----
    term_reason, state = ns(key, term_cond=term_cond)
    results = ns.to_results(termination_reason=term_reason, state=state)

    # Print termination reason to CLI
    print(f"JAXNS termination_reason: {term_reason}")

    # Save full results to experiment directory
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_file = exp_dir / "jaxns_results.json"
    save_results(results, str(results_file))

    # Optional CLI summary
    summary(results)

    # ---- evidence info ----
    evidence_info: Dict[str, Any] = {
        "logZ":      float(results.log_Z_mean),
        "logZ_err":  float(results.log_Z_uncert),
        "ESS":       float(results.ESS),
        "H_mean":    float(results.H_mean),
        "n_samples": int(results.total_num_samples),
        "n_like":    int(results.total_num_likelihood_evaluations),
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

    # ---- convert to your standard samples_dict format ----
    samples_dict: Dict[str, np.ndarray] = {}

    # Bayesian parameters
    for p in cfg.params:
        name = p.name
        if name in eq_samples:
            samples_dict[name] = np.asarray(eq_samples[name])

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
