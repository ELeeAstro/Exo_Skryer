"""
sampler_nessai_NS.py
====================

nessai nested sampler driver (normalising-flow proposals) with JAX forward-model
integration. nessai needs PyTorch; install with ``pip install "exo_skryer[nessai]"``.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .limb_asymmetry import jitter_param_name
from .sampler_nautilus_NS import build_loglikelihood_nautilus
from .retrieval_target import _prior_and_bounds

try:
    from nessai.flowsampler import FlowSampler
    from nessai.model import Model
    from nessai.utils import configure_logger
    NESSAI_AVAILABLE = True
except ImportError:
    NESSAI_AVAILABLE = False
    FlowSampler = None
    Model = object
    configure_logger = None


__all__ = [
    "build_model_nessai",
    "run_nested_nessai",
]

class _RetrievalModel(Model):
    """nessai model around the Exo Skryer priors and log-likelihood."""

    # The JAX likelihood takes one point at a time.
    allow_vectorised = False

    def __init__(self, priors: Dict[str, Any], bounds: Dict[str, Tuple[float, float]], loglike: Callable):
        self.names = list(priors)
        self.bounds = bounds
        self._priors = priors
        self._loglike = loglike

    def log_prior(self, x):
        with np.errstate(divide="ignore"):
            log_p = np.log(self.in_bounds(x), dtype="float")
        for name, prior in self._priors.items():
            log_p = log_p + prior.logpdf(x[name])
        return log_p

    def log_likelihood(self, x):
        return self._loglike({name: float(np.squeeze(x[name])) for name in self.names})


def build_model_nessai(cfg, obs: dict, fm: Callable) -> Tuple[_RetrievalModel, List[str]]:
    """
    Build the nessai model from cfg.params and return (model, free_param_names).
    """
    priors: Dict[str, Any] = {}
    bounds: Dict[str, Tuple[float, float]] = {}
    for p in cfg.params:
        if str(getattr(p, "dist", "")).lower() == "delta":
            continue
        priors[p.name], bounds[p.name] = _prior_and_bounds(p)

    param_names = list(priors)
    loglike = build_loglikelihood_nautilus(cfg, obs, fm, param_names)
    return _RetrievalModel(priors, bounds, loglike), param_names


def run_nested_nessai(
    cfg,
    obs: dict,
    fm: Callable,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run nessai nested sampling and return (samples_dict, evidence_info).
    """
    if not NESSAI_AVAILABLE:
        raise ImportError('nessai is not installed. Install with: pip install "exo_skryer[nessai]"')

    ncfg = getattr(cfg.sampling, "nessai", None)
    if ncfg is None:
        raise ValueError("Missing cfg.sampling.nessai configuration.")

    nlive = int(getattr(ncfg, "nlive", 2000))
    stopping = float(getattr(ncfg, "stopping", 0.1))
    seed = int(getattr(ncfg, "seed", 42))
    max_iteration = getattr(ncfg, "max_iteration", None)
    resume = bool(getattr(ncfg, "resume", False))
    checkpointing = bool(getattr(ncfg, "checkpointing", True))
    checkpoint_interval = int(getattr(ncfg, "checkpoint_interval", 600))
    pytorch_threads = int(getattr(ncfg, "pytorch_threads", 1))
    plot = bool(getattr(ncfg, "plot", False))
    log_level = str(getattr(ncfg, "log_level", "INFO")).upper()

    output = Path(getattr(ncfg, "output", "nessai_run"))
    if not output.is_absolute():
        output = exp_dir / output
    output.mkdir(parents=True, exist_ok=True)

    model, param_names = build_model_nessai(cfg, obs, fm)

    print(f"[nessai] Running nested sampling...")
    print(f"[nessai] Free parameters: {len(param_names)}")
    print(f"[nessai] Parameter names: {param_names}")
    print(f"[nessai] nlive: {nlive}")
    print(f"[nessai] stopping: {stopping}")
    print(f"[nessai] output: {output}")
    print(f"[nessai] resume: {resume}")

    configure_logger(output=str(output), log_level=log_level)

    sampler_kwargs: Dict[str, Any] = dict(
        output=str(output),
        resume=resume,
        nlive=nlive,
        stopping=stopping,
        seed=seed,
        checkpointing=checkpointing,
        checkpoint_interval=checkpoint_interval,
        pytorch_threads=pytorch_threads,
        plot=plot,
    )
    if max_iteration is not None:
        sampler_kwargs["max_iteration"] = int(max_iteration)

    sampler = FlowSampler(model, **sampler_kwargs)
    sampler.run(plot=plot)

    posterior = sampler.posterior_samples
    n_samples = int(posterior.size)

    samples_dict: Dict[str, np.ndarray] = {
        name: np.asarray(posterior[name], dtype=np.float64) for name in param_names
    }

    # Add fixed/delta parameters
    for p in cfg.params:
        if str(getattr(p, "dist", "")).lower() == "delta":
            val = getattr(p, "value", getattr(p, "init", None))
            if val is not None:
                samples_dict[p.name] = np.full((n_samples,), float(val), dtype=np.float64)

    # Add the optional jitter default if it is not a YAML parameter
    jitter_key = jitter_param_name(getattr(getattr(cfg, "physics", None), "rt_scheme", None))
    if jitter_key not in {p.name for p in cfg.params}:
        samples_dict[jitter_key] = np.full((n_samples,), -99.0, dtype=np.float64)

    log_w = np.asarray(sampler.ns.state.log_posterior_weights, dtype=np.float64)
    ess = float(np.exp(2.0 * logsumexp(log_w) - logsumexp(2.0 * log_w)))

    evidence_info: Dict[str, Any] = {
        "logZ": float(sampler.log_evidence),
        "logZ_err": float(sampler.log_evidence_error),
        "ESS": ess,
        "n_samples": n_samples,
        "sampler": "nessai",
        "nlive": nlive,
        "stopping": stopping,
        "output": str(output),
        "resumed": resume,
    }

    print(f"[nessai] ESS: {int(ess)}")
    print(f"[nessai] logZ estimate: {evidence_info['logZ']:.2f} ± {evidence_info['logZ_err']:.2f}")
    print(f"[nessai] Posterior samples: {n_samples}")

    return samples_dict, evidence_info
