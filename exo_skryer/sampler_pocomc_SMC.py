"""
sampler_pocomc_SMC.py
=====================

pocoMC preconditioned Monte Carlo driver. pocoMC needs PyTorch; use the pixi
``pocomc`` environment or ``pip install pocomc``. The likelihood is evaluated
in batches, so all active particles reach the forward model together.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .retrieval_target import batched, build_target

try:
    import pocomc
    POCOMC_AVAILABLE = True
except ImportError:
    pocomc = None
    POCOMC_AVAILABLE = False


__all__ = [
    "run_smc_pocomc",
]

# pocoMC pickles the likelihood into its state files, but the forward-model
# closure cannot be pickled. State files store _likelihood by name, and
# run_smc_pocomc() points it at the model rebuilt in the current process.
_active_likelihood = None


def _likelihood(theta: np.ndarray) -> np.ndarray:
    return _active_likelihood(theta)


def _latest_state(output: Path) -> Path | None:
    states = sorted(output.glob("*.state"), key=lambda path: path.stat().st_mtime)
    return states[-1] if states else None


def run_smc_pocomc(
    cfg,
    obs: dict,
    fm,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run pocoMC and return (samples_dict, evidence_info).
    """
    global _active_likelihood

    if not POCOMC_AVAILABLE:
        raise ImportError("pocomc is not installed. Use the pixi pocomc environment or: pip install pocomc")

    pcfg = getattr(cfg.sampling, "pocomc", None)
    if pcfg is None:
        raise ValueError("Missing cfg.sampling.pocomc configuration.")

    n_effective = int(getattr(pcfg, "n_effective", 512))
    n_active = int(getattr(pcfg, "n_active", 256))
    n_total = int(getattr(pcfg, "n_total", 4096))
    n_evidence = int(getattr(pcfg, "n_evidence", 4096))
    # Absent = 32; an explicit null means n_active.
    batch_size = int(getattr(pcfg, "batch_size", 32) or n_active)
    posterior_samples = int(getattr(pcfg, "posterior_samples", 5000))
    seed = int(getattr(pcfg, "seed", 42))
    resume = bool(getattr(pcfg, "resume", False))
    save_every = getattr(pcfg, "save_every", 5)
    pytorch_threads = int(getattr(pcfg, "pytorch_threads", 1))
    progress = bool(getattr(pcfg, "progress", False))

    output = Path(getattr(pcfg, "output", "pocomc_run"))
    if not output.is_absolute():
        output = exp_dir / output
    output.mkdir(parents=True, exist_ok=True)

    target = build_target(cfg, obs, fm)
    loglik = batched(target.loglik_u, batch_size, target.runtime)
    _active_likelihood = lambda theta: loglik(target.theta_to_u(theta))

    state_path = _latest_state(output) if resume else None

    print(f"[pocomc] Running preconditioned Monte Carlo...")
    print(f"[pocomc] Free parameters: {len(target.names)}")
    print(f"[pocomc] Parameter names: {target.names}")
    print(f"[pocomc] n_effective: {n_effective}")
    print(f"[pocomc] n_active: {n_active}")
    print(f"[pocomc] n_total: {n_total}")
    print(f"[pocomc] batch_size: {batch_size}")
    print(f"[pocomc] output: {output}")
    print(f"[pocomc] resuming from: {state_path}")

    sampler = pocomc.Sampler(
        pocomc.Prior(target.priors),
        _likelihood,
        n_effective=n_effective,
        n_active=n_active,
        vectorize=True,
        pytorch_threads=pytorch_threads,
        output_dir=str(output),
        random_state=seed,
    )
    sampler.run(
        n_total=n_total,
        n_evidence=n_evidence,
        progress=progress,
        resume_state_path=state_path,
        save_every=None if save_every is None else int(save_every),
    )

    theta, log_w, _, _ = sampler.posterior(return_logw=True)
    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()
    ess = float(1.0 / np.sum(weights**2))

    rng = np.random.default_rng(seed + 1)
    theta = theta[rng.choice(weights.size, size=posterior_samples, p=weights)]
    samples_dict = target.samples_dict(theta)

    log_z, log_z_err = sampler.evidence()
    evidence_info: Dict[str, Any] = {
        "logZ": float(log_z),
        "logZ_err": float("nan") if log_z_err is None else float(log_z_err),
        "ESS": ess,
        "n_samples": posterior_samples,
        "n_like": int(sampler.calls),
        "sampler": "pocomc",
        "output": str(output),
    }

    print(f"[pocomc] ESS: {int(ess)}")
    print(f"[pocomc] Likelihood calls: {evidence_info['n_like']}")
    print(f"[pocomc] logZ estimate: {evidence_info['logZ']:.2f} ± {evidence_info['logZ_err']:.2f}")
    print(f"[pocomc] Posterior samples: {posterior_samples}")

    return samples_dict, evidence_info
