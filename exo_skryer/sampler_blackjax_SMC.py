"""
sampler_blackjax_SMC.py
=======================

BlackJAX adaptive persistent sampling (Karamanis et al. 2025), a tempered SMC
that keeps the particles of every temperature. Particles move with a random
walk whose proposal covariance and scale are retuned from the ensemble at each
step, so the likelihood needs no gradients.
"""

from __future__ import annotations
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from blackjax.mcmc import random_walk
from blackjax.smc import adaptive_persistent_sampling as aps
from blackjax.smc import persistent_sampling as ps
from blackjax.smc.resampling import systematic

from .retrieval_target import build_target

__all__ = [
    "run_smc_blackjax",
]

_TARGET_ACCEPTANCE = 0.234
_additive_step = random_walk.build_additive_step()


def _rw_step(rng_key, state, logdensity_fn, chol, scale):
    def random_step(key, position):
        return scale * (chol @ jax.random.normal(key, position.shape))

    return _additive_step(rng_key, state, logdensity_fn, random_step)


def _proposal_parameters(particles: np.ndarray, scale: float) -> Dict[str, jnp.ndarray]:
    """Random-walk parameters; a leading axis of 1 shares them between particles."""
    cov = np.atleast_2d(np.cov(particles, rowvar=False))
    chol = np.linalg.cholesky(cov + 1e-10 * np.eye(cov.shape[0]))
    return {"chol": jnp.asarray(chol)[None], "scale": jnp.asarray([scale])}


def _save_checkpoint(path: Path, state, key, scale: float) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, key=np.asarray(key), scale=scale, **state._asdict())
    os.replace(temporary, path)


def _load_checkpoint(path: Path):
    with np.load(path) as saved:
        state = ps.PersistentSMCState(*[jnp.asarray(saved[name]) for name in ps.PersistentSMCState._fields])
        return state, jnp.asarray(saved["key"]), float(saved["scale"])


def run_smc_blackjax(
    cfg,
    obs: dict,
    fm,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run BlackJAX persistent sampling and return (samples_dict, evidence_info).
    """
    scfg = getattr(cfg.sampling, "blackjax_smc", None)
    if scfg is None:
        raise ValueError("Missing cfg.sampling.blackjax_smc configuration.")

    num_particles = int(getattr(scfg, "num_particles", 1000))
    num_mcmc_steps = int(getattr(scfg, "num_mcmc_steps", 25))
    target_ess = float(getattr(scfg, "target_ess", 3.0))
    max_iterations = int(getattr(scfg, "max_iterations", 200))
    batch_size = int(getattr(scfg, "batch_size", 32))
    posterior_samples = int(getattr(scfg, "posterior_samples", 5000))
    seed = int(getattr(scfg, "seed", 42))
    resume = bool(getattr(scfg, "resume", False))
    checkpoint_interval = float(getattr(scfg, "checkpoint_interval", 600))
    checkpoint = exp_dir / str(getattr(scfg, "checkpoint", "blackjax_smc_checkpoint.npz"))

    target = build_target(cfg, obs, fm)
    ndim = len(target.names)

    print(f"[blackjax_smc] Running persistent sampling...")
    print(f"[blackjax_smc] Free parameters: {ndim}")
    print(f"[blackjax_smc] Parameter names: {target.names}")
    print(f"[blackjax_smc] num_particles: {num_particles}")
    print(f"[blackjax_smc] num_mcmc_steps: {num_mcmc_steps}")
    print(f"[blackjax_smc] target_ess: {target_ess}")
    print(f"[blackjax_smc] batch_size: {batch_size}")
    print(f"[blackjax_smc] checkpoint: {checkpoint}")

    # Built inside jit so that the likelihood sees runtime as a jit argument,
    # not as captured arrays.
    @jax.jit
    def step(key, state, params, runtime):
        kernel = aps.build_kernel(
            logprior_fn=target.logprior_u,
            loglikelihood_fn=partial(target.loglik_u, runtime=runtime),
            mcmc_step_fn=_rw_step,
            mcmc_init_fn=random_walk.init,
            resampling_fn=systematic,
            target_ess=target_ess,
            batch_size=batch_size,
        )
        return kernel(key, state, num_mcmc_steps, params)

    @jax.jit
    def init_state(positions, runtime):
        return aps.init(positions, partial(target.loglik_u, runtime=runtime), max_iterations, batch_size)

    if resume and checkpoint.exists():
        state, key, scale = _load_checkpoint(checkpoint)
        if state.num_particles != num_particles or state.tempering_schedule.size != max_iterations + 1:
            raise ValueError(f"{checkpoint} does not match num_particles and max_iterations in the YAML.")
        print(f"[blackjax_smc] Resuming at iteration {int(state.iteration)}")
    else:
        rng = np.random.default_rng(seed)
        particles = jnp.asarray(target.theta_to_u(target.sample_prior_theta(num_particles, rng)))
        state = init_state(particles, target.runtime)
        key = jax.random.PRNGKey(seed)
        scale = 2.38 / np.sqrt(ndim)

    last_save = time.monotonic()
    while float(state.tempering_param) < 1.0:
        if int(state.iteration) >= max_iterations:
            _save_checkpoint(checkpoint, state, key, scale)
            raise RuntimeError(
                f"[blackjax_smc] Reached max_iterations={max_iterations} at "
                f"temperature {float(state.tempering_param):.3g}. Increase max_iterations."
            )
        key, step_key = jax.random.split(key)
        params = _proposal_parameters(np.asarray(state.particles), scale)
        state, info = step(step_key, state, params, target.runtime)

        acceptance = float(jnp.mean(info.update_info.acceptance_rate))
        print(
            f"[blackjax_smc] iteration={int(state.iteration)} "
            f"temperature={float(state.tempering_param):.4g} "
            f"logZ={float(state.log_Z):.3f} acceptance={acceptance:.3f} scale={scale:.3f}",
            flush=True,
        )
        scale *= float(np.exp(acceptance - _TARGET_ACCEPTANCE))

        if time.monotonic() - last_save >= checkpoint_interval:
            _save_checkpoint(checkpoint, state, key, scale)
            last_save = time.monotonic()
    _save_checkpoint(checkpoint, state, key, scale)

    iteration = int(state.iteration)
    log_w, _ = ps.compute_log_persistent_weights(
        state.persistent_log_likelihoods,
        state.persistent_log_Z,
        state.tempering_schedule,
        iteration,
        include_current=True,
        normalize_to_one=True,
    )
    log_w = np.asarray(log_w[1:iteration + 1], dtype=np.float64).ravel()
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()
    ess = float(1.0 / np.sum(weights**2))

    u_all = np.asarray(state.persistent_particles[1:iteration + 1]).reshape(-1, ndim)
    rng = np.random.default_rng(seed + 1)
    u = u_all[rng.choice(weights.size, size=posterior_samples, p=weights)]
    samples_dict = target.samples_dict(target.u_to_theta(u))

    evidence_info: Dict[str, Any] = {
        "logZ": float(state.log_Z),
        # Importance-weight scatter only, so a lower limit. Repeat seeds for a full error.
        "logZ_err": float(np.sqrt(1.0 / ess)),
        "ESS": ess,
        "n_samples": posterior_samples,
        "n_like": num_particles * (1 + iteration * (num_mcmc_steps + 1)),
        "n_iterations": iteration,
        "sampler": "blackjax_smc",
    }

    print(f"[blackjax_smc] ESS: {int(ess)}")
    print(f"[blackjax_smc] logZ estimate: {evidence_info['logZ']:.2f} (scatter ± {evidence_info['logZ_err']:.2f})")
    print(f"[blackjax_smc] Posterior samples: {posterior_samples}")

    return samples_dict, evidence_info
