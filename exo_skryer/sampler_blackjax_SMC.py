"""
sampler_blackjax_SMC.py
=======================

BlackJAX adaptive persistent sampling (Karamanis et al. 2025), a tempered SMC
that keeps the particles of every temperature. Particles move with a random
walk whose proposal covariance and scale are retuned from the ensemble at each
step, so the likelihood needs no gradients.

The forward model runs only inside ``retrieval_target.batched``, the fixed-size
jit(vmap) chunks that pocoMC also uses. BlackJAX's own kernel would instead
compile the forward model into one program per temperature step, nested inside
a lax.map over particle batches and a scan over MCMC steps. Here the tempering,
weighting and resampling still come from BlackJAX, and the Metropolis loop runs
from Python with the same random numbers, so the results match BlackJAX's kernel.
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
from blackjax.smc import persistent_sampling as ps
from blackjax.smc import solver
from blackjax.smc.resampling import systematic

from .retrieval_target import batched, build_target

__all__ = [
    "run_smc_blackjax",
]

_TARGET_ACCEPTANCE = 0.234


def _proposal_parameters(particles: np.ndarray, scale: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Cholesky factor of the ensemble covariance, and the random-walk scale."""
    cov = np.atleast_2d(np.cov(particles, rowvar=False))
    chol = np.linalg.cholesky(cov + 1e-10 * np.eye(cov.shape[0]))
    return jnp.asarray(chol), jnp.asarray(scale)


def _initial_state(particles: jnp.ndarray, log_likelihoods: jnp.ndarray, max_iterations: int):
    """blackjax persistent_sampling.init, with the likelihoods already evaluated."""
    n = max_iterations + 1
    return ps.PersistentSMCState(
        jnp.zeros((n, *particles.shape)).at[0].set(particles),
        jnp.zeros((n, particles.shape[0])).at[0].set(log_likelihoods),
        jnp.zeros(n),
        jnp.zeros(n),
        jnp.array(0),
    )


def _next_temperature(state, target_ess: float) -> jnp.ndarray:
    """blackjax adaptive_persistent_sampling.calculate_lambda."""
    target_val = jnp.log(state.persistent_weights.shape[1] * target_ess)
    current = state.tempering_schedule[state.iteration]
    max_delta = 1 - current

    def ess_gap(delta):
        log_weights, _ = ps.compute_log_persistent_weights(
            state.persistent_log_likelihoods,
            state.persistent_log_Z,
            state.tempering_schedule.at[state.iteration + 1].set(current + delta),
            state.iteration + 1,
            normalize_to_one=True,
        )
        return jnp.log(ps.compute_persistent_ess(log_weights)) - target_val

    # No root sets delta to 0, adding particles at this temperature before trying again.
    delta = jnp.nan_to_num(solver.dichotomy(ess_gap, 0.0, max_delta))
    return current + jnp.clip(delta, 0.0, max_delta)


@partial(jax.jit, static_argnames=("num_mcmc_steps",))
def _start_step(rng_key, state, target_ess, num_mcmc_steps: int):
    """Pick the temperature, weight and resample the persistent ensemble."""
    lmbda = _next_temperature(state, target_ess)
    iteration = state.iteration + 1
    updating_key, resampling_key = jax.random.split(rng_key, 2)
    log_weights, log_Z = ps.compute_log_persistent_weights(
        state.persistent_log_likelihoods,
        state.persistent_log_Z,
        state.tempering_schedule.at[iteration].set(lmbda),
        iteration,
        normalize_to_one=True,
    )
    particles, idx = ps.resample_from_persistent(
        resampling_key, state.persistent_particles, jnp.exp(log_weights), systematic
    )
    log_likelihoods = state.persistent_log_likelihoods.reshape(-1)[idx]
    keys = jax.random.split(updating_key, particles.shape[0])
    step_keys = jax.vmap(lambda k: jax.random.split(k, num_mcmc_steps))(keys)
    return lmbda, log_Z, particles, log_likelihoods, step_keys


@jax.jit
def _propose(keys, particles, chol, scale):
    """Random-walk proposals, as blackjax random_walk.build_additive_step."""
    pairs = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
    moves = jax.vmap(lambda k, x: scale * (chol @ jax.random.normal(k, x.shape)))(pairs[:, 0], particles)
    return particles + moves, pairs[:, 1]


@jax.jit
def _metropolis(accept_keys, lmbda, current, proposed):
    """Accept or reject, as blackjax build_rmh with static binomial sampling."""
    (x, log_prior, log_like), (x_new, log_prior_new, log_like_new) = current, proposed
    log_ratio = (log_prior_new + lmbda * log_like_new) - (log_prior + lmbda * log_like)
    log_ratio = jnp.where(jnp.isnan(log_ratio), -jnp.inf, log_ratio)
    p_accept = jnp.clip(jnp.exp(log_ratio), max=1)
    accept = jax.vmap(jax.random.bernoulli)(accept_keys, p_accept)
    keep = lambda new, old: jnp.where(accept.reshape(-1, *[1] * (new.ndim - 1)), new, old)
    return (keep(x_new, x), keep(log_prior_new, log_prior), keep(log_like_new, log_like)), jnp.sum(p_accept)


@jax.jit
def _finish_step(state, lmbda, log_Z, particles, log_likelihoods):
    iteration = state.iteration + 1
    return ps.PersistentSMCState(
        state.persistent_particles.at[iteration].set(particles),
        state.persistent_log_likelihoods.at[iteration].set(log_likelihoods),
        state.persistent_log_Z.at[iteration].set(log_Z),
        state.tempering_schedule.at[iteration].set(lmbda),
        iteration,
    )


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
    # 0 evaluates all particles in one call.
    batch_size = int(getattr(scfg, "batch_size", 32)) or num_particles
    posterior_samples = int(getattr(scfg, "posterior_samples", 5000))
    seed = int(getattr(scfg, "seed", 42))
    resume = bool(getattr(scfg, "resume", False))
    checkpoint_interval = float(getattr(scfg, "checkpoint_interval", 600))
    checkpoint = exp_dir / str(getattr(scfg, "checkpoint", "blackjax_smc_checkpoint.npz"))

    target = build_target(cfg, obs, fm)
    ndim = len(target.names)
    loglik = batched(target.loglik_u, batch_size, target.runtime)
    log_prior = jax.jit(jax.vmap(target.logprior_u))

    print(f"[blackjax_smc] Running persistent sampling...")
    print(f"[blackjax_smc] Free parameters: {ndim}")
    print(f"[blackjax_smc] Parameter names: {target.names}")
    print(f"[blackjax_smc] num_particles: {num_particles}")
    print(f"[blackjax_smc] num_mcmc_steps: {num_mcmc_steps}")
    print(f"[blackjax_smc] target_ess: {target_ess}")
    print(f"[blackjax_smc] batch_size: {batch_size}")
    print(f"[blackjax_smc] checkpoint: {checkpoint}")

    if resume and checkpoint.exists():
        state, key, scale = _load_checkpoint(checkpoint)
        if state.num_particles != num_particles or state.tempering_schedule.size != max_iterations + 1:
            raise ValueError(f"{checkpoint} does not match num_particles and max_iterations in the YAML.")
        print(f"[blackjax_smc] Resuming at iteration {int(state.iteration)}")
    else:
        rng = np.random.default_rng(seed)
        particles = target.theta_to_u(target.sample_prior_theta(num_particles, rng))
        state = _initial_state(jnp.asarray(particles), jnp.asarray(loglik(particles)), max_iterations)
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
        chol, step_scale = _proposal_parameters(np.asarray(state.particles), scale)
        lmbda, log_Z, x, log_like, step_keys = _start_step(step_key, state, target_ess, num_mcmc_steps)

        current = (x, log_prior(x), log_like)
        accepted = 0.0
        for k in range(num_mcmc_steps):
            x_new, accept_keys = _propose(step_keys[:, k], current[0], chol, step_scale)
            proposed = (x_new, log_prior(x_new), jnp.asarray(loglik(x_new)))
            current, p_sum = _metropolis(accept_keys, lmbda, current, proposed)
            accepted += float(p_sum)
        state = _finish_step(state, lmbda, log_Z, current[0], current[2])

        acceptance = accepted / (num_particles * num_mcmc_steps)
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
        "n_like": num_particles * (1 + iteration * num_mcmc_steps),
        "n_iterations": iteration,
        "sampler": "blackjax_smc",
    }

    print(f"[blackjax_smc] ESS: {int(ess)}")
    print(f"[blackjax_smc] logZ estimate: {evidence_info['logZ']:.2f} (scatter ± {evidence_info['logZ_err']:.2f})")
    print(f"[blackjax_smc] Posterior samples: {posterior_samples}")

    return samples_dict, evidence_info
