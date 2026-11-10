"""
RT_em_common.py
===============

Shared pieces of the emission solvers: Gauss-Jacobi-5 angular quadrature
(Hogan 2024) and the layer intensity recursion.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

__all__: list = []

# Gauss-Jacobi-5 quadrature, 4 nodes (8 streams); weights for the measure 2 mu dmu (F = pi sum w I).
# NumPy float64, so the values do not depend on whether jax_enable_x64 was set before import.
_MU_NODES = np.array([0.0454586727, 0.2322334416, 0.5740198775, 0.9030775973])
_MU_WEIGHTS = np.array([0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087])


def _sweep(I0: jnp.ndarray, T: jnp.ndarray, S: jnp.ndarray, reverse: bool = False) -> jnp.ndarray:
    """Layer recursion ``I <- I * T + S`` over the leading (layer) axis; returns the end value only."""
    def step(I, x):
        return I * x[0] + x[1], None
    return lax.scan(step, I0, (T, S), reverse=reverse)[0]


def _fused_sweep(I0: jnp.ndarray, xs, layer_fn: Callable, reverse: bool = False) -> jnp.ndarray:
    """Layer recursion ``I <- I * T + S`` with ``T, S = layer_fn(*x)`` built inside each step.

    Only one layer's coefficients exist at a time, instead of (nlay, ...) arrays
    of them; the step is checkpointed, so reverse-mode AD stores only the carry
    per layer. Returns the end value only.
    """
    @jax.checkpoint
    def step(I, x):
        T, S = layer_fn(*x)
        return I * T + S, None
    return lax.scan(step, I0, xs, reverse=reverse)[0]


def _layer_contrib(w_q, src_up: jnp.ndarray, dtau: jnp.ndarray, mu) -> jnp.ndarray:
    """Upward layer sources attenuated to TOA, quadrature summed: (nlay, nwl), top-first."""
    tau_above = jnp.cumsum(dtau, axis=0) - dtau
    return jnp.einsum("m,lmw->lw", w_q, src_up * jnp.exp(-tau_above[:, None] / mu))
