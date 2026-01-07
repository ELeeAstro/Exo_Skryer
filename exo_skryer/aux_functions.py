'''
aux_functions.py
================
'''

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
import functools
from typing import Optional, Literal



__all__ = ['pchip_1d', 'latin_hypercube', 'simpson', 'simpson_padded']


def _pchip_slopes(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute slopes for PCHIP interpolation.

    Parameters
    ----------
    x : `~jax.numpy.ndarray`
        1D array of node positions with shape (N,), must be sorted in ascending
        order. Minimum length is 2.
    y : `~jax.numpy.ndarray`
        1D array of function values at the node positions with shape (N,).
        Must have the same length as `x`.

    Returns
    -------
    m : `~jax.numpy.ndarray`
        1D array of slopes (first derivatives) at each node position with shape (N,).
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    N = x.shape[0]
    h = jnp.diff(x)
    delta = jnp.diff(y) / h

    def slopes_N2():
        m = jnp.full_like(y, delta[0])
        return m

    def slopes_Nge3():
        h0 = h[:-1]
        h1 = h[1:]
        d0 = delta[:-1]
        d1 = delta[1:]

        w1 = 2.0 * h1 + h0
        w2 = 2.0 * h0 + h1

        same_sign = (d0 * d1) > 0.0

        denom = (w1 / jnp.where(jnp.abs(d0) > 0.0, d0, 1.0)
              + w2 / jnp.where(jnp.abs(d1) > 0.0, d1, 1.0))
        m_inner = (w1 + w2) / denom
        m_inner = jnp.where(same_sign, m_inner, 0.0)

        m0 = ((2.0*h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
        m0 = jnp.where(jnp.sign(m0) != jnp.sign(delta[0]), 0.0, m0)
        m0 = jnp.where(
            (jnp.sign(delta[0]) != jnp.sign(delta[1])) & (jnp.abs(m0) > jnp.abs(3.0 * delta[0])),
            3.0 * delta[0],
            m0,
        )

        mN = ((2.0*h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
        mN = jnp.where(jnp.sign(mN) != jnp.sign(delta[-1]), 0.0, mN)
        mN = jnp.where(
            (jnp.sign(delta[-1]) != jnp.sign(delta[-2])) & (jnp.abs(mN) > jnp.abs(3.0 * delta[-1])),
            3.0 * delta[-1],
            mN,
        )

        m = jnp.empty_like(y)
        m = m.at[0].set(m0)
        m = m.at[1:-1].set(m_inner)
        m = m.at[-1].set(mN)
        return m

    return jnp.where(N == 2, slopes_N2(), slopes_Nge3())


def pchip_1d(x: jnp.ndarray,
             x_nodes: jnp.ndarray,
             y_nodes: jnp.ndarray) -> jnp.ndarray:
    """Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).

    Provides a 1D monotonic cubic interpolation. Values outside the node
    range are clipped to the boundary values.

    Parameters
    ----------
    x : `~jax.numpy.ndarray`
        The x-coordinates at which to evaluate the interpolated values.
        Can be any shape; interpolation is performed element-wise.
    x_nodes : `~jax.numpy.ndarray`
        1D array of data point x-coordinates, must be sorted in ascending order.
        Minimum length is 2.
    y_nodes : `~jax.numpy.ndarray`
        1D array of data point y-coordinates corresponding to `x_nodes`.
        Must have the same length as `x_nodes`.

    Returns
    -------
    y : `~jax.numpy.ndarray`
        The interpolated values at positions `x`, with the same shape as `x`.
        Values are computed using shape-preserving cubic Hermite interpolation.
        Points outside the range [x_nodes[0], x_nodes[-1]] are clipped to
        boundary values.
    """
    x = jnp.asarray(x)
    x_nodes = jnp.asarray(x_nodes)
    y_nodes = jnp.asarray(y_nodes)
    x_min = x_nodes[0]
    x_max = x_nodes[-1]
    x_eval = jnp.clip(x, x_min, x_max)

    m_nodes = _pchip_slopes(x_nodes, y_nodes)  # (N,)

    idx = jnp.searchsorted(x_nodes, x_eval, side="right") - 1
    nseg = x_nodes.shape[0] - 1
    idx = jnp.clip(idx, 0, nseg - 1)

    x0 = x_nodes[idx]
    x1 = x_nodes[idx + 1]
    y0 = y_nodes[idx]
    y1 = y_nodes[idx + 1]
    m0 = m_nodes[idx]
    m1 = m_nodes[idx + 1]

    h = x1 - x0
    t = (x_eval - x0) / jnp.maximum(h, 1e-30)

    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2

    y = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    return y


def latin_hypercube(
    key: jax.Array,
    n_samples: int,
    n_dim: int,
    *,
    scramble: bool = True,
    dtype=jnp.float64,
) -> tuple[jnp.ndarray, jax.Array]:
    """Generate Latin hypercube samples in the unit hypercube [0, 1)^n_dim.

    Parameters
    ----------
    key : `~jax.Array`
        JAX PRNG key for random number generation.
    n_samples : int
        Number of samples to generate. Must be positive.
    n_dim : int
        Number of dimensions for each sample. Must be positive.
    scramble : bool, optional
        If True (default), randomly permutes the stratum assignments for each
        dimension independently, reducing correlation between dimensions and
        improving space-filling properties. If False, strata are assigned
        sequentially without permutation.
    dtype : dtype, optional
        Data type for the output array. Default is `jax.numpy.float64`.

    Returns
    -------
    samples : `~jax.numpy.ndarray`
        Generated Latin hypercube samples with shape `(n_samples, n_dim)`.
        Each value is in the range [0, 1).
    key : `~jax.Array`
        Updated PRNG key for subsequent random operations.
    """
    dtype = jnp.dtype(dtype)
    key, key_u, key_perm = jax.random.split(key, 3)

    base = (
        jnp.arange(n_samples, dtype=dtype)[:, None]
        + jax.random.uniform(key_u, (n_samples, n_dim), dtype=dtype)
    ) / jnp.asarray(n_samples, dtype=dtype)

    if not scramble:
        return base, key

    perm_keys = jax.random.split(key_perm, n_dim)

    def _permute_one(col: jnp.ndarray, k: jax.Array) -> jnp.ndarray:
        perm = jax.random.permutation(k, n_samples)
        return col[perm]

    cols = jax.vmap(_permute_one, in_axes=(0, 0))(base.T, perm_keys)  # (n_dim, n_samples)
    return cols.T, key


EvenMode = Optional[Literal["simpson", "avg", "first", "last"]]


def _as_last_axis(a: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.moveaxis(a, axis, -1)


def _trapz_last(y: jnp.ndarray, x: Optional[jnp.ndarray], dx: float) -> jnp.ndarray:
    # y shape (..., N>=2)
    y0 = y[..., -2]
    y1 = y[..., -1]
    if x is None:
        h = jnp.asarray(dx, dtype=y.dtype)
    else:
        h = x[..., -1] - x[..., -2]
    return 0.5 * h * (y0 + y1)


def _trapz_first(y: jnp.ndarray, x: Optional[jnp.ndarray], dx: float) -> jnp.ndarray:
    # y shape (..., N>=2)
    y0 = y[..., 0]
    y1 = y[..., 1]
    if x is None:
        h = jnp.asarray(dx, dtype=y.dtype)
    else:
        h = x[..., 1] - x[..., 0]
    return 0.5 * h * (y0 + y1)


def _simpson_odd_uniform(y: jnp.ndarray, dx: float) -> jnp.ndarray:
    # y shape (..., N) with N odd >= 3
    n = y.shape[-1]
    y0 = y[..., 0]
    yN = y[..., -1]
    odd_sum = jnp.sum(y[..., 1:n-1:2], axis=-1)
    even_sum = jnp.sum(y[..., 2:n-1:2], axis=-1)
    return (jnp.asarray(dx, dtype=y.dtype) / 3.0) * (y0 + yN + 4.0 * odd_sum + 2.0 * even_sum)


def _simpson_odd_unequal(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    # y, x shape (..., N) with N odd >= 3
    # Composite Simpson for irregular spacing:
    # sum over pairs of intervals with widths h0, h1:
    #  (h0+h1)/6 * [ (2 - h1/h0) y0 + ((h0+h1)^2/(h0*h1)) y1 + (2 - h0/h1) y2 ]
    #
    # IMPORTANT: When h0 or h1 is zero (repeated x-values from padding),
    # that interval contributes nothing to the integral.
    h = jnp.diff(x, axis=-1)              # (..., N-1)
    h0 = h[..., 0::2]                     # (..., (N-1)/2)
    h1 = h[..., 1::2]                     # (..., (N-1)/2)

    y0 = y[..., 0:-2:2]
    y1 = y[..., 1:-1:2]
    y2 = y[..., 2::2]

    hsum = h0 + h1

    # Use a small epsilon to detect zero spacing (from repeated x-values)
    eps = 1e-15
    h0_is_zero = jnp.abs(h0) < eps
    h1_is_zero = jnp.abs(h1) < eps

    # When either h0 or h1 is zero, the interval contributes 0
    # Use jnp.where to avoid division by zero
    safe_h0 = jnp.where(h0_is_zero, 1.0, h0)  # Replace 0 with 1 to avoid division by zero
    safe_h1 = jnp.where(h1_is_zero, 1.0, h1)

    term0 = (2.0 - (h1 / safe_h0)) * y0
    term1 = ((hsum * hsum) / (safe_h0 * safe_h1)) * y1
    term2 = (2.0 - (h0 / safe_h1)) * y2

    # Zero out contributions from intervals with zero spacing
    interval_valid = ~(h0_is_zero | h1_is_zero)
    contribution = (hsum / 6.0) * (term0 + term1 + term2)
    contribution = jnp.where(interval_valid, contribution, 0.0)

    return jnp.sum(contribution, axis=-1)


def _simpson_odd(y: jnp.ndarray, x: Optional[jnp.ndarray], dx: float) -> jnp.ndarray:
    if x is None:
        return _simpson_odd_uniform(y, dx)
    return _simpson_odd_unequal(y, x)


def _simpson_even_cartwright_last_interval(y: jnp.ndarray, x: Optional[jnp.ndarray], dx: float) -> jnp.ndarray:
    # "simpson" behaviour for even N: Simpson on first N-1 points + special last-interval correction
    # Uses last three points. For uniform spacing this reduces to:
    #   dx * (5/12*y[-1] + 2/3*y[-2] - 1/12*y[-3])
    if x is None:
        h = jnp.asarray(dx, dtype=y.dtype)
        return h * ((5.0/12.0) * y[..., -1] + (2.0/3.0) * y[..., -2] - (1.0/12.0) * y[..., -3])

    h0 = x[..., -2] - x[..., -3]
    h1 = x[..., -1] - x[..., -2]

    # Handle zero spacing (repeated x-values from padding)
    eps = 1e-15
    h0_is_zero = jnp.abs(h0) < eps
    h1_is_zero = jnp.abs(h1) < eps

    # If either spacing is zero, fall back to trapezoid rule on the valid portion
    # or return zero if both are zero
    safe_h0 = jnp.where(h0_is_zero, 1.0, h0)
    safe_h1 = jnp.where(h1_is_zero, 1.0, h1)

    alpha = (2.0 * safe_h1 * safe_h1 + 3.0 * safe_h0 * safe_h1) / (6.0 * (safe_h0 + safe_h1))
    beta  = (safe_h1 * safe_h1 + 3.0 * safe_h0 * safe_h1)       / (6.0 * safe_h0)
    eta   = (safe_h1 * safe_h1 * safe_h1)                       / (6.0 * safe_h0 * (safe_h0 + safe_h1))

    result = alpha * y[..., -1] + beta * y[..., -2] - eta * y[..., -3]

    # If the last interval has zero spacing, the contribution should be zero
    result = jnp.where(h1_is_zero, 0.0, result)

    return result


@functools.partial(jax.jit, static_argnames=("axis", "even"))
def simpson(
    y,
    *,
    x: Optional[jnp.ndarray] = None,
    dx: float = 1.0,
    axis: int = -1,
    even: EvenMode = None,
):
    """
    JAX-compatible composite Simpson integrator, similar to scipy.integrate.simpson.

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        Sample points. If 1D, must have length y.shape[axis]. If broadcastable, must match y.
    dx : float
        Spacing used when x is None.
    axis : int
        Axis of integration.
    even : {None, 'simpson', 'avg', 'first', 'last'}
        Handling when number of samples is even. Matches SciPy's documented behaviours. 
    """
    y = jnp.asarray(y)
    y = _as_last_axis(y, axis)
    n = y.shape[-1]

    # Prepare x in "last-axis" layout, broadcasted to y if provided.
    if x is not None:
        x = jnp.asarray(x)
        if x.ndim == 1:
            # broadcast to y's leading dims
            x = jnp.broadcast_to(x, y.shape)
        else:
            x = _as_last_axis(x, axis)
            x = jnp.broadcast_to(x, y.shape)

    # Degenerate cases
    if n == 0:
        return jnp.zeros(y.shape[:-1], dtype=y.dtype)
    if n == 1:
        return jnp.zeros(y.shape[:-1], dtype=y.dtype)
    if n == 2:
        # Simpson not possible; fall back to trapezoid
        return _trapz_last(y, x, dx)

    # Odd number of samples: standard composite Simpson
    if (n % 2) == 1:
        return _simpson_odd(y, x, dx)

    # Even number of samples: choose strategy
    mode = "simpson" if even is None else even

    if mode == "first":
        # Simpson over first N-1 points + trapezoid on last interval
        base = _simpson_odd(y[..., :-1], None if x is None else x[..., :-1], dx)
        return base + _trapz_last(y, x, dx)

    if mode == "last":
        # trapezoid on first interval + Simpson over last N-1 points
        base = _simpson_odd(y[..., 1:], None if x is None else x[..., 1:], dx)
        return _trapz_first(y, x, dx) + base

    if mode == "avg":
        # average of 'first' and 'last'
        first = (
            _simpson_odd(y[..., :-1], None if x is None else x[..., :-1], dx) +
            _trapz_last(y, x, dx)
        )
        last = (
            _trapz_first(y, x, dx) +
            _simpson_odd(y[..., 1:], None if x is None else x[..., 1:], dx)
        )
        return 0.5 * (first + last)

    # mode == "simpson": Simpson over first N-1 points + Cartwright correction
    base = _simpson_odd(y[..., :-1], None if x is None else x[..., :-1], dx)
    corr = _simpson_even_cartwright_last_interval(y, x, dx)
    return base + corr


def _move_last(a: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Move specified axis to last position."""
    return jnp.moveaxis(a, axis, -1)


def _take_last_axis(a: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Take element from last axis at index idx. vmap-friendly."""
    return jnp.take(a, idx, axis=-1, mode="clip")


@functools.partial(jax.jit, static_argnames=("axis", "even"))
def simpson_padded(
    y: jnp.ndarray,
    x: jnp.ndarray,
    n_valid: jnp.ndarray,
    *,
    axis: int = -1,
    even: str = "first",
):
    """
    Composite Simpson integration for padded arrays with non-uniform spacing.

    Parameters
    ----------
    y : `~jax.numpy.ndarray`
        Function values, padded to Nmax along `axis`.
    x : `~jax.numpy.ndarray`
        Sample points (same shape as y), padded to Nmax along `axis`.
        Must be strictly increasing on the valid prefix [0:n_valid].
    n_valid : int or `~jax.numpy.ndarray`
        Number of valid (unpadded) points along `axis` (>= 2).
        Can be scalar int or per-batch int array (vmap-friendly).
    axis : int, optional
        Axis of integration. Default: -1
    even : {'first', 'last', 'avg'}, optional
        Handling when n_valid is even:
        - 'first': Simpson on first n_valid-1 + trapezoid on last interval
        - 'last': Trapezoid on first + Simpson on last n_valid-1
        - 'avg': Average of 'first' and 'last'
        Default: 'first'

    Returns
    -------
    integral : `~jax.numpy.ndarray`
    """


    y = jnp.asarray(y)
    x = jnp.asarray(x)
    y = _move_last(y, axis)
    x = _move_last(x, axis)

    Nmax = y.shape[-1]
    n = jnp.clip(n_valid, 2, Nmax)

    # ---- Grid sanity: require strictly increasing x on valid prefix ----
    # Dynamic slicing not allowed under jit, so mask the diffs.
    h_all = jnp.diff(x, axis=-1)  # (..., Nmax-1)
    idx_h = jnp.arange(Nmax - 1)
    h_mask = idx_h < (n - 1)      # True for valid diffs only
    grid_ok = jnp.all(jnp.where(h_mask, h_all > 0.0, True), axis=-1)

    # Simpson needs odd number of points; for even n apply Simpson on first n-1
    m = jnp.where((n % 2) == 1, n, n - 1)  # odd, >= 3 unless n == 2

    def core_simpson(_):
        # Handle m == 2 (happens if n == 2): trapezoid on that single interval.
        def trap_only(_):
            i1 = n - 1
            i0 = n - 2
            h = _take_last_axis(x, i1) - _take_last_axis(x, i0)
            return 0.5 * h * (_take_last_axis(y, i0) + _take_last_axis(y, i1))

        def simpson_body(_):
            # Pair indices for Simpson panels (y0, y1, y2) up to max possible for Nmax
            n_pairs_max = (Nmax - 1) // 2
            j = jnp.arange(n_pairs_max)

            i0 = 2 * j
            i1 = i0 + 1
            i2 = i0 + 2

            # A pair is valid iff i2 < m (all three points inside Simpson prefix)
            pair_mask = i2 < m

            # Gather y triplets (static-length slices, JIT-friendly)
            y0 = y[..., 0:-2:2]
            y1 = y[..., 1:-1:2]
            y2 = y[..., 2::2]

            # Interval widths for each Simpson panel
            h = jnp.diff(x, axis=-1)          # (..., Nmax-1)
            h0 = h[..., 0:-1:2]               # (..., n_pairs_max)
            h1 = h[..., 1::2]                 # (..., n_pairs_max)

            # Also require positive widths inside used region
            pair_mask = pair_mask & (h0 > 0.0) & (h1 > 0.0)

            hsum = h0 + h1

            # Safe ratios (only used where pair_mask True)
            r10 = jnp.where(pair_mask, h1 / h0, 0.0)
            r01 = jnp.where(pair_mask, h0 / h1, 0.0)
            mid = jnp.where(pair_mask, (hsum * hsum) / (h0 * h1), 0.0)

            term0 = (2.0 - r10) * y0
            term1 = mid * y1
            term2 = (2.0 - r01) * y2

            panel = (hsum / 6.0) * (term0 + term1 + term2)
            panel = jnp.where(pair_mask, panel, 0.0)

            simpson_part = jnp.sum(panel, axis=-1)

            # Even-n handling
            def add_last_trap(_):
                i1 = n - 1
                i0 = n - 2
                hlast = _take_last_axis(x, i1) - _take_last_axis(x, i0)
                trap = 0.5 * hlast * (_take_last_axis(y, i0) + _take_last_axis(y, i1))
                return simpson_part + trap

            def add_first_trap(_):
                i0 = jnp.array(0, dtype=jnp.int32)
                i1 = jnp.array(1, dtype=jnp.int32)
                hfirst = _take_last_axis(x, i1) - _take_last_axis(x, i0)
                trap = 0.5 * hfirst * (_take_last_axis(y, i0) + _take_last_axis(y, i1))
                return trap + simpson_part

            def even_adjust(_):
                if even == "last":
                    return add_first_trap(None)
                if even == "avg":
                    return 0.5 * (add_last_trap(None) + add_first_trap(None))
                # default "first": Simpson on first n-1 + last trap
                return add_last_trap(None)

            return lax.cond((n % 2) == 0, even_adjust, lambda _: simpson_part, operand=None)

        return lax.cond(m <= 2, trap_only, simpson_body, operand=None)

    # If grid is bad (e.g., repeated x inside valid region), return NaN
    def bad_grid(_):
        return jnp.nan * jnp.ones(y.shape[:-1], dtype=y.dtype)

    return lax.cond(grid_ok, core_simpson, bad_grid, operand=None)
