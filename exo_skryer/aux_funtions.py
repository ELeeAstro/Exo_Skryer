"""
aux_functions.py
"""

import jax
import jax.numpy as jnp

__all__ = ['pchip_1d', 'latin_hypercube']


def _pchip_slopes(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute slopes for PCHIP interpolation.

    This is a helper function for `pchip_1d` that computes slopes at each
    data point, ensuring the interpolation is shape-preserving (monotonic).

    Parameters
    ----------
    x : `~jax.numpy.ndarray`
        Node positions (N,), must be sorted in ascending order.
    y : `~jax.numpy.ndarray`
        Function values at nodes (N,).

    Returns
    -------
    `~jax.numpy.ndarray`
        Slopes (derivatives) at each node (N,).
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    N = x.shape[0]
    # Require at least 2 nodes; endpoint formulas need >=3
    # For N==2, linear interpolation: slopes are secant
    h = jnp.diff(x)
    delta = jnp.diff(y) / h

    def slopes_N2():
        m = jnp.full_like(y, delta[0])
        return m

    def slopes_Nge3():
        # interior
        h0 = h[:-1]
        h1 = h[1:]
        d0 = delta[:-1]
        d1 = delta[1:]

        w1 = 2.0 * h1 + h0
        w2 = 2.0 * h0 + h1

        same_sign = (d0 * d1) > 0.0

        # safe harmonic mean only where same_sign
        denom = (w1 / jnp.where(jnp.abs(d0) > 0.0, d0, 1.0)
              + w2 / jnp.where(jnp.abs(d1) > 0.0, d1, 1.0))
        m_inner = (w1 + w2) / denom
        m_inner = jnp.where(same_sign, m_inner, 0.0)

        # endpoints (SciPy-style)
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
    x_nodes : `~jax.numpy.ndarray`
        1D array of data point x-coordinates, must be sorted.
    y_nodes : `~jax.numpy.ndarray`
        1D array of data point y-coordinates, same length as `x_nodes`.

    Returns
    -------
    `~jax.numpy.ndarray`
        The interpolated values, same shape as `x`.
    """
    x = jnp.asarray(x)
    x_nodes = jnp.asarray(x_nodes)
    y_nodes = jnp.asarray(y_nodes)
    x_min = x_nodes[0]
    x_max = x_nodes[-1]
    x_eval = jnp.clip(x, x_min, x_max)

    m_nodes = _pchip_slopes(x_nodes, y_nodes)  # (N,)

    # Find interval index i so that x in [x_i, x_{i+1}]
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

    # Cubic Hermite basis (same as before, but slopes are PCHIP slopes)
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
    dtype=jnp.float32,
) -> tuple[jnp.ndarray, jax.Array]:
    """Generate Latin hypercube samples in [0, 1).

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    n_samples : int
        Number of samples to generate.
    n_dim : int
        Number of dimensions for each sample.
    scramble : bool, default=True
        If True, permutes strata assignment to reduce alignment.
    dtype : dtype, default=jax.numpy.float32
        Output dtype.

    Returns
    -------
    tuple[jax.numpy.ndarray, jax.Array]
        A tuple containing the generated samples of shape `(n_samples, n_dim)`
        and the updated PRNG key.
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
