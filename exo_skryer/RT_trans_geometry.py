"""
RT_trans_geometry.py
====================

Ray geometry for all transmission spectra (1D, 1.5D, 2D, 3D), following TRIDENT
(MacDonald & Lewis 2022, arXiv:2111.05862). The radiative transfer that uses it
lives in ``RT_trans_core_os`` and ``RT_trans_core_ck``.

A sector of the terminator holds one or more zenith slices (columns). Each slice
covers an angle range [theta_lo, theta_hi] measured from the terminator plane
along the line of sight (negative = dayside, positive = nightside), and has its
own layer radii on a shared layer index. A ray with impact parameter b sits at
radius r = b / cos(theta), so slice k only sees the radius window
[b / cos(theta_near), b / cos(theta_far)] of each half it covers.

``path_tensor`` returns P[i, k, l] = (chord length of ray i inside layer l of
slice k) / dr_kl, so the slant optical depth is
    tau_i = sum_kl P[i, k, l] * dtau_vert[k, l]       with dtau_vert = kappa * rho * dz
(dz == dr for every altitude scheme). With one slice covering the whole chord
(theta from -pi/2 to pi/2) this is the 1D path matrix (TRIDENT Eq. 19).
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp

from .aux_functions import safe_sqrt

__all__ = [
    "FULL_CHORD",
    "layer_radii",
    "impact_parameters_1d",
    "path_tensor",
    "path_matrix_1d",
    "uniform_theta_edges",
    "uniform_sector_fractions",
    "shared_ray_grid",
    "sector_geometry",
    "day_night_slices",
]

# One slice spanning the whole chord (dayside and nightside): the 1D case.
FULL_CHORD = jnp.array([-0.5 * jnp.pi, 0.5 * jnp.pi])


def layer_radii(state: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Lower and upper layer radii (nlay,) of a column."""
    R0, z_lev = state["R0"], state["z_lev"]
    return R0 + z_lev[:-1], R0 + z_lev[1:]


def impact_parameters_1d(state: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """1D ray grid: one ray per layer midpoint.

    Returns impact parameters b (nray,), annulus weights 2 b db (nray,) (the factor
    pi cancels against pi R_s^2), and the opaque base radius (bottom level).
    """
    r_low, r_up = layer_radii(state)
    b = state["R0"] + state["z_lay"]
    return b, 2.0 * b * (r_up - r_low), r_low[0]


def _half_chord(b, r_low, r_up, r_near, r_far):
    """Chord length of rays b (nray, 1) through layers [r_low, r_up] (1, nlay), restricted to
    the radius window [r_near, r_far] (nray, 1) of one half of the line of sight."""
    lo = jnp.maximum(r_low, r_near)
    hi = jnp.minimum(r_up, r_far)
    # (r - b)(r + b) instead of r^2 - b^2: exactly zero at the tangent point (r = b) and no
    # cancellation of two ~R_p^2 numbers (which XLA may also contract into an FMA under jit).
    length = safe_sqrt((hi - b) * (hi + b)) - safe_sqrt((lo - b) * (lo + b))
    return jnp.where(hi > lo, length, 0.0)


def _window_radius(b, theta):
    """Radius b / cos(theta) reached by a ray at angle |theta| (inf at |theta| = pi/2)."""
    c = jnp.cos(theta)
    open_end = c <= 1e-12
    return jnp.where(open_end, jnp.inf, b / jnp.where(open_end, 1.0, c))


def path_tensor(
    b: jnp.ndarray,           # (nray,)
    r_low: jnp.ndarray,       # (nslice, nlay)
    r_up: jnp.ndarray,        # (nslice, nlay)
    theta_edges: jnp.ndarray, # (nslice + 1,), increasing, within [-pi/2, pi/2]
) -> jnp.ndarray:
    """P[i, k, l]: chord length of ray i in layer l of slice k, divided by that layer's dr.

    A slice that straddles the terminator plane (theta = 0) contributes both halves of
    its chord: [0, |theta_lo|] on the dayside and [0, theta_hi] on the nightside.
    """
    b2 = b[:, None, None]                      # (nray, 1, 1)
    rl, ru = r_low[None], r_up[None]           # (1, nslice, nlay)
    t_lo, t_hi = theta_edges[:-1], theta_edges[1:]

    def window(a, c):
        # radius window [b / cos(a), b / cos(c)] for 0 <= a <= c
        return _window_radius(b2, a[None, :, None]), _window_radius(b2, c[None, :, None])

    # Nightside part of each slice: angles [max(t_lo, 0), t_hi] if t_hi > 0.
    n_a = jnp.maximum(t_lo, 0.0)
    n_near, n_far = window(n_a, jnp.maximum(t_hi, 0.0))
    night = jnp.where((t_hi > 0.0)[None, :, None], _half_chord(b2, rl, ru, n_near, n_far), 0.0)

    # Dayside part: angles [max(-t_hi, 0), -t_lo] if t_lo < 0.
    d_a = jnp.maximum(-t_hi, 0.0)
    d_near, d_far = window(d_a, jnp.maximum(-t_lo, 0.0))
    day = jnp.where((t_lo < 0.0)[None, :, None], _half_chord(b2, rl, ru, d_near, d_far), 0.0)

    return (night + day) / (ru - rl)


def path_matrix_1d(state: Dict[str, jnp.ndarray], b: jnp.ndarray) -> jnp.ndarray:
    """1D path matrix (nray, nlay): one column covering the whole chord."""
    r_low, r_up = layer_radii(state)
    return path_tensor(b, r_low[None], r_up[None], FULL_CHORD)[:, 0, :]


# ---------------------------------------------------------------------------
# Multi-column geometry (sectors x zenith slices)
# ---------------------------------------------------------------------------

def uniform_theta_edges(n_slice: int) -> jnp.ndarray:
    """Zenith-slice edges, equal widths from -pi/2 (dayside) to pi/2 (nightside)."""
    return jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, n_slice + 1)


def uniform_sector_fractions(n_sector: int) -> jnp.ndarray:
    """Azimuthal fraction of the annulus covered by each sector (equal sectors)."""
    return jnp.full((n_sector,), 1.0 / n_sector)


def shared_ray_grid(r_low: jnp.ndarray, r_up: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Impact parameters shared by all columns.

    ``r_low``, ``r_up``: layer radii of every column, shape (..., nlay) (any leading axes).
    The ray edges span [lowest column bottom, highest column top] with the layer spacing of
    the column of largest vertical extent (as in TRIDENT); rays sit at edge midpoints. For a
    single column this is exactly the 1D grid (one ray per layer midpoint).

    Returns b (nray,), full-annulus weights 2 b db (nray,), and the opaque base radius.
    """
    lo = r_low.reshape(-1, r_low.shape[-1])
    up = r_up.reshape(-1, r_up.shape[-1])
    bottom, top = jnp.min(lo[:, 0]), jnp.max(up[:, -1])
    ref_idx = jnp.argmax(up[:, -1] - lo[:, 0])
    ref = jnp.concatenate([lo[ref_idx], up[ref_idx, -1:]])        # (nlev,) level radii of the reference
    edges = bottom + (ref - ref[0]) * ((top - bottom) / (ref[-1] - ref[0]))
    b = 0.5 * (edges[:-1] + edges[1:])
    return b, 2.0 * b * jnp.diff(edges), bottom


def sector_geometry(
    b: jnp.ndarray,           # (nray,)
    r_low: jnp.ndarray,       # (nslice, nlay) columns of one sector
    r_up: jnp.ndarray,        # (nslice, nlay)
    theta_edges: jnp.ndarray, # (nslice + 1,)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Path tensor P (nray, nslice, nlay) and opaque-ray flags (nray,) for one sector.

    A ray is opaque if, in any slice, its lowest radius there (b at the terminator plane,
    b / cos(theta) otherwise) lies below that column's bottom: it hits the deep atmosphere.
    """
    P = path_tensor(b, r_low, r_up, theta_edges)
    t_lo, t_hi = theta_edges[:-1], theta_edges[1:]
    theta_near = jnp.where((t_lo <= 0.0) & (t_hi >= 0.0), 0.0, jnp.minimum(jnp.abs(t_lo), jnp.abs(t_hi)))
    r_near = _window_radius(b[:, None], theta_near[None, :])                # (nray, nslice)
    opaque = jnp.any(r_near < r_low[None, :, 0] * (1.0 - 1e-12), axis=1)
    return P, opaque


def day_night_slices(n_zenith: int, beta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Zenith slices and interpolation weights for a day-night (2D) terminator (TRIDENT).

    Slice 0 is the dayside (theta from -pi/2 to -beta/2), the last slice the nightside
    (beta/2 to pi/2), and the ``n_zenith - 2`` slices between share the transition region
    |theta| < beta/2 equally. ``beta`` (radians, may be traced) is the opening angle.

    Returns the slice edges (n_zenith + 1,) and the night weight of each slice (n_zenith,):
    0 for the day slice, 1 for the night slice, and (theta_centre + beta/2) / beta for the
    transition slices, i.e. linear in angle across the transition.
    """
    if n_zenith < 2:
        raise ValueError("A day-night terminator needs n_zenith >= 2.")
    half = 0.5 * jnp.pi
    if n_zenith == 2:
        return jnp.array([-half, 0.0, half]), jnp.array([0.0, 1.0])
    n_tr = n_zenith - 2
    inner = -0.5 * beta + beta * jnp.arange(n_tr + 1) / n_tr            # transition edges
    edges = jnp.concatenate([jnp.array([-half]), inner, jnp.array([half])])
    centres = 0.5 * (inner[:-1] + inner[1:])
    w_tr = (centres + 0.5 * beta) / jnp.where(beta > 0, beta, 1.0)
    w = jnp.concatenate([jnp.zeros(1), w_tr, jnp.ones(1)])
    return edges, w
