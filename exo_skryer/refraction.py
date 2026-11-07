"""
refraction.py
=============

Approximate refraction support for transmission spectroscopy.

Current implementation: "cutoff" mode (option A) that applies a refractive
boundary (fully opaque below a wavelength-dependent impact parameter) without
curved-ray optical-depth integration.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp
from jax import lax

from . import registry_ray as XR
from .data_constants import AU, kb, amu

__all__ = ["refraction_cutoff_mask", "maybe_refraction_cutoff_mask", "refraction_extra_tau"]


def _blocked_fraction(alpha: jnp.ndarray, theta_star: jnp.ndarray) -> jnp.ndarray:
    """Fraction of each ray's annulus below the refractive boundary.

    ``alpha`` (nlay, nwl) is the bending angle at the layer midpoints, bottom
    first, and decreases with height. The boundary is placed where
    log(alpha / theta_star) crosses zero, interpolated linearly in layer index
    between the bracketing midpoints; each ray covers +-0.5 layer around its
    midpoint. This moves the boundary continuously through the grid instead of
    switching whole rays on and off, so gradients stay finite.
    """
    nlay = alpha.shape[0]
    q = jnp.log(jnp.maximum(alpha, 1e-300)) - jnp.log(jnp.maximum(theta_star, 1e-300))
    count = jnp.sum(q > 0.0, axis=0)                      # blocked rays per wavelength
    k = jnp.clip(count - 1, 0, nlay - 2)
    q_lo = jnp.take_along_axis(q, k[None, :], axis=0)[0]
    q_hi = jnp.take_along_axis(q, (k + 1)[None, :], axis=0)[0]
    den = q_lo - q_hi
    ok = den > 0.0
    s = jnp.where(ok, k + q_lo / jnp.where(ok, den, 1.0), count - 0.5)
    j = jnp.arange(nlay, dtype=alpha.dtype)[:, None]
    return jnp.clip(s[None, :] - j + 0.5, 0.0, 1.0)


def refraction_extra_tau(blocked: jnp.ndarray) -> jnp.ndarray:
    """Optical depth that makes a fraction ``blocked`` of a ray opaque.

    exp(-(tau + extra)) = (1 - blocked) * exp(-tau); fully blocked rays get
    1e30, as the hard cutoff did.
    """
    full = blocked >= 1.0
    part = jnp.where(full, 0.0, blocked)
    return jnp.where(full, 1.0e30, -jnp.log1p(-part))


def refraction_cutoff_mask(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Return the fraction of each ray blocked by refraction.

    The mask is defined on the same "impact parameter grid" used by the current
    transit RT kernels: `b ≈ R0 + z_lay` (layer midpoints).

    For each (layer, wavelength), we estimate the bending angle using the
    exponential-atmosphere approximation:

        alpha(b, λ) ≈ (n(b, λ) - 1) * sqrt(2π b / H)

    A ray is blocked where

        alpha(b, λ) > theta_star,    theta_star = asin(R_s / a)

    with `a = a_sm * AU`. The boundary is interpolated between layer midpoints,
    so the ray holding it is partly blocked (see `_blocked_fraction`).

    Parameters
    ----------
    state : dict
        Must contain `R0`, `R_s`, `z_lay`, `T_lay`, `mu_lay`, `nd_lay`, and `vmr_lay`.
    params : dict
        Must contain `log_10_g` and `a_sm` (AU).
    opac : dict
        Must contain `ray_refractivity_coeff_table` aligned with
        `registry_ray.ray_species_names()`.

    Returns
    -------
    blocked : jnp.ndarray, shape (nlay, nwl), float in [0, 1]
        Fraction of each ray's annulus blocked by refraction. Add
        `refraction_extra_tau(blocked)` to the path optical depth.
    """
    nwl = state["wl"].shape[0]
    nlay = state["nd_lay"].shape[0]

    if "ray_refractivity_coeff_table" not in opac:
        raise RuntimeError("Refraction requested but Rayleigh refractivity tables are missing from opac cache.")

    # Stellar angular radius at the planet
    a_cm = params["a_sm"] * AU
    theta_star = jnp.arcsin(jnp.clip(state["R_s"] / a_cm, 0.0, 1.0))  # scalar

    # Scale height at each layer midpoint (include spherical gravity correction)
    R0 = state["R0"]
    z_lay = state["z_lay"]
    b = R0 + z_lay  # (nlay,)

    g0 = 10.0 ** params["log_10_g"]
    g_z = g0 * (R0 / b) ** 2
    H = (kb * state["T_lay"]) / (state["mu_lay"] * amu * g_z)  # (nlay,)

    # Build (n-1)(layer, wl) from STP refractivities + ideal-gas scaling with number density.
    refractivity_coeff = opac["ray_refractivity_coeff_table"]  # (nspec, nwl)
    species_names = XR.ray_runtime_species_order()
    vmr_lay = state["vmr_lay"]
    nd_lay = state["nd_lay"]  # (nlay,)
    mixing_ratios = jnp.stack(
        [jnp.broadcast_to(vmr_lay[name], (nlay,)) for name in species_names],
        axis=0,
    )  # (n_species, nlay)
    nm1_coeff = jnp.einsum("sl,sw->lw", mixing_ratios, refractivity_coeff)

    nm1_layer = nd_lay[:, None] * nm1_coeff  # (nlay, nwl)
    nm1_layer = jnp.maximum(nm1_layer, 0.0)

    # Exponential-atmosphere bending approximation
    alpha = nm1_layer * jnp.sqrt(2.0 * jnp.pi * b[:, None] / jnp.maximum(H[:, None], 1.0))

    return _blocked_fraction(alpha, theta_star)


def maybe_refraction_cutoff_mask(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray] | None,
) -> jnp.ndarray:
    """Return the blocked fraction per ray, or zeros when refraction is off.

    This avoids Python-side branching on traced `refraction_mode` values inside
    jitted/vmapped transit kernels.
    """
    if "z_lay" in state:
        nlay = state["z_lay"].shape[0]
    elif "dz" in state:
        nlay = state["dz"].shape[0]
    else:
        nlay = int(state["nlay"])

    if "wl" in state:
        nwl = state["wl"].shape[0]
    else:
        nwl = int(state["nwl"])

    zeros = jnp.zeros((nlay, nwl), dtype=jnp.float64)

    if opac is None:
        return zeros
    if "ray_refractivity_coeff_table" not in opac:
        return zeros
    required_state = ("wl", "T_lay", "mu_lay", "nd_lay", "vmr_lay", "z_lay", "R0", "R_s")
    required_params = ("log_10_g", "a_sm")
    if any(name not in state for name in required_state):
        return zeros
    if any(name not in params for name in required_params):
        return zeros

    refraction_mode = jnp.asarray(state.get("refraction_mode", 0))
    return lax.cond(
        refraction_mode == 1,
        lambda _: refraction_cutoff_mask(state, params, opac),
        lambda _: zeros,
        operand=None,
    )
