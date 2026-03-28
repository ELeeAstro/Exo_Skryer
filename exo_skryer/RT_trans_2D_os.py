"""
RT_trans_2D_os.py
=================
"""

from __future__ import annotations

from typing import Dict, Mapping

import jax.numpy as jnp

from .RT_trans_1D_os import _build_transit_geometry, _sum_opacity_components_os, _transit_depth_from_opacity
from .refraction import maybe_refraction_cutoff_mask

__all__ = ["compute_transit_depth_2d_os"]


def _compute_limb_transit(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Mapping[str, jnp.ndarray] | None,
):
    state = dict(state)
    state["contri_func"] = False

    refraction_mask = maybe_refraction_cutoff_mask(state, params, opac)

    geometry = _build_transit_geometry(state)
    k_tot = _sum_opacity_components_os(state, opacity_components)
    return _transit_depth_from_opacity(state, k_tot, geometry=geometry, refraction_mask=refraction_mask)


def compute_transit_depth_2d_os(
    state_east: Dict[str, jnp.ndarray],
    params_east: Dict[str, jnp.ndarray],
    opacity_east: Mapping[str, jnp.ndarray],
    state_west: Dict[str, jnp.ndarray],
    params_west: Dict[str, jnp.ndarray],
    opacity_west: Mapping[str, jnp.ndarray],
    opac: Mapping[str, jnp.ndarray] | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute separate east/west transmission spectra for the transit_2d mode."""
    limb_east = _compute_limb_transit(state_east, params_east, opacity_east, opac)
    limb_west = _compute_limb_transit(state_west, params_west, opacity_west, opac)

    return {
        "hires_east": limb_east,
        "hires_west": limb_west,
    }
