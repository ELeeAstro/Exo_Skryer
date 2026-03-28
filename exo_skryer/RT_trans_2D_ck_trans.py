"""
RT_trans_2D_ck_trans.py
=======================
"""

from __future__ import annotations

from typing import Dict, Mapping

import jax.numpy as jnp

from .RT_trans_1D_ck_trans import compute_transit_depth_1d_ck_trans

__all__ = ["compute_transit_depth_2d_ck_trans"]


def compute_transit_depth_2d_ck_trans(
    state_east: Dict[str, jnp.ndarray],
    params_east: Dict[str, jnp.ndarray],
    opacity_east: Mapping[str, jnp.ndarray],
    state_west: Dict[str, jnp.ndarray],
    params_west: Dict[str, jnp.ndarray],
    opacity_west: Mapping[str, jnp.ndarray],
    opac: Mapping[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Compute separate east/west CK-TRANS transmission spectra for transit_2d."""
    state_east = dict(state_east)
    state_west = dict(state_west)
    state_east["contri_func"] = False
    state_west["contri_func"] = False

    limb_east, _ = compute_transit_depth_1d_ck_trans(state_east, params_east, opacity_east, opac)
    limb_west, _ = compute_transit_depth_1d_ck_trans(state_west, params_west, opacity_west, opac)

    return {
        "hires_east": limb_east,
        "hires_west": limb_west,
    }
