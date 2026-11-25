"""
RT_alb_core_ck.py
=================

Reflected-light (albedo) spectra with correlated-k opacities (planned registry entry:
albedo_1d/ck). Mirrors ``RT_em_core_ck``: g-resolved layer optical properties are passed
to a solver from ``RT_alb_solvers`` and the result is g-weighted.

The entry point is a stub until a shortwave solver and the albedo convention are defined.
The opacity summation is in place.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax.numpy as jnp

from .aux_functions import sum_present

__all__ = ["compute_albedo_spectrum_1d_ck"]


def _get_ck_weights(opac: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    g_weights = opac.get("g_weights")
    if g_weights is None:
        raise RuntimeError("Missing opac['g_weights'] for c-k integration.")
    return g_weights[0] if g_weights.ndim > 1 else g_weights


def _sum_opacity_components_ck(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """Total opacity (nlay, nwl, ng): line (nlay, nwl, ng) plus the (nlay, nwl) components broadcast over g."""
    line_opacity = opacity_components.get("line")
    if line_opacity is None:
        ng = _get_ck_weights(opac).shape[-1]
        line_opacity = jnp.zeros((state["rho_lay"].shape[0], state["wl"].shape[0], ng))
    summed_2d = sum_present(opacity_components, ("rayleigh", "cia", "special", "cloud"))
    if summed_2d is None:
        return line_opacity
    return line_opacity + summed_2d[:, :, None]


def compute_albedo_spectrum_1d_ck(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Mapping[str, jnp.ndarray],
    albedo_solver=None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reflected-light spectrum (nwl,) and contribution function (nlay, nwl) (stub)."""
    raise NotImplementedError(
        "compute_albedo_spectrum_1d_ck is a stub. Implement a shortwave solver in RT_alb_solvers "
        "and define the albedo convention."
    )
