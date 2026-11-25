"""
RT_alb_solvers.py
=================

Reflected-light (albedo) radiative transfer solvers, to be selected with
``physics.alb_scheme`` (planned). Mirrors ``RT_em_solvers``: every solver takes
bottom-first layer optical properties and returns the reflected flux at the top of
the atmosphere for a given illumination geometry.

Planned solvers (stubs for now):

- ``toon89``: Toon et al. (1989) two-stream solution for the shortwave (delta-Eddington /
  hemispheric-mean closures), as in Exo-FMS sw_Toon89.
- ``adding``: multi-stream doubling-adding method (e.g. Exo-FMS sw_adding / DISORT-like),
  for accurate multiple scattering.

Common signature (to be confirmed when implemented)::

    solver(dtau_layers, ssa, g_phase, mu0, surface_albedo) -> (reflected_flux, layer_contrib)

with ``dtau_layers``, ``ssa`` and ``g_phase`` of shape (nlay, nwl), bottom first, ``mu0`` the
cosine of the stellar zenith angle, and the reflected flux normalised by the incident
stellar flux (so a geometric albedo follows from the phase-angle integration).
The entry points that call them are in ``RT_alb_core_os`` and ``RT_alb_core_ck``.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

__all__ = ["solve_toon89_sw", "solve_adding_sw", "get_albedo_solver"]


def solve_toon89_sw(
    dtau_layers: jnp.ndarray,
    ssa: jnp.ndarray,
    g_phase: jnp.ndarray,
    mu0: jnp.ndarray,
    surface_albedo: jnp.ndarray,
    return_layer_contrib: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Toon et al. (1989) two-stream shortwave solver (stub)."""
    raise NotImplementedError("solve_toon89_sw is a stub: the shortwave Toon89 solver is not implemented yet.")


def solve_adding_sw(
    dtau_layers: jnp.ndarray,
    ssa: jnp.ndarray,
    g_phase: jnp.ndarray,
    mu0: jnp.ndarray,
    surface_albedo: jnp.ndarray,
    return_layer_contrib: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-stream doubling-adding shortwave solver (stub)."""
    raise NotImplementedError("solve_adding_sw is a stub: the shortwave adding solver is not implemented yet.")


def get_albedo_solver(name: str):
    """Albedo RT solver by name: ``toon89`` or ``adding`` (both stubs for now)."""
    name = str(name).lower().strip()
    if name in ("toon89", "toon89_sw"):
        return solve_toon89_sw
    if name in ("adding", "doubling_adding"):
        return solve_adding_sw
    raise NotImplementedError(f"Unknown albedo scheme '{name}'. Options: toon89 | adding (both planned)")
