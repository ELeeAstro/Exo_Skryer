"""
RT_alb_core_os.py
=================

Reflected-light (albedo) spectra with opacity-sampled (os) opacities (planned registry
entry: albedo_1d/os). Mirrors ``RT_em_core_os``: sums the opacity components, builds the
layer optical properties and calls a solver from ``RT_alb_solvers``.

The entry point is a stub until a shortwave solver and the albedo convention (geometric
albedo vs. reflectance at a phase angle) are defined. The opacity summation is in place.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax.numpy as jnp

from .aux_functions import sum_present

__all__ = ["compute_albedo_spectrum_1d_os"]

_COMPONENTS = ("line", "rayleigh", "cia", "special", "cloud")


def _sum_opacity_components_os(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """Summed opacity (nlay, nwl) of all components present (zeros if none)."""
    total = sum_present(opacity_components, _COMPONENTS) if opacity_components else None
    if total is None:
        return jnp.zeros((state["rho_lay"].shape[0], state["wl"].shape[0]))
    return total


def compute_albedo_spectrum_1d_os(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Mapping[str, jnp.ndarray] | None = None,
    albedo_solver=None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reflected-light spectrum (nwl,) and contribution function (nlay, nwl) (stub).

    Same signature as the emission entry points: ``albedo_solver`` comes from
    ``RT_alb_solvers.get_albedo_solver``.
    """
    raise NotImplementedError(
        "compute_albedo_spectrum_1d_os is a stub. Implement a shortwave solver in RT_alb_solvers "
        "and define the albedo convention."
    )
