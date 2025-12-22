"""
RT_alb_1D.py
============
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax.numpy as jnp

from . import build_opacities as XS

__all__ = ["compute_albedo_spectrum_1d"]


def _get_ck_weights(state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    g_weights = state.get("g_weights")
    if g_weights is not None:
        return g_weights
    if not XS.has_ck_data():
        raise RuntimeError("c-k g-weights not built; run build_opacities() with ck tables.")
    g_weights = XS.ck_g_weights()
    if g_weights.ndim > 1:
        g_weights = g_weights[0]
    return g_weights


def _sum_opacity_components_ck(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    Return the summed opacity grid for correlated-k mode.

    Line opacity has shape (nlay, nwl, ng).
    Other opacities have shape (nlay, nwl) and are broadcast over g-dimension.
    Returns shape (nlay, nwl, ng).
    """
    nlay = state["nlay"]
    nwl = state["nwl"]

    line_opacity = opacity_components.get("line")
    if line_opacity is None:
        ng = _get_ck_weights(state).shape[-1]
        line_opacity = jnp.zeros((nlay, nwl, ng))

    zeros_2d = jnp.zeros((nlay, nwl), dtype=line_opacity.dtype)
    component_keys_2d = ("rayleigh", "cia", "special", "cloud")
    components_2d = jnp.stack([opacity_components.get(k, zeros_2d) for k in component_keys_2d], axis=0)
    summed_2d = jnp.sum(components_2d, axis=0)

    return line_opacity + summed_2d[:, :, None]


def _sum_opacity_components_lbl(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """Return the summed opacity grid for all provided components (LBL mode)."""
    nlay = state["nlay"]
    nwl = state["nwl"]

    if not opacity_components:
        return jnp.zeros((nlay, nwl))

    component_keys = ("line", "rayleigh", "cia", "special", "cloud")
    first = next((opacity_components.get(k) for k in component_keys if k in opacity_components), None)
    if first is None:
        return jnp.zeros((nlay, nwl))

    zeros = jnp.zeros_like(first)
    stacked = jnp.stack([opacity_components.get(k, zeros) for k in component_keys], axis=0)
    return jnp.sum(stacked, axis=0)


def compute_albedo_spectrum_1d(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Placeholder shortwave albedo RT kernel.

    Parameters
    ----------
    state
        Standard forward-model state dict (see `build_model.forward_model`).
        Expected keys include `wl`, `nwl`, `nlay`, `ck`, and optionally `contri_func`.
    params
        Retrieval parameters (may include cloud patchiness, phase angle, etc.).
    opacity_components
        Opacity component mapping (line/rayleigh/cia/special/cloud and cloud scattering props).

    Returns
    -------
    (alb_spectrum, contrib_func)
        `alb_spectrum`: (nwl,) reflected-light metric (TBD: geometric albedo vs reflectance).
        `contrib_func`: (nlay, nwl) normalized contribution function if enabled, else zeros.
    """
    _ = (state, params, opacity_components)
    raise NotImplementedError(
        "compute_albedo_spectrum_1d is a placeholder. "
        "Implement a shortwave scattering/reflection solver and define the albedo convention."
    )

