"""
RT_em_1D_ck.py
==============

Correlated-k emission spectrum: the RT solver is run per g-point and the
fluxes summed with the g weights.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from .data_constants import kb, h, c_light, pc
from .RT_em_schemes import solve_alpha_eaa

__all__ = ["compute_emission_spectrum_1d_ck"]


def _get_ck_weights(opac: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    g_weights = opac.get("g_weights")
    if g_weights is None:
        raise RuntimeError("Missing opac['g_weights'] for c-k integration.")
    return g_weights


def _planck_lambda(wavelength_cm: jnp.ndarray, temperature: jnp.ndarray) -> jnp.ndarray:
    exponent = (h * c_light) / (wavelength_cm * kb * jnp.maximum(temperature, 1.0))
    expm1 = jnp.expm1(jnp.clip(exponent, None, 80.0))
    prefactor = 2.0 * h * c_light**2 / (wavelength_cm**5)
    return prefactor / jnp.maximum(expm1, 1e-300)


def _scattering_2d(
    k_ray: jnp.ndarray,
    k_cloud_ext: jnp.ndarray,
    cloud_ssa: jnp.ndarray,
    cloud_g: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Scattering opacity and scattering-weighted asymmetry, (nlay, nwl); same for every g-point."""
    # Clamped because MADT can give a slightly negative Q_sca.
    k_cloud_scat = jnp.maximum(cloud_ssa * k_cloud_ext, 0.0)
    k_tot_scat = k_ray + k_cloud_scat
    # Rayleigh contributes g = 0.
    g = k_cloud_scat * cloud_g / jnp.maximum(k_tot_scat, 1.0e-30)
    return k_tot_scat, g


def _scale_flux_ratio(
    flux: jnp.ndarray,
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    dtype = flux.dtype
    stellar_flux = jnp.asarray(state.get("stellar_flux", jnp.ones_like(flux)), dtype=dtype)
    has_stellar_flux = jnp.asarray(state.get("has_stellar_flux", 0), dtype=jnp.int32)
    f_star_param = jnp.asarray(params.get("F_star", jnp.ones_like(stellar_flux)), dtype=dtype)
    use_stellar = has_stellar_flux != 0
    F_star = jnp.where(use_stellar, stellar_flux, f_star_param)
    R0 = jnp.asarray(state["R0"], dtype=dtype)
    R_s = jnp.asarray(state["R_s"], dtype=dtype)
    scale = (R0**2) / (F_star * (R_s**2))
    return flux * scale


def compute_emission_spectrum_1d_ck(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray],
    emission_solver=solve_alpha_eaa,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    contri_func = state.get("contri_func", False)
    dtype = state["wl"].dtype
    wl_cm = jnp.asarray(state["wl"], dtype=dtype) * 1.0e-4
    T_lev = jnp.asarray(state["T_lev"], dtype=dtype)
    rho_lay = jnp.asarray(state["rho_lay"], dtype=dtype)
    dz = jnp.asarray(state["dz"], dtype=dtype)
    be_levels = _planck_lambda(wl_cm[None, :], T_lev[:, None])
    if "T_int" in params:
        T_int = jnp.asarray(params["T_int"], dtype=dtype)
        be_internal = _planck_lambda(wl_cm[None, :], T_int[None, None])[0]
    else:
        be_internal = jnp.zeros_like(be_levels[-1])

    # Loop over g-points, building each g-point's (nlay, nwl) optical depth and
    # single-scattering albedo inside the step from the line slice and the 2D
    # continuum/cloud terms, so no (nlay, nwl, ng) arrays beyond the line opacity
    # itself are formed. With f_cloud, the cloudy and clear columns share the step.
    # jax.checkpoint stores only the carry per g-point under reverse-mode AD
    # (rather than every g-point's solver intermediates).
    nlay, nwl = rho_lay.shape[0], wl_cm.shape[0]
    g_weights = _get_ck_weights(opac)
    line = opacity_components.get("line")
    if line is None:
        line = jnp.zeros((nlay, nwl, g_weights.shape[-1]), dtype=dtype)
    line_by_g = jnp.moveaxis(line, -1, 0)
    g_weights = g_weights[: line_by_g.shape[0]]

    zeros_2d = jnp.zeros((nlay, nwl), dtype=line.dtype)

    def _comp(name):
        arr = opacity_components.get(name)
        return zeros_2d if arr is None else arr

    k_2d = jnp.sum(jnp.stack([_comp(k) for k in ("rayleigh", "cia", "special", "cloud")], axis=0), axis=0)
    k_ray = _comp("rayleigh")
    cloud_ext = _comp("cloud")
    k_scat_cloud, g_cloud = _scattering_2d(k_ray, cloud_ext, _comp("cloud_ssa"), _comp("cloud_g"))
    two_column = "f_cloud" in params and "cloud" in opacity_components
    if two_column:
        k_scat_clear, g_clear = _scattering_2d(k_ray, zeros_2d, zeros_2d, zeros_2d)

    def _column(k_tot, k_scat, g_eff):
        ssa = jnp.clip(k_scat / jnp.maximum(k_tot, 1.0e-30), 0.0, 0.99)
        dtau = k_tot * rho_lay[:, None] * dz[:, None]
        olr_g, contrib_g = emission_solver(
            be_levels, dtau, ssa, g_eff, be_internal, return_layer_contrib=bool(contri_func)
        )
        return (olr_g, contrib_g) if contri_func else (olr_g,)

    @jax.checkpoint
    def _g_step(acc, x):
        line_g, weight = x
        k_tot = line_g + k_2d
        out = _column(k_tot, k_scat_cloud, g_cloud)
        if two_column:
            out = out + _column(k_tot - cloud_ext, k_scat_clear, g_clear)
        weight = weight.astype(dtype)
        return tuple(a + weight * b for a, b in zip(acc, out)), None

    n_out = (2 if contri_func else 1) * (2 if two_column else 1)
    acc0 = tuple(jnp.zeros((nlay, nwl), dtype=dtype) if (contri_func and i % 2) else jnp.zeros_like(be_levels[0])
                 for i in range(n_out))
    acc, _ = lax.scan(_g_step, acc0, (line_by_g, g_weights))

    zero_contrib = jnp.zeros((nlay, nwl), dtype=be_levels.dtype)
    if contri_func:
        olr_cloud, layer_contrib_cloud = acc[0], acc[1]
    else:
        olr_cloud, layer_contrib_cloud = acc[0], zero_contrib

    if two_column:
        f_cloud = jnp.clip(params["f_cloud"], 0.0, 1.0)
        if contri_func:
            olr_clear, layer_contrib_clear = acc[2], acc[3]
        else:
            olr_clear, layer_contrib_clear = acc[1], zero_contrib
        olr = f_cloud * olr_cloud + (1.0 - f_cloud) * olr_clear
        layer_contrib_flux = f_cloud * layer_contrib_cloud + (1.0 - f_cloud) * layer_contrib_clear
    else:
        olr = olr_cloud
        layer_contrib_flux = layer_contrib_cloud

    s_dilute = jnp.asarray(params.get("s_dilute", 1.0))
    top_flux = s_dilute * olr
    if state.get("is_brown_dwarf", False):
        R0 = jnp.asarray(state["R0"], dtype=top_flux.dtype)
        D = params["D"]
        f_p = 10.0 ** jnp.asarray(params.get("log_10_f_p", 0.0), dtype=top_flux.dtype)
        distance = D * pc
        final_spectrum = top_flux * f_p * (R0 / distance) ** 2
    else:
        final_spectrum = _scale_flux_ratio(top_flux, state, params)

    if contri_func:
        layer_contrib = jnp.clip(layer_contrib_flux, 0.0)
        contrib_func_norm = layer_contrib / jnp.maximum(layer_contrib.sum(axis=0, keepdims=True), 1e-30)
    else:
        contrib_func_norm = jnp.zeros((state["dz"].shape[0], final_spectrum.shape[0]), dtype=final_spectrum.dtype)

    return final_spectrum, contrib_func_norm
