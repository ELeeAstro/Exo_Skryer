"""
build_model.py
==============

Overview:
    Build a JAX-jitted forward model for the chosen physics / opacity / RT setup.
"""

from __future__ import annotations
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from data_constants import kb, amu, R_jup, R_sun, bar

from vert_alt import hypsometric, hypsometric_variable_g
from vert_struct import isothermal, Milne, Guillot, Line, Barstow

from opacity_line import zero_line_opacity, compute_line_opacity
from opacity_ray import zero_ray_opacity, compute_ray_opacity
from opacity_cia import zero_cia_opacity, compute_cia_opacity
from opacity_cloud import zero_cloud_opacity, compute_grey_cloud_opacity, compute_f18_cloud_opacity

import build_opacities as XS
from RT_trans_1D import compute_transit_depth_1d
from vert_mu import compute_mean_molecular_weight
from instru_convolve import apply_response_functions


def _gather_param_mixing_ratios(params: Dict[str, jnp.ndarray], nlay: int) -> Dict[str, jnp.ndarray]:
    ratios: Dict[str, jnp.ndarray] = {}
    for key, value in params.items():
        if not key.startswith("f_"):
            continue
        species = key[2:]
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            ratios[species] = jnp.full((nlay,), arr)
        elif arr.ndim == 1:
            if arr.shape[0] != nlay:
                raise ValueError(f"Mixing ratio '{key}' has length {arr.shape[0]}, expected {nlay}.")
            ratios[species] = arr
        else:
            raise ValueError(f"Mixing ratio '{key}' has unsupported shape {arr.shape}.")
    return ratios


def build_forward_model(cfg, obs, return_highres: bool = False):

    # Example: number of layers from YAML
    nlay = int(getattr(cfg.physics, "nlay", 99))
    nlev = nlay + 1

    # Observational wavelengths/widths (currently only used by bandpass loader, not here)
    obs_wl_np = np.asarray(obs["wl"], dtype=float)
    obs_dwl_np = np.asarray(obs["dwl"], dtype=float)
    lam_obs = jnp.asarray(obs_wl_np)
    dlam_obs = jnp.asarray(obs_dwl_np)

    # Get the kernel for forward model
    phys = cfg.physics

    vert_struct = getattr(phys, "vert_struct", "isothermal")
    if vert_struct == "isothermal":
        vert_kernel = isothermal
    elif vert_struct == "Milne":
        vert_kernel = Milne
    elif vert_struct == "Guillot":
        vert_kernel = Guillot
    elif vert_struct == "Line":
        vert_kernel = Line
    elif vert_struct == "Barstow":
        vert_kernel = Barstow
    else:
        raise NotImplementedError(f"Unknown vert_struct='{vert_struct}'")

    line_opac_scheme = getattr(phys, "opac_line", "None")
    if line_opac_scheme == "None":
        print(f"[info] Line opacity is None:", line_opac_scheme)
        line_opac_kernel = zero_line_opacity
    elif line_opac_scheme == "lbl":
        line_opac_kernel = compute_line_opacity
    else:
        raise NotImplementedError(f"Unknown line_opac_scheme='{line_opac_scheme}'")

    ray_opac_scheme = getattr(phys, "opac_ray", "None")
    if ray_opac_scheme == "None":
        print(f"[info] Rayleigh opacity is None:", ray_opac_scheme)
        ray_opac_kernel = zero_ray_opacity
    elif ray_opac_scheme == "lbl":
        ray_opac_kernel = compute_ray_opacity
    else:
        raise NotImplementedError(f"Unknown ray_opac_scheme='{ray_opac_scheme}'")

    cia_opac_scheme = getattr(phys, "opac_cia", "None")
    if cia_opac_scheme == "None":
        print(f"[info] CIA opacity is None:", cia_opac_scheme)
        cia_opac_kernel = zero_cia_opacity
    elif cia_opac_scheme == "lbl":
        cia_opac_kernel = compute_cia_opacity
    else:
        raise NotImplementedError(f"Unknown cia_opac_scheme='{cia_opac_scheme}'")

    cld_opac_scheme = getattr(phys, "opac_cloud", "None")
    if cld_opac_scheme == "None":
        print(f"[info] Cloud opacity is None:", cld_opac_scheme)
        cld_opac_kernel = zero_cloud_opacity
    elif cld_opac_scheme == "grey":
        cld_opac_kernel = compute_grey_cloud_opacity
    elif cld_opac_scheme == "F18":
        cld_opac_kernel = compute_f18_cloud_opacity
    else:
        raise NotImplementedError(f"Unknown cld_opac_scheme='{cld_opac_scheme}'")

    rt_scheme = getattr(phys, "rt_scheme", "transit_1d")
    if rt_scheme == "transit_1d":
        rt_kernel = compute_transit_depth_1d
    else:
        raise NotImplementedError(f"Unknown rt_scheme='{rt_scheme}'")

    # High-resolution master grid (must match cut_grid used in bandpass loader)
    wl_hi_array = np.asarray(XS.master_wavelength_cut(), dtype=float)
    wl_hi = jnp.asarray(wl_hi_array)

    @jax.jit
    def forward_model(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:

        # High-res wavelength grid
        wl = wl_hi
        nwl = jnp.size(wl)

        # Planet and star radii (R0 is radius at p_bot)
        R0 = jnp.asarray(params["R_p"]) * R_jup
        R_s = jnp.asarray(params["R_s"]) * R_sun

        # Atmospheric pressure grid
        p_bot = jnp.asarray(params["p_bot"]) * bar
        p_top = jnp.asarray(params["p_top"]) * bar
        p_lev = jnp.logspace(jnp.log10(p_bot), jnp.log10(p_top), nlev)

        # Vertical atmospheric T-p structure
        T_lay = vert_kernel(p_lev, params)

        # Mean molecular weight and mixing ratios
        if "mu" in params:
            mu_const = jnp.asarray(params["mu"])
            mu_lay = jnp.full((nlay,), mu_const)
            mix_ratios = _gather_param_mixing_ratios(params, nlay)
        else:
            layer_template = jnp.ones((nlay,))
            mu_lay, mix_ratios, mu_dynamic = compute_mean_molecular_weight(layer_template, params)
            if mu_lay is None or (not mu_dynamic):
                raise ValueError("Dynamic mean molecular weight failed; provide 'mu' parameter or fix vert_mu.")

        # Vertical altitude calculation
        z_lev = hypsometric_variable_g(p_lev, T_lay, mu_lay, params)
        z_lay = (z_lev[:-1] + z_lev[1:]) / 2.0
        dz = jnp.diff(z_lev)

        # Interpolate to find p_lay (pressure at mid height)
        p_lay = 10.0**jnp.interp(z_lay, z_lev, jnp.log10(p_lev))

        # Atmospheric density and number density
        rho = (mu_lay * amu * p_lay) / (kb * T_lay)
        nd = p_lay / (kb * T_lay)

        # State dictionary for physics kernels
        state = {
            "nwl": nwl,
            "nlay": nlay,
            "wl": wl,
            "R0": R0,
            "R_s": R_s,
            "p_lev": p_lev,
            "mu_lay": mu_lay,
            "T_lay": T_lay,
            "z_lev": z_lev,
            "z_lay": z_lay,
            "dz": dz,
            "p_lay": p_lay,
            "rho": rho,
            "nd": nd,
            "mixing_ratios": mix_ratios,
        }

        # Opacity components
        k_line = line_opac_kernel(state, params)
        k_ray = ray_opac_kernel(state, params)
        k_cia = cia_opac_kernel(state, params)
        k_cld = cld_opac_kernel(state, params)

        opacity_components = {
            "line": k_line,
            "rayleigh": k_ray,
            "cia": k_cia,
            "cloud": k_cld,
        }

        # Radiative transfer
        D_hires = rt_kernel(state, params, opacity_components)

        # Instrumental convolution → binned spectrum
        D_bin = apply_response_functions(wl, D_hires)

        if return_highres:
            return {"hires": D_hires, "binned": D_bin}

        return D_bin

    return forward_model
