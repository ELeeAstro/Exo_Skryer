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
from opacity_ck import zero_ck_opacity, compute_ck_opacity
from opacity_ray import zero_ray_opacity, compute_ray_opacity
from opacity_cia import zero_cia_opacity, compute_cia_opacity
from opacity_cloud import zero_cloud_opacity, compute_grey_cloud_opacity, compute_f18_cloud_opacity

import build_opacities as XS
from RT_trans_1D import compute_transit_depth_1d
from vert_mu import compute_mean_molecular_weight
from instru_convolve import apply_response_functions

solar_h2 = 0.5
solar_he = 0.085114
solar_h2_he = solar_h2 + solar_he

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

    ck = False
    line_opac_scheme = getattr(phys, "opac_line", "None")
    if line_opac_scheme == "None":
        print(f"[info] Line opacity is None:", line_opac_scheme)
        line_opac_kernel = zero_line_opacity
    elif line_opac_scheme == "lbl":
        line_opac_kernel = compute_line_opacity
    elif line_opac_scheme == "ck":
        ck = True
        line_opac_kernel = compute_ck_opacity
    else:
        raise NotImplementedError(f"Unknown line_opac_scheme='{line_opac_scheme}'")

    ray_opac_scheme = getattr(phys, "opac_ray", "None")
    if ray_opac_scheme == "None":
        print(f"[info] Rayleigh opacity is None:", ray_opac_scheme)
        ray_opac_kernel = zero_ray_opacity
    elif ray_opac_scheme == "lbl" or ray_opac_scheme == "ck":
        ray_opac_kernel = compute_ray_opacity
    else:
        raise NotImplementedError(f"Unknown ray_opac_scheme='{ray_opac_scheme}'")

    cia_opac_scheme = getattr(phys, "opac_cia", "None")
    if cia_opac_scheme == "None":
        print(f"[info] CIA opacity is None:", cia_opac_scheme)
        cia_opac_kernel = zero_cia_opacity
    elif cia_opac_scheme == "lbl" or cia_opac_scheme == "ck":
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

        wl = wl_hi

        # Dimension constants
        nwl = jnp.size(wl)

        # Planet and star radii (R0 is radius at p_bot)
        R0 = jnp.asarray(params["R_p"]) * R_jup
        R_s = jnp.asarray(params["R_s"]) * R_sun

        # Atmospheric pressure grid
        p_bot = jnp.asarray(params["p_bot"]) * bar
        p_top = jnp.asarray(params["p_top"]) * bar
        p_lev = jnp.logspace(jnp.log10(p_bot), jnp.log10(p_top), nlev)
        
        # Vertical atmospheric T-p layer structure
        p_lay = (p_lev[1:] - p_lev[:-1]) / jnp.log(p_lev[1:]/p_lev[:-1])
        T_lay = vert_kernel(p_lev, params)

        # Get the VMR structure of the atmosphere
        vmr = {}
        for k, v in params.items():
            if k.startswith("log_10_f_"):
                sp = k[len("log_10_f_"):]
                vmr[sp] = 10.0 ** v          # store species key, e.g. "H2O"
            elif k.startswith("f_"):
                sp = k[len("f_"):]
                vmr[sp] = v

        # Calculate the mixing ratios of H2 and He
        # Sum all trace species VMRs
        total_trace_vmr = jnp.sum(jnp.array([v for v in vmr.values()]))
        background_vmr = 1.0 - total_trace_vmr

        vmr['H2'] = background_vmr * solar_h2 / solar_h2_he
        vmr['He'] = background_vmr * solar_he / solar_h2_he

        # Cast scalar VMR to per-layer VMR
        vmr_lay = {species: jnp.full((nlay,), value) for species, value in vmr.items()}

        # Mean molecular weight calculation
        if "mu" in params:
            mu_const = jnp.asarray(params["mu"])
            mu_lay = jnp.full((nlay,), mu_const)
        else:
            mu_lay, mu_dynamic = compute_mean_molecular_weight(vmr_lay)
            if mu_lay is None or (not mu_dynamic):
                raise ValueError("Dynamic mean molecular weight failed; provide 'mu' parameter or fix vert_mu.")

        # Vertical altitude calculation
        z_lev = hypsometric_variable_g(p_lev, T_lay, mu_lay, params)
        z_lay = (z_lev[:-1] + z_lev[1:]) / 2.0
        dz = jnp.diff(z_lev)

        # Atmospheric density and number density
        rho_lay = (mu_lay * amu * p_lay) / (kb * T_lay)
        nd_lay = p_lay / (kb * T_lay)

        # State dictionary for physics kernels
        g_weights = None
        if ck and XS.has_ck_data():
            g_weights = XS.ck_g_weights()
            if g_weights.ndim > 1:
                g_weights = g_weights[0]
            g_weights = jnp.asarray(g_weights)

        state = {
            'ck': ck,
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
            "rho_lay": rho_lay,
            "nd_lay": nd_lay,
            "vmr_lay": vmr_lay,
        }
        if g_weights is not None:
            state["g_weights"] = g_weights

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
