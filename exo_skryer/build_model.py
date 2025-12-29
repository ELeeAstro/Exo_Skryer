"""
build_model.py
==============
"""

from __future__ import annotations
from typing import Dict, Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from .data_constants import kb, amu, R_jup, R_sun, bar, G, M_jup

from .vert_alt import hypsometric, hypsometric_variable_g, hypsometric_variable_g_pref
from .vert_Tp import isothermal, Milne, Guillot, Barstow, MandS09, picket_fence, Milne_modified
from .vert_chem import constant_vmr, CE_fastchem_jax, CE_rate_jax
from .vert_mu import constant_mu, compute_mu

from .opacity_line import zero_line_opacity, compute_line_opacity
from .opacity_ck import zero_ck_opacity, compute_ck_opacity
from .opacity_ray import zero_ray_opacity, compute_ray_opacity
from .opacity_cia import zero_cia_opacity, compute_cia_opacity
from .opacity_special import zero_special_opacity, compute_special_opacity
from .opacity_cloud import zero_cloud_opacity, grey_cloud, powerlaw_cloud, F18_cloud, F18_cloud_2, direct_nk, direct_nk_slab

from . import build_opacities as XS
from .build_chem import prepare_chemistry_kernel

from .RT_trans_1D_ck import compute_transit_depth_1d_ck
from .RT_trans_1D_lbl import compute_transit_depth_1d_lbl
from .RT_em_1D_ck import compute_emission_spectrum_1d_ck
from .RT_em_1D_lbl import compute_emission_spectrum_1d_lbl
from .RT_em_schemes import get_emission_solver

from .instru_convolve import apply_response_functions

__all__ = [
    'build_forward_model'
]


def build_forward_model(
    cfg,
    obs: Dict,
    stellar_flux: Optional[np.ndarray] = None,
    return_highres: bool = False,
) -> Callable[[Dict[str, jnp.ndarray]], Union[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    """Build a JIT-compiled forward model for atmospheric retrieval.

    This function constructs a forward model by assembling physics kernels for
    vertical structure (temperature, chemistry, altitude), opacity sources
    (line, continuum, clouds), and radiative transfer. The returned function
    is JIT-compiled for efficient gradient-based inference.

    Parameters
    ----------
    cfg : config object
        Configuration object containing physics settings (`cfg.physics`),
        opacity configuration (`cfg.opac`), and retrieval parameters (`cfg.params`).
        Must specify schemes for vertical structure (vert_Tp, vert_alt, vert_chem,
        vert_mu), opacity sources (opac_line, opac_ray, opac_cia, opac_cloud,
        opac_special), and radiative transfer (rt_scheme).
    obs : dict
        Observational data dictionary containing:
        - 'wl' : Observed wavelengths in microns (for bandpass loading)
        - 'dwl' : Wavelength bin widths in microns
    stellar_flux : `~numpy.ndarray`, optional
        Stellar flux array for emission spectroscopy calculations. Required when
        rt_scheme is 'emission_1d' and emission_mode is 'planet' (not brown dwarf).
        Should match the high-resolution wavelength grid.
    return_highres : bool, optional
        If True, the forward model returns both high-resolution and binned spectra
        as a dictionary: `{'hires': D_hires, 'binned': D_bin}`. If False (default),
        returns only the binned spectrum as a 1D array.

    Returns
    -------
    forward_model : callable
        A JIT-compiled function with signature:
        `forward_model(params: Dict[str, jnp.ndarray]) -> Union[jnp.ndarray, Dict]`

        The function takes a parameter dictionary (free parameters from the retrieval)
        and returns:
        - If `return_highres=False`: 1D array of binned transit depth or emission flux
        - If `return_highres=True`: Dict with keys 'hires' (high-res spectrum) and
          'binned' (convolved spectrum)

    Notes
    -----
    The forward model pipeline consists of:

    1. **Vertical Structure**: Computes pressure-temperature profile (vert_Tp),
       altitude grid (vert_alt), chemical abundances (vert_chem), and mean
       molecular weight (vert_mu) for each atmospheric layer.

    2. **Opacity Calculation**: Computes wavelength-dependent opacity from line
       absorption (opac_line), Rayleigh scattering (opac_ray), collision-induced
       absorption (opac_cia), clouds (opac_cloud), and special opacity sources
       like H- bound-free (opac_special).

    3. **Radiative Transfer**: Solves the radiative transfer equation using the
       specified rt_scheme (transit_1d or emission_1d) to produce a high-resolution
       spectrum.

    4. **Instrumental Response**: Applies wavelength-dependent response functions
       to produce the final binned spectrum matching observational resolution.

    Configuration schemes are selected from `cfg.physics`:
    - **vert_Tp**: isothermal, guillot, barstow, milne, picket_fence, ms09, piecewise_polynomial
    - **vert_alt**: constant_g, variable_g, p_ref
    - **vert_chem**: constant_vmr, ce (FastChem placeholder), ce_rate_jax
    - **vert_mu**: auto, constant, dynamic
    - **opac_line**: none, lbl, ck
    - **opac_ray, opac_cia**: none, lbl, ck
    - **opac_cloud**: none, grey, powerlaw_cloud, f18, f18_2, nk
    - **opac_special**: on (default), off
    - **rt_scheme**: transit_1d, emission_1d

    For constant VMR chemistry, required trace species are automatically inferred
    from the opacity configuration and validated against `cfg.params`.
    """

    # Extract fixed (delta) parameters from cfg.params
    fixed_params = {}
    for param in cfg.params:
        if param.dist == "delta":
            fixed_params[param.name] = float(param.value)

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

    vert_tp_raw = getattr(phys, "vert_Tp", None)
    if vert_tp_raw in (None, "None"):
        vert_tp_raw = getattr(phys, "vert_struct", None)
    if vert_tp_raw in (None, "None"):
        raise ValueError("physics.vert_Tp (or vert_struct) must be specified explicitly.")
    vert_tp_name = str(vert_tp_raw).lower()
    if vert_tp_name in ("isothermal", "constant"):
        Tp_kernel = isothermal
    elif vert_tp_name == "barstow":
        Tp_kernel = Barstow
    elif vert_tp_name == "milne":
        Tp_kernel = Milne
    elif vert_tp_name == "guillot":
        Tp_kernel = Guillot
    elif vert_tp_name == "picket_fence":
        Tp_kernel = picket_fence
    elif vert_tp_name == "ms09":
        Tp_kernel = MandS09
    elif vert_tp_name in ("milne_2", "milne_modified"):
        Tp_kernel = Milne_modified
    else:
        raise NotImplementedError(f"Unknown vert_Tp='{vert_tp_name}'")

    vert_alt_raw = getattr(phys, "vert_alt", None)
    if vert_alt_raw in (None, "None"):
        raise ValueError("physics.vert_alt must be specified explicitly.")
    vert_alt_name = str(vert_alt_raw).lower()
    if vert_alt_name in ("constant", "constant_g", "fixed", "hypsometric"):
        altitude_kernel = hypsometric
    elif vert_alt_name in ("variable", "variable_g", "hypsometric_variable_g"):
        altitude_kernel = hypsometric_variable_g
    elif vert_alt_name in ("p_ref", "hypsometric_variable_g_pref"):
        altitude_kernel = hypsometric_variable_g_pref
    else:
        raise NotImplementedError(f"Unknown altitude scheme='{vert_alt_name}'")

    vert_chem_raw = getattr(phys, "vert_chem", None)
    if vert_chem_raw in (None, "None"):
        raise ValueError("physics.vert_chem must be specified explicitly.")
    vert_chem_name = str(vert_chem_raw).lower()
    if vert_chem_name in ("constant", "constant_vmr"):
        chemistry_kernel = constant_vmr
    elif vert_chem_name in ("ce", "chemical_equilibrium", "ce_fastchem_jax", "fastchem_jax"):
        chemistry_kernel = CE_fastchem_jax
    elif vert_chem_name in ("rate_ce", "rate_jax", "ce_rate_jax"):
        chemistry_kernel = CE_rate_jax
    else:
        raise NotImplementedError(f"Unknown chemistry scheme='{vert_chem_name}'")

    vert_mu_raw = getattr(phys, "vert_mu", None)
    if vert_mu_raw in (None, "None"):
        raise ValueError("physics.vert_mu must be specified explicitly.")
    vert_mu_name = str(vert_mu_raw).lower()
    if vert_mu_name == "auto":
        def mu_kernel(params, vmr_lay, nlay):
            if "mu" in params:
                return constant_mu(params, nlay)
            return compute_mu(vmr_lay)
    elif vert_mu_name in ("constant", "fixed"):
        def mu_kernel(params, vmr_lay, nlay):
            del vmr_lay
            return constant_mu(params, nlay)
    elif vert_mu_name in ("dynamic", "variable", "vmr"):
        def mu_kernel(params, vmr_lay, nlay):
            del params, nlay
            return compute_mu(vmr_lay)
    else:
        raise NotImplementedError(f"Unknown mean-molecular-weight scheme='{vert_mu_name}'")

    ck = False
    line_opac_scheme = getattr(phys, "opac_line", None)
    if line_opac_scheme is None:
        raise ValueError("physics.opac_line must be specified explicitly (use 'None' to disable).")
    line_opac_scheme_str = str(line_opac_scheme)
    if line_opac_scheme_str.lower() == "none":
        print(f"[info] Line opacity is None:", line_opac_scheme)
        line_opac_kernel = None
    elif line_opac_scheme_str.lower() == "lbl":
        if not XS.has_line_data():
            raise RuntimeError(
                "Line opacity requested but registry is empty. "
                "Check cfg.opac.line and ensure build_opacities() loaded line tables."
            )
        line_opac_kernel = compute_line_opacity
    elif line_opac_scheme_str.lower() == "ck":
        ck = True
        line_opac_kernel = compute_ck_opacity
    else:
        raise NotImplementedError(f"Unknown line_opac_scheme='{line_opac_scheme}'")

    ray_opac_scheme = getattr(phys, "opac_ray", None)
    if ray_opac_scheme is None:
        raise ValueError("physics.opac_ray must be specified explicitly (use 'None' to disable).")
    ray_opac_scheme_str = str(ray_opac_scheme)
    if ray_opac_scheme_str.lower() == "none":
        print(f"[info] Rayleigh opacity is None:", ray_opac_scheme)
        ray_opac_kernel = None
    elif ray_opac_scheme_str.lower() in ("lbl", "ck"):
        ray_opac_kernel = compute_ray_opacity
    else:
        raise NotImplementedError(f"Unknown ray_opac_scheme='{ray_opac_scheme}'")

    cia_opac_scheme = getattr(phys, "opac_cia", None)
    if cia_opac_scheme is None:
        raise ValueError("physics.opac_cia must be specified explicitly (use 'None' to disable).")
    cia_opac_scheme_str = str(cia_opac_scheme)
    if cia_opac_scheme_str.lower() == "none":
        print(f"[info] CIA opacity is None:", cia_opac_scheme)
        cia_opac_kernel = None
    elif cia_opac_scheme_str.lower() in ("lbl", "ck"):
        cia_opac_kernel = compute_cia_opacity
    else:
        raise NotImplementedError(f"Unknown cia_opac_scheme='{cia_opac_scheme}'")

    cld_opac_scheme = getattr(phys, "opac_cloud", None)
    if cld_opac_scheme is None:
        raise ValueError("physics.opac_cloud must be specified explicitly (use 'None' to disable).")
    cld_opac_scheme_str = str(cld_opac_scheme)
    if cld_opac_scheme_str.lower() == "none":
        print(f"[info] Cloud opacity is None:", cld_opac_scheme)
        cld_opac_kernel = None
    elif cld_opac_scheme_str.lower() == "grey":
        cld_opac_kernel = grey_cloud
    elif cld_opac_scheme_str.lower() == "powerlaw_cloud":
        cld_opac_kernel = powerlaw_cloud
    elif cld_opac_scheme_str.lower() == "f18":
        cld_opac_kernel = F18_cloud
    elif cld_opac_scheme_str.lower() == "f18_2":
        cld_opac_kernel = F18_cloud_2
    elif cld_opac_scheme_str.lower() == "nk":
        cld_opac_kernel = direct_nk
    elif cld_opac_scheme_str.lower() == "nk_slab":
        cld_opac_kernel = direct_nk_slab        
    else:
        raise NotImplementedError(f"Unknown cld_opac_scheme='{cld_opac_scheme}'")

    special_opac_scheme = getattr(phys, "opac_special", "on")
    special_opac_scheme_str = str(special_opac_scheme).lower()
    if special_opac_scheme_str in ("none", "off", "false", "0"):
        special_opac_kernel = None
    elif special_opac_scheme_str in ("on", "lbl", "ck"):
        special_opac_kernel = compute_special_opacity
    else:
        raise NotImplementedError(f"Unknown opac_special='{special_opac_scheme}'")

    rt_raw = getattr(phys, "rt_scheme", None)
    if rt_raw in (None, "None"):
        raise ValueError("physics.rt_scheme must be specified explicitly.")
    rt_scheme = str(rt_raw).lower()
    if rt_scheme == "transit_1d":
        rt_kernel = compute_transit_depth_1d_ck if ck else compute_transit_depth_1d_lbl
    elif rt_scheme == "emission_1d":
        em_scheme = getattr(phys, "em_scheme", "eaa")
        emission_solver = get_emission_solver(em_scheme)
        if ck:
            rt_kernel = lambda state, params, components: compute_emission_spectrum_1d_ck(
                state, params, components, emission_solver=emission_solver
            )
        else:
            rt_kernel = lambda state, params, components: compute_emission_spectrum_1d_lbl(
                state, params, components, emission_solver=emission_solver
            )
    else:
        raise NotImplementedError(f"Unknown rt_scheme='{rt_scheme}'")

    # High-resolution master grid (must match cut_grid used in bandpass loader)
    wl_hi_array = np.asarray(XS.master_wavelength_cut(), dtype=float)
    wl_hi = jnp.asarray(wl_hi_array)
    stellar_flux_arr = None
    if stellar_flux is not None:
        stellar_flux_arr = jnp.asarray(stellar_flux, dtype=jnp.float64)

    emission_mode = getattr(phys, "emission_mode", "planet")
    if emission_mode is None:
        emission_mode = "planet"
    emission_mode = str(emission_mode).lower().replace(" ", "_")
    is_brown_dwarf = emission_mode in ("brown_dwarf", "browndwarf", "bd")

    chemistry_kernel, trace_species = prepare_chemistry_kernel(
        cfg,
        chemistry_kernel,
        {
            'line_opac': line_opac_scheme_str,
            'ray_opac': ray_opac_scheme_str,
            'cia_opac': cia_opac_scheme_str,
            'special_opac': special_opac_scheme_str,
        }
    )

    @jax.jit
    def forward_model(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:

        # Merge fixed (delta) parameters with varying parameters
        full_params = {**fixed_params, **params}

        wl = wl_hi

        # Dimension constants
        nwl = wl.shape[0]

        # Planet and star radii (R0 is radius at p_bot)
        R0 = jnp.asarray(full_params["R_p"]) * R_jup
        R_s = jnp.asarray(full_params["R_s"]) * R_sun

        # Atmospheric pressure grid
        p_bot = jnp.asarray(full_params["p_bot"]) * bar
        p_top = jnp.asarray(full_params["p_top"]) * bar
        p_lev = jnp.logspace(jnp.log10(p_bot), jnp.log10(p_top), nlev)

        # Vertical atmospheric T-p layer structure
        p_lay = (p_lev[1:] - p_lev[:-1]) / jnp.log(p_lev[1:]/p_lev[:-1])
        T_lev, T_lay = Tp_kernel(p_lev, full_params)

        # Get the vertical chemical structure (VMRs at each layer)
        vmr_lay = chemistry_kernel(p_lay, T_lay, full_params, nlay)

        # Mean molecular weight calculation
        mu_lay = mu_kernel(full_params, vmr_lay, nlay)

        # Vertical altitude calculation
        z_lev, z_lay, dz = altitude_kernel(p_lev, T_lay, mu_lay, full_params)

        # Atmospheric density and number density
        rho_lay = (mu_lay * amu * p_lay) / (kb * T_lay)
        nd_lay = p_lay / (kb * T_lay)

        # State dictionary for physics kernels
        g_weights = None
        ck_mix_code = None
        if ck and XS.has_ck_data():
            g_weights = XS.ck_g_weights()
            if g_weights.ndim > 1:
                g_weights = g_weights[0]
            g_weights = jnp.asarray(g_weights)

            # Get ck_mix option from config
            ck_mix_raw = getattr(cfg.opac, "ck_mix", "RORR")
            ck_mix_raw = str(ck_mix_raw).upper()
            if ck_mix_raw == "PRAS":
                ck_mix_code = 2
            else:
                ck_mix_code = 1

        state = {
            "nwl": nwl,
            "nlay": nlay,
            "wl": wl,
            "is_brown_dwarf": is_brown_dwarf,
            "R0": R0,
            "R_s": R_s,
            "p_lev": p_lev,
            "T_lev": T_lev,
            "z_lev": z_lev,
            "z_lay": z_lay,
            "dz": dz,
            "mu_lay": mu_lay,
            "T_lay": T_lay,
            "p_lay": p_lay,
            "rho_lay": rho_lay,
            "nd_lay": nd_lay,
            "vmr_lay": vmr_lay,
            "contri_func": bool(getattr(phys, "contri_func", False)),
        }
        if stellar_flux_arr is not None:
            state["stellar_flux"] = stellar_flux_arr
        if g_weights is not None:
            state["g_weights"] = g_weights
        if ck_mix_code is not None:
            state["ck_mix"] = ck_mix_code

        opacity_components = {}
        if line_opac_kernel is not None:
            opacity_components["line"] = line_opac_kernel(state, full_params)
        if ray_opac_kernel is not None:
            opacity_components["rayleigh"] = ray_opac_kernel(state, full_params)
        if cia_opac_kernel is not None:
            opacity_components["cia"] = cia_opac_kernel(state, full_params)
        if special_opac_kernel is not None:
            opacity_components["special"] = special_opac_kernel(state, full_params)
        if cld_opac_kernel is not None:
            k_cld_ext, cld_ssa, cld_g = cld_opac_kernel(state, full_params)
            opacity_components["cloud"] = k_cld_ext
            opacity_components["cloud_ssa"] = cld_ssa
            opacity_components["cloud_g"] = cld_g

        # Radiative transfer
        # RT kernels always return (spectrum, contrib_func)
        # contrib_func is zeros if state["contri_func"] is False
        D_hires, contrib_func = rt_kernel(state, full_params, opacity_components)

        # Instrumental convolution → binned spectrum
        D_bin = apply_response_functions(D_hires)

        if return_highres:
            result_dict = {"hires": D_hires, "binned": D_bin}
            if state["contri_func"]:
                result_dict["contrib_func"] = contrib_func
                result_dict["p_lay"] = p_lay
            return result_dict

        return D_bin

    return forward_model
