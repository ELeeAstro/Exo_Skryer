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
from .vert_Tp import (
    isothermal,
    Milne,
    Guillot,
    Modified_Guillot,
    Line,
    Barstow,
    MandS,
    picket_fence,
    Milne_modified,
)
from .vert_chem import constant_vmr, constant_vmr_clr, CE_fastchem_jax, CE_rate_jax, quench_approx
from .vert_mu import constant_mu, compute_mu
from .vert_cloud import no_cloud, exponential_decay_profile, slab_profile, const_profile

from .opacity_line import zero_line_opacity, compute_line_opacity
from .opacity_ck import zero_ck_opacity, compute_ck_opacity, compute_ck_opacity_perspecies
from .opacity_ray import zero_ray_opacity, compute_ray_opacity
from .opacity_cia import zero_cia_opacity, compute_cia_opacity
from .opacity_special import zero_special_opacity, compute_special_opacity
from .opacity_cloud import compute_cloud_opacity, zero_cloud_opacity, grey_cloud, deck_and_powerlaw, direct_nk

from . import build_opacities as XS
from .build_chem import prepare_chemistry_kernel

from .RT_trans_1D_ck import compute_transit_depth_1d_ck
from .RT_trans_1D_ck_trans import compute_transit_depth_1d_ck_trans
from .RT_trans_1D_lbl import compute_transit_depth_1d_lbl
from .RT_em_1D_ck import compute_emission_spectrum_1d_ck
from .RT_em_1D_lbl import compute_emission_spectrum_1d_lbl
from .RT_em_schemes import get_emission_solver

from .instru_convolve import apply_response_functions_cached, get_bandpass_cache

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
    """

    # Extract fixed (delta) parameters from cfg.params
    fixed_params = {}
    for param in cfg.params:
        if param.dist == "delta":
            raw_value = getattr(param, "value", None)
            if isinstance(raw_value, str):
                raw_lower = raw_value.strip().lower()
                if raw_lower in ("true", "false"):
                    raw_value = (raw_lower == "true")
                else:
                    raw_value = float(raw_value)
            fixed_params[param.name] = jnp.asarray(raw_value)

    # Cloud size distribution selector from YAML (static, not sampled).
    # 1 = monodisperse, 2 = polydisperse (lognormal)
    cloud_dist_raw = getattr(cfg.physics, "cloud_dist", None)
    if cloud_dist_raw is not None:
        cloud_dist_str = str(cloud_dist_raw).lower().strip()
        if cloud_dist_str in ("1", "mono", "monodisperse"):
            fixed_params["cloud_dist"] = jnp.asarray(1, dtype=jnp.int32)
        elif cloud_dist_str in ("2", "log_normal", "lognormal", "log-normal", "ln"):
            fixed_params["cloud_dist"] = jnp.asarray(2, dtype=jnp.int32)
        else:
            raise ValueError("physics.cloud_dist must be 'mono' or 'log_normal' (or 1/2).")

    # Example: number of layers from YAML
    nlay = int(getattr(cfg.physics, "nlay", 99))
    nlev = nlay + 1

    # Observational wavelengths/widths are consumed by response-function caches,
    # not directly by this function body.
    _ = np.asarray(obs["wl"], dtype=float)
    _ = np.asarray(obs["dwl"], dtype=float)

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
    elif vert_tp_name in ("modified_guillot", "guillot_modified", "guillot_2"):
        Tp_kernel = Modified_Guillot
    elif vert_tp_name == "line":
        Tp_kernel = Line     
    elif vert_tp_name == "picket_fence":
        Tp_kernel = picket_fence
    elif vert_tp_name == "mands":
        Tp_kernel = MandS
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
    elif vert_chem_name in ("constant_vmr_clr", "constant_clr", "clr"):
        chemistry_kernel = constant_vmr_clr
    elif vert_chem_name in ("ce", "chemical_equilibrium", "ce_fastchem_jax", "fastchem_jax"):
        chemistry_kernel = CE_fastchem_jax
    elif vert_chem_name in ("rate_ce", "rate_jax", "ce_rate_jax"):
        chemistry_kernel = CE_rate_jax
    elif vert_chem_name in ("quench", "quench_approx"):
        chemistry_kernel = quench_approx
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

    vert_cloud_raw = getattr(phys, "vert_cloud", None)
    if vert_cloud_raw in (None, "None"):
        vert_cloud_name = "none"
    else:
        vert_cloud_name = str(vert_cloud_raw).lower()

    if vert_cloud_name in ("none", "off", "no_cloud"):
        vert_cloud_kernel = no_cloud
    elif vert_cloud_name in ("exponential", "exp_decay", "exponential_decay", "exponential_decay_profile"):
        vert_cloud_kernel = exponential_decay_profile
    elif vert_cloud_name in ("slab", "slab_profile"):
        vert_cloud_kernel = slab_profile
    elif vert_cloud_name in ("const", "constant", "const_profile"):
        vert_cloud_kernel = const_profile
    else:
        raise NotImplementedError(f"Unknown vert_cloud scheme='{vert_cloud_name}'")

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
    elif cld_opac_scheme_str.lower() in ("powerlaw", "deck_and_powerlaw"):
        cld_opac_kernel = deck_and_powerlaw
    elif cld_opac_scheme_str.lower() == "f18":
        cld_opac_kernel = lambda state, params: compute_cloud_opacity(state, params, opacity_scheme="f18")
    elif cld_opac_scheme_str.lower() in ("madt_rayleigh", "madt-rayleigh", "mie_madt"):
        cld_opac_kernel = lambda state, params: compute_cloud_opacity(state, params, opacity_scheme="madt_rayleigh")
    elif cld_opac_scheme_str.lower() in ("lxmie", "mie_full", "full_mie"):
        cld_opac_kernel = lambda state, params: compute_cloud_opacity(state, params, opacity_scheme="lxmie")
    elif cld_opac_scheme_str.lower() in ("nk", "direct_nk"):
        cld_opac_kernel = direct_nk
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

    # Refraction (transmission only): apply a refractive cutoff without ray tracing.
    refraction_raw = getattr(phys, "refraction", None)
    if refraction_raw in (None, "None"):
        refraction_mode = 0
    else:
        refraction_str = str(refraction_raw).strip().lower()
        if refraction_str in ("none", "off", "false", "0"):
            refraction_mode = 0
        elif refraction_str in ("cutoff", "refractive_cutoff", "refraction_cutoff"):
            refraction_mode = 1
        else:
            raise NotImplementedError(f"Unknown physics.refraction='{refraction_raw}'")

    if refraction_mode == 1:
        if rt_scheme != "transit_1d":
            raise NotImplementedError("physics.refraction is only supported for rt_scheme: transit_1d.")
        param_names = {p.name for p in cfg.params}
        if "a_sm" not in param_names:
            raise ValueError("physics.refraction: cutoff requires a delta parameter 'a_sm' (semi-major axis in AU).")
        if not XS.has_ray_data():
            raise RuntimeError("physics.refraction: cutoff requires Rayleigh registry data (cfg.opac.ray).")

    # Get ck_mix for RT kernel selection (TRANS uses different RT kernel)
    ck_mix_str = str(getattr(cfg.opac, "ck_mix", "RORR")).upper()
    contri_func_enabled = bool(getattr(phys, "contri_func", False))
    if ck:
        if ck_mix_str == "TRANS" and rt_scheme != "transit_1d":
            raise NotImplementedError("ck_mix: TRANS is only supported for rt_scheme: transit_1d.")
        if ck_mix_str == "TRANS" and contri_func_enabled:
            raise ValueError(
                "physics.contri_func=True is not supported with opac.ck_mix=TRANS. "
                "Use ck_mix=RORR/PRAS or disable contribution functions."
            )

    if rt_scheme == "transit_1d":
        if ck:
            if ck_mix_str == "TRANS":
                rt_kernel = compute_transit_depth_1d_ck_trans
            else:
                rt_kernel = compute_transit_depth_1d_ck
        else:
            rt_kernel = compute_transit_depth_1d_lbl
    elif rt_scheme == "emission_1d":
        em_scheme = getattr(phys, "em_scheme", "eaa")
        emission_solver = get_emission_solver(em_scheme)
        if ck:
            rt_kernel = lambda state, params, components, opac: compute_emission_spectrum_1d_ck(
                state, params, components, opac, emission_solver=emission_solver
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
    has_stellar_flux_arr = jnp.asarray(1 if stellar_flux is not None else 0, dtype=jnp.int32)
    if stellar_flux is not None:
        stellar_flux_arr = jnp.asarray(stellar_flux, dtype=jnp.float64)
    else:
        stellar_flux_arr = jnp.zeros_like(wl_hi, dtype=jnp.float64)

    bandpass_cache = get_bandpass_cache()

    opac_cache: Dict[str, jnp.ndarray] = {}
    if XS.has_ck_data():
        opac_cache["ck_sigma_cube"] = XS.ck_sigma_cube()
        opac_cache["ck_log10_pressure_grid"] = XS.ck_log10_pressure_grid()
        opac_cache["ck_log10_temperature_grids"] = XS.ck_log10_temperature_grids()
        opac_cache["ck_g_points"] = XS.ck_g_points()
        opac_cache["ck_g_weights"] = XS.ck_g_weights()
    if XS.has_line_data():
        opac_cache["line_sigma_cube"] = XS.line_sigma_cube()
        opac_cache["line_log10_pressure_grid"] = XS.line_log10_pressure_grid()
        opac_cache["line_log10_temperature_grids"] = XS.line_log10_temperature_grids()
    if XS.has_cia_data():
        opac_cache["cia_master_wavelength"] = XS.cia_master_wavelength()
        opac_cache["cia_sigma_cube"] = XS.cia_sigma_cube()
        opac_cache["cia_log10_temperature_grids"] = XS.cia_log10_temperature_grids()
        opac_cache["cia_temperature_grids"] = XS.cia_temperature_grids()
    if XS.has_ray_data():
        opac_cache["ray_master_wavelength"] = XS.ray_master_wavelength()
        opac_cache["ray_sigma_table"] = XS.ray_sigma_table()
        opac_cache["ray_nm1_table"] = XS.ray_nm1_table()
        opac_cache["ray_nd_ref"] = XS.ray_nd_ref()
    if XS.has_cloud_nk_data():
        opac_cache["cloud_nk_n"] = XS.cloud_nk_n()
        opac_cache["cloud_nk_k"] = XS.cloud_nk_k()
    if XS.has_special_data():
        opac_cache["hminus_master_wavelength"] = XS.special_master_wavelength()
        opac_cache["hminus_temperature_grid"] = XS.hminus_temperature_grid()
        opac_cache["hminus_log10_temperature_grid"] = XS.hminus_log10_temperature_grid()
        # Table presence depends on cfg.opac.special flags; only include those that exist.
        try:
            opac_cache["hminus_bf_log10_sigma"] = XS.hminus_bf_log10_sigma_table()
        except RuntimeError:
            pass
        try:
            opac_cache["hminus_ff_log10_sigma"] = XS.hminus_ff_log10_sigma_table()
        except RuntimeError:
            pass

    # Opacities must be present in the runtime cache 
    if ck and line_opac_scheme_str.lower() == "ck":
        required = ("ck_sigma_cube", "ck_log10_pressure_grid", "ck_log10_temperature_grids", "ck_g_points", "ck_g_weights")
        missing = [k for k in required if k not in opac_cache]
        if missing:
            raise RuntimeError(f"Missing correlated-k cache entries: {missing}")
    if (not ck) and line_opac_scheme_str.lower() == "lbl":
        required = ("line_sigma_cube", "line_log10_pressure_grid", "line_log10_temperature_grids")
        missing = [k for k in required if k not in opac_cache]
        if missing:
            raise RuntimeError(f"Missing line-by-line cache entries: {missing}")
    if cia_opac_kernel is not None:
        required = ("cia_master_wavelength", "cia_sigma_cube", "cia_log10_temperature_grids", "cia_temperature_grids")
        missing = [k for k in required if k not in opac_cache]
        if missing:
            raise RuntimeError(f"Missing CIA cache entries: {missing}")
    if ray_opac_kernel is not None:
        required = ("ray_master_wavelength", "ray_sigma_table")
        missing = [k for k in required if k not in opac_cache]
        if missing:
            raise RuntimeError(f"Missing Rayleigh cache entries: {missing}")
    if special_opac_kernel is not None and XS.has_special_data():
        required = ("hminus_master_wavelength", "hminus_temperature_grid", "hminus_log10_temperature_grid")
        missing = [k for k in required if k not in opac_cache]
        if missing:
            raise RuntimeError(f"Missing special-opacity cache entries: {missing}")

    # Mie cloud schemes require cached n,k.
    if cld_opac_scheme_str.lower() in ("madt_rayleigh", "madt-rayleigh", "mie_madt", "lxmie", "mie_full", "full_mie"):
        required = ("cloud_nk_n", "cloud_nk_k")
        missing = [k for k in required if k not in opac_cache]
        if missing:
            raise RuntimeError(f"Missing cloud n,k cache entries: {missing}")

    emission_mode = getattr(phys, "emission_mode", "planet")
    if emission_mode is None:
        emission_mode = "planet"
    emission_mode = str(emission_mode).lower().replace(" ", "_")
    is_brown_dwarf = emission_mode in ("brown_dwarf", "browndwarf", "bd")

    # For planet emission, require stellar normalization information up front.
    # This avoids silent fallbacks inside JIT code paths.
    if rt_scheme == "emission_1d" and (not is_brown_dwarf) and (stellar_flux is None):
        param_names = {p.name for p in cfg.params}
        if "F_star" not in param_names:
            raise ValueError(
                "Planet emission mode requires either stellar_flux input or parameter 'F_star'."
            )

    # Resolve ck_mix once (static config), not inside the JIT function.
    ck_mix_code_static = None
    if ck:
        if ck_mix_str == "PRAS":
            ck_mix_code_static = 2
        elif ck_mix_str == "TRANS":
            ck_mix_code_static = 3
        else:
            ck_mix_code_static = 1

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
    def _forward_model_impl(
        params: Dict[str, jnp.ndarray],
        wl_runtime: jnp.ndarray,
        stellar_flux_runtime: jnp.ndarray,
        has_stellar_flux_runtime: jnp.ndarray,
        opac_cache_runtime: Dict[str, jnp.ndarray],
        bandpass_cache_runtime: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:

        # Merge fixed (delta) parameters with varying parameters
        full_params = {**fixed_params, **params}

        wl = wl_runtime

        # Dimension constants
        nwl = wl.shape[0]

        # Planet and star radii (R0 is radius at p_bot)
        R0 = full_params["R_p"] * R_jup
        R_s = full_params["R_s"] * R_sun

        # Calculate log_10_g from mass and radius if M_p is provided
        if "M_p" in full_params:
            M_p = full_params["M_p"] * M_jup  # Convert to g
            R_p = full_params["R_p"] * R_jup  # Convert to cm
            g = G * M_p / (R_p ** 2)  # Surface gravity in cm/s^2
            full_params["log_10_g"] = jnp.log10(g)

        # Atmospheric pressure grid
        p_bot = full_params["p_bot"] * bar
        p_top = full_params["p_top"] * bar
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

        # Cloud vertical profile (mass mixing ratio)
        q_c_lay = vert_cloud_kernel(p_lay, T_lay, mu_lay, rho_lay, nd_lay, full_params)

        # Opacity cache for kernels (separate from atmospheric state)
        opac = opac_cache_runtime
        if ck:
            if "ck_g_weights" not in opac:
                raise RuntimeError("Missing opac['ck_g_weights'] for c-k mode.")
            if "ck_g_points" not in opac:
                raise RuntimeError("Missing opac['ck_g_points'] for c-k mode.")
            g_weights = opac["ck_g_weights"]
            if g_weights.ndim > 1:
                g_weights = g_weights[0]
            g_points = opac["ck_g_points"]
            if g_points.ndim > 1:
                g_points = g_points[0]
            opac = dict(opac)
            opac["g_weights"] = g_weights
            opac["g_points"] = g_points

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
            "q_c_lay": q_c_lay,
            "vmr_lay": vmr_lay,
            "contri_func": contri_func_enabled,
            "refraction_mode": refraction_mode,
        }
        if "cloud_nk_n" in opac:
            state["cloud_nk_n"] = opac["cloud_nk_n"]
            state["cloud_nk_k"] = opac["cloud_nk_k"]
        state["stellar_flux"] = stellar_flux_runtime
        state["has_stellar_flux"] = has_stellar_flux_runtime
        if ck_mix_code_static is not None:
            state["ck_mix"] = ck_mix_code_static

        if ck:
            line_zero = zero_ck_opacity(state, opac, full_params)
        else:
            line_zero = zero_line_opacity(state, full_params)
        k_cld_zero, cld_ssa_zero, cld_g_zero = zero_cloud_opacity(state, full_params)
        opacity_components = {
            "line": line_zero,
            "rayleigh": zero_ray_opacity(state, full_params),
            "cia": zero_cia_opacity(state, full_params),
            "special": zero_special_opacity(state, full_params),
            "cloud": k_cld_zero,
            "cloud_ssa": cld_ssa_zero,
            "cloud_g": cld_g_zero,
        }
        if line_opac_kernel is not None:
            # For TRANS method, compute per-species opacities (mixing happens in RT)
            if ck and ck_mix_code_static == 3:  # TRANS
                sigma_ps, vmr_ps = compute_ck_opacity_perspecies(state, opac, full_params)
                opacity_components["line_perspecies"] = sigma_ps
                opacity_components["vmr_perspecies"] = vmr_ps
            else:
                opacity_components["line"] = line_opac_kernel(state, opac, full_params)
        if ray_opac_kernel is not None:
            opacity_components["rayleigh"] = ray_opac_kernel(state, opac, full_params)
        if cia_opac_kernel is not None:
            opacity_components["cia"] = cia_opac_kernel(state, opac, full_params)
        if special_opac_kernel is not None:
            opacity_components["special"] = special_opac_kernel(state, opac, full_params)
        if cld_opac_kernel is not None:
            k_cld_ext, cld_ssa, cld_g = cld_opac_kernel(state, full_params)
            opacity_components["cloud"] = k_cld_ext
            opacity_components["cloud_ssa"] = cld_ssa
            opacity_components["cloud_g"] = cld_g

        # Radiative transfer
        # RT kernels always return (spectrum, contrib_func)
        # contrib_func is zeros if state["contri_func"] is False
        if rt_scheme == "transit_1d":
            # All transit RT kernels accept the opac cache (LBL may use it for refraction).
            D_hires, contrib_func = rt_kernel(state, full_params, opacity_components, opac)
        else:
            if ck:
                D_hires, contrib_func = rt_kernel(state, full_params, opacity_components, opac)
            else:
                D_hires, contrib_func = rt_kernel(state, full_params, opacity_components)

        # Instrumental convolution → binned spectrum
        D_bin = apply_response_functions_cached(D_hires, bandpass_cache_runtime)

        if return_highres:
            result_dict = {"hires": D_hires, "binned": D_bin}
            if state["contri_func"]:
                result_dict["contrib_func"] = contrib_func
                result_dict["p_lay"] = p_lay
            return result_dict

        return D_bin

    def forward_model(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        return _forward_model_impl(
            params,
            wl_hi,
            stellar_flux_arr,
            has_stellar_flux_arr,
            opac_cache,
            bandpass_cache,
        )

    return forward_model
