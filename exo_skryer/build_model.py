"""
build_model.py
==============
"""

from __future__ import annotations
from types import SimpleNamespace
from typing import Dict, Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from .data_constants import kb, amu, R_jup, R_sun, bar, G, M_jup

from .opacity_line import zero_line_opacity, compute_line_opacity
from .opacity_ck import zero_ck_opacity, compute_ck_opacity, compute_ck_opacity_perspecies
from .opacity_ray import zero_ray_opacity, compute_ray_opacity
from .opacity_cia import zero_cia_opacity, compute_cia_opacity
from .opacity_special import zero_special_opacity, compute_special_opacity
from .opacity_cloud import zero_cloud_opacity

from . import build_opacities as XS
from .build_chem import prepare_chemistry_kernel

from .RT_trans_1D_ck import compute_transit_depth_1d_ck
from .RT_trans_1D_ck_trans import compute_transit_depth_1d_ck_trans
from .RT_trans_1D_lbl import compute_transit_depth_1d_lbl
from .RT_em_1D_ck import compute_emission_spectrum_1d_ck
from .RT_em_1D_lbl import compute_emission_spectrum_1d_lbl
from .RT_em_schemes import get_emission_solver

from .instru_convolve import apply_response_functions_cached, get_bandpass_cache

from . import kernel_registry as KR

__all__ = [
    'build_forward_model'
]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_fixed_params(cfg) -> Dict[str, jnp.ndarray]:
    """Extract delta-distribution parameters and static cloud_dist into a fixed-params dict."""
    fixed_params: Dict[str, jnp.ndarray] = {}
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

    cloud_dist_raw = getattr(cfg.physics, "cloud_dist", None)
    if cloud_dist_raw is not None:
        cloud_dist_str = str(cloud_dist_raw).lower().strip()
        if cloud_dist_str in ("1", "mono", "monodisperse"):
            fixed_params["cloud_dist"] = jnp.asarray(1, dtype=jnp.int32)
        elif cloud_dist_str in ("2", "log_normal", "lognormal", "log-normal", "ln"):
            fixed_params["cloud_dist"] = jnp.asarray(2, dtype=jnp.int32)
        else:
            raise ValueError("physics.cloud_dist must be 'mono' or 'log_normal' (or 1/2).")

    return fixed_params


def _resolve_lbl_ck_opac(phys, key: str, fn: Callable):
    """Resolve a simple none/lbl/ck opacity setting.

    Returns ``(scheme_str, kernel_or_None)``.  The scheme string is lowercased.
    """
    raw = getattr(phys, key, None)
    if raw is None:
        raise ValueError(
            f"physics.{key} must be specified explicitly (use 'None' to disable)."
        )
    s = str(raw).lower()
    if s == "none":
        print(f"[info] {key} is None:", raw)
        return s, None
    if s in ("lbl", "ck"):
        return s, fn
    raise NotImplementedError(
        f"Unknown physics.{key}='{raw}'. Options: none | lbl | ck"
    )


def _resolve_refraction(phys) -> int:
    """Return the refraction mode integer (0 = off, 1 = cutoff)."""
    refraction_raw = getattr(phys, "refraction", None)
    if refraction_raw is None:
        return 0
    s = str(refraction_raw).strip().lower()
    if s in ("none", "off", "false", "0"):
        return 0
    if s in ("cutoff", "refractive_cutoff", "refraction_cutoff"):
        return 1
    raise NotImplementedError(f"Unknown physics.refraction='{refraction_raw}'")


def _build_rt_kernel(phys, rt_scheme: str, ck: bool, ck_mix_str: str) -> Callable:
    """Select and return the radiative-transfer kernel for the given scheme."""
    if rt_scheme == "transit_1d":
        if ck:
            return compute_transit_depth_1d_ck_trans if ck_mix_str == "TRANS" else compute_transit_depth_1d_ck
        return compute_transit_depth_1d_lbl
    if rt_scheme == "emission_1d":
        em_scheme = getattr(phys, "em_scheme", "eaa")
        emission_solver = get_emission_solver(em_scheme)
        if ck:
            return lambda state, params, components, opac: compute_emission_spectrum_1d_ck(
                state, params, components, opac, emission_solver=emission_solver
            )
        return lambda state, params, components: compute_emission_spectrum_1d_lbl(
            state, params, components, emission_solver=emission_solver
        )
    raise NotImplementedError(
        f"Unknown physics.rt_scheme='{rt_scheme}'. Options: transit_1d | emission_1d"
    )


def _select_kernels(cfg) -> SimpleNamespace:
    """Select all physics and opacity kernels from the YAML config.

    Returns a :class:`~types.SimpleNamespace` with the following fields:

    Kernels
        ``Tp_kernel``, ``altitude_kernel``, ``chemistry_kernel``,
        ``mu_kernel``, ``vert_cloud_kernel``, ``line_opac_kernel``,
        ``ray_opac_kernel``, ``cia_opac_kernel``, ``cld_opac_kernel``,
        ``special_opac_kernel``, ``rt_kernel``

    Metadata used by validation / forward model
        ``ck``, ``rt_scheme``, ``ck_mix_str``, ``ck_mix_code_static``,
        ``contri_func_enabled``, ``refraction_mode``,
        ``line_opac_str``, ``ray_opac_str``, ``cia_opac_str``,
        ``cld_opac_str``, ``special_opac_str``
    """
    phys = cfg.physics

    # --- vertical structure ---
    vert_tp_raw = getattr(phys, "vert_Tp", None) or getattr(phys, "vert_struct", None)
    Tp_kernel = KR.resolve(vert_tp_raw, KR.VERT_TP, "physics.vert_Tp")

    altitude_kernel = KR.resolve(
        getattr(phys, "vert_alt", None), KR.VERT_ALT, "physics.vert_alt"
    )

    chemistry_kernel = KR.resolve(
        getattr(phys, "vert_chem", None), KR.VERT_CHEM, "physics.vert_chem"
    )

    mu_kernel = KR.resolve(
        getattr(phys, "vert_mu", None), KR.VERT_MU, "physics.vert_mu"
    )

    # vert_cloud defaults to "none" if absent from YAML
    vert_cloud_raw = getattr(phys, "vert_cloud", "none") or "none"
    vert_cloud_kernel = KR.resolve(vert_cloud_raw, KR.VERT_CLOUD, "physics.vert_cloud")

    # --- line opacity (also determines ck mode) ---
    line_opac_raw = getattr(phys, "opac_line", None)
    if line_opac_raw is None:
        raise ValueError(
            "physics.opac_line must be specified explicitly (use 'None' to disable)."
        )
    line_opac_str = str(line_opac_raw).lower()
    ck = (line_opac_str == "ck")
    if line_opac_str == "none":
        print(f"[info] Line opacity is None:", line_opac_raw)
        line_opac_kernel = None
    elif line_opac_str == "lbl":
        line_opac_kernel = compute_line_opacity
    elif line_opac_str == "ck":
        line_opac_kernel = compute_ck_opacity
    else:
        raise NotImplementedError(
            f"Unknown physics.opac_line='{line_opac_raw}'. Options: none | lbl | ck"
        )

    # --- continuum opacities ---
    ray_opac_str, ray_opac_kernel = _resolve_lbl_ck_opac(phys, "opac_ray", compute_ray_opacity)
    cia_opac_str, cia_opac_kernel = _resolve_lbl_ck_opac(phys, "opac_cia", compute_cia_opacity)

    # --- cloud opacity ---
    cld_opac_raw = getattr(phys, "opac_cloud", None)
    if cld_opac_raw is None:
        raise ValueError(
            "physics.opac_cloud must be specified explicitly (use 'None' to disable)."
        )
    cld_opac_str = str(cld_opac_raw).lower()
    if cld_opac_str == "none":
        print(f"[info] Cloud opacity is None:", cld_opac_raw)
    cld_opac_kernel = KR.resolve(cld_opac_raw, KR.OPAC_CLOUD, "physics.opac_cloud")

    # --- special opacity (H-) ---
    special_opac_str = str(getattr(phys, "opac_special", "on")).lower()
    special_opac_kernel = (
        None if special_opac_str in ("none", "off", "false", "0")
        else compute_special_opacity
    )

    # --- RT scheme ---
    rt_raw = getattr(phys, "rt_scheme", None)
    if rt_raw is None or str(rt_raw).lower() == "none":
        raise ValueError("physics.rt_scheme must be specified explicitly.")
    rt_scheme = str(rt_raw).lower()

    refraction_mode = _resolve_refraction(phys)

    ck_mix_str = str(getattr(cfg.opac, "ck_mix", "RORR")).upper()
    contri_func_enabled = bool(getattr(phys, "contri_func", False))

    ck_mix_code_static = None
    if ck:
        if ck_mix_str == "PRAS":
            ck_mix_code_static = 2
        elif ck_mix_str == "TRANS":
            ck_mix_code_static = 3
        else:
            ck_mix_code_static = 1

    rt_kernel = _build_rt_kernel(phys, rt_scheme, ck, ck_mix_str)

    return SimpleNamespace(
        Tp_kernel=Tp_kernel,
        altitude_kernel=altitude_kernel,
        chemistry_kernel=chemistry_kernel,
        mu_kernel=mu_kernel,
        vert_cloud_kernel=vert_cloud_kernel,
        line_opac_kernel=line_opac_kernel,
        ray_opac_kernel=ray_opac_kernel,
        cia_opac_kernel=cia_opac_kernel,
        cld_opac_kernel=cld_opac_kernel,
        special_opac_kernel=special_opac_kernel,
        rt_kernel=rt_kernel,
        ck=ck,
        rt_scheme=rt_scheme,
        ck_mix_str=ck_mix_str,
        ck_mix_code_static=ck_mix_code_static,
        contri_func_enabled=contri_func_enabled,
        refraction_mode=refraction_mode,
        line_opac_str=line_opac_str,
        ray_opac_str=ray_opac_str,
        cia_opac_str=cia_opac_str,
        cld_opac_str=cld_opac_str,
        special_opac_str=special_opac_str,
    )


def _build_opac_cache() -> Dict[str, jnp.ndarray]:
    """Assemble the runtime opacity cache dict from all loaded registries."""
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
    return opac_cache


def _require_cache_keys(opac_cache: dict, keys: tuple, label: str) -> None:
    missing = [k for k in keys if k not in opac_cache]
    if missing:
        raise RuntimeError(f"Missing {label} cache entries: {missing}")


def _validate_config(
    cfg,
    k: SimpleNamespace,
    opac_cache: Dict[str, jnp.ndarray],
) -> None:
    """Validate consistency of config, kernel selection, and loaded opacity data.

    Raises ``ValueError`` or ``RuntimeError`` on any inconsistency so that
    problems surface at build time rather than silently producing wrong results.
    """
    # Line opacity data present in registries
    if k.line_opac_str == "lbl" and not XS.has_line_data():
        raise RuntimeError(
            "Line opacity requested but registry is empty. "
            "Check cfg.opac.line and ensure build_opacities() loaded line tables."
        )
    if k.ck and not XS.has_ck_data():
        raise RuntimeError(
            "CK opacity requested but registry is empty. "
            "Check cfg.opac and ensure build_opacities() loaded ck tables."
        )

    # Refraction constraints
    if k.refraction_mode == 1:
        if k.rt_scheme != "transit_1d":
            raise NotImplementedError(
                "physics.refraction is only supported for rt_scheme: transit_1d."
            )
        param_names = {p.name for p in cfg.params}
        if "a_sm" not in param_names:
            raise ValueError(
                "physics.refraction: cutoff requires a delta parameter "
                "'a_sm' (semi-major axis in AU)."
            )
        if not XS.has_ray_data():
            raise RuntimeError(
                "physics.refraction: cutoff requires Rayleigh registry data (cfg.opac.ray)."
            )

    # CK-mix constraints
    if k.ck:
        if k.ck_mix_str == "TRANS" and k.rt_scheme != "transit_1d":
            raise NotImplementedError(
                "ck_mix: TRANS is only supported for rt_scheme: transit_1d."
            )
        if k.ck_mix_str == "TRANS" and k.contri_func_enabled:
            raise ValueError(
                "physics.contri_func=True is not supported with opac.ck_mix=TRANS. "
                "Use ck_mix=RORR/PRAS or disable contribution functions."
            )

    # Required opacity cache entries
    if k.ck and k.line_opac_str == "ck":
        _require_cache_keys(
            opac_cache,
            ("ck_sigma_cube", "ck_log10_pressure_grid", "ck_log10_temperature_grids",
             "ck_g_points", "ck_g_weights"),
            "correlated-k",
        )
    if (not k.ck) and k.line_opac_str == "lbl":
        _require_cache_keys(
            opac_cache,
            ("line_sigma_cube", "line_log10_pressure_grid", "line_log10_temperature_grids"),
            "line-by-line",
        )
    if k.cia_opac_kernel is not None:
        _require_cache_keys(
            opac_cache,
            ("cia_master_wavelength", "cia_sigma_cube",
             "cia_log10_temperature_grids", "cia_temperature_grids"),
            "CIA",
        )
    if k.ray_opac_kernel is not None:
        _require_cache_keys(opac_cache, ("ray_master_wavelength", "ray_sigma_table"), "Rayleigh")
    if k.special_opac_kernel is not None and XS.has_special_data():
        _require_cache_keys(
            opac_cache,
            ("hminus_master_wavelength", "hminus_temperature_grid",
             "hminus_log10_temperature_grid"),
            "special-opacity",
        )

    # Mie cloud schemes require cached n,k
    if k.cld_opac_str in ("madt_rayleigh", "madt-rayleigh", "mie_madt",
                           "lxmie", "mie_full", "full_mie"):
        _require_cache_keys(opac_cache, ("cloud_nk_n", "cloud_nk_k"), "cloud n,k")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    fixed_params = _extract_fixed_params(cfg)

    nlay = int(getattr(cfg.physics, "nlay", 99))
    nlev = nlay + 1

    # Observational wavelengths/widths are consumed by response-function caches,
    # not directly by this function body.
    _ = np.asarray(obs["wl"], dtype=float)
    _ = np.asarray(obs["dwl"], dtype=float)

    # Select all physics and opacity kernels
    k = _select_kernels(cfg)

    # Assemble runtime opacity cache from loaded registries
    opac_cache = _build_opac_cache()

    # Validate consistency of the full configuration
    _validate_config(cfg, k, opac_cache)

    emission_mode = str(getattr(cfg.physics, "emission_mode", "planet")).lower().replace(" ", "_")
    if emission_mode is None:
        emission_mode = "planet"
    is_brown_dwarf = emission_mode in ("brown_dwarf", "browndwarf", "bd")

    # For planet emission, require stellar normalization information up front.
    # This avoids silent fallbacks inside JIT code paths.
    if k.rt_scheme == "emission_1d" and (not is_brown_dwarf) and (stellar_flux is None):
        param_names = {p.name for p in cfg.params}
        if "F_star" not in param_names:
            raise ValueError(
                "Planet emission mode requires either stellar_flux input or parameter 'F_star'."
            )

    # High-resolution master grid (must match cut_grid used in bandpass loader)
    wl_hi_array = np.asarray(XS.master_wavelength_cut(), dtype=float)
    wl_hi = jnp.asarray(wl_hi_array)
    has_stellar_flux_arr = jnp.asarray(1 if stellar_flux is not None else 0, dtype=jnp.int32)
    if stellar_flux is not None:
        stellar_flux_arr = jnp.asarray(stellar_flux, dtype=jnp.float64)
    else:
        stellar_flux_arr = jnp.zeros_like(wl_hi, dtype=jnp.float64)

    bandpass_cache = get_bandpass_cache()

    chemistry_kernel, trace_species = prepare_chemistry_kernel(
        cfg,
        k.chemistry_kernel,
        {
            'line_opac': k.line_opac_str,
            'ray_opac': k.ray_opac_str,
            'cia_opac': k.cia_opac_str,
            'special_opac': k.special_opac_str,
        }
    )

    # Capture kernel selections into local names for the JIT closure
    Tp_kernel = k.Tp_kernel
    altitude_kernel = k.altitude_kernel
    mu_kernel = k.mu_kernel
    vert_cloud_kernel = k.vert_cloud_kernel
    line_opac_kernel = k.line_opac_kernel
    ray_opac_kernel = k.ray_opac_kernel
    cia_opac_kernel = k.cia_opac_kernel
    cld_opac_kernel = k.cld_opac_kernel
    special_opac_kernel = k.special_opac_kernel
    rt_kernel = k.rt_kernel
    ck = k.ck
    rt_scheme = k.rt_scheme
    ck_mix_code_static = k.ck_mix_code_static
    contri_func_enabled = k.contri_func_enabled
    refraction_mode = k.refraction_mode

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
