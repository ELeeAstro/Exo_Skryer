"""
opacity_cloud.py
================
"""

from typing import Dict, Tuple, Optional
import jax
import jax.numpy as jnp

from .aux_functions import pchip_1d
from .registry_cloud import get_or_create_kk_cache, KKGridCache, get_cloud_nk_data
from .mie_schemes import rayleigh, madt

__all__ = [
    "compute_cloud_opacity",
    "zero_cloud_opacity",
    "grey_cloud",
    "deck_and_powerlaw",
    "F18_cloud",
    "direct_nk",
    "given_nk",
]


def _compute_mie_madt_efficiencies(
    wl_val: jnp.ndarray,
    n_val: jnp.ndarray,
    k_val: jnp.ndarray,
    r_eff: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute extinction and scattering efficiencies using Rayleigh + MADT blend.

    This function computes Q_ext, Q_sca, and g using a smooth blend between
    Rayleigh scattering (small particles) and Modified Anomalous Diffraction
    Theory (MADT, large particles) based on the size parameter x.

    Parameters
    ----------
    wl_val : `~jax.numpy.ndarray`
        Wavelength in microns.
    n_val : `~jax.numpy.ndarray`
        Real part of the refractive index.
    k_val : `~jax.numpy.ndarray`
        Imaginary part of the refractive index.
    r_eff : `~jax.numpy.ndarray`
        Effective particle radius in microns.

    Returns
    -------
    Q_ext : `~jax.numpy.ndarray`
        Extinction efficiency.
    Q_sca : `~jax.numpy.ndarray`
        Scattering efficiency.
    g : `~jax.numpy.ndarray`
        Asymmetry parameter.
    """
    # Compute size parameter
    x = 2.0 * jnp.pi * r_eff / jnp.maximum(wl_val, 1e-12)

    # Compute Rayleigh and MADT efficiencies using modular functions
    Q_ext_ray, Q_sca_ray, g_ray = rayleigh(n_val, k_val, x)
    Q_ext_madt, Q_sca_madt, g_madt = madt(n_val, k_val, x)

    # Smooth blend between Rayleigh (x=1.0) and MADT (x=3.0) using smootherstep
    t = jnp.clip((x - 1.0) / 2.0, 0.0, 1.0)  # Maps x=1→0, x=3→1
    w = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


    Q_ext = (1.0 - w) * Q_ext_ray + w * Q_ext_madt
    Q_sca = (1.0 - w) * Q_sca_ray + w * Q_sca_madt
    g = (1.0 - w) * g_ray + w * g_madt

    return Q_ext, Q_sca, g


def _compute_mie_or_zero(
    wl_val: jnp.ndarray,
    n_val: jnp.ndarray,
    k_val: jnp.ndarray,
    r_eff: jnp.ndarray,
    is_in_support: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Conditionally compute Mie efficiencies or return zeros.

    This wrapper uses lax.cond to skip expensive Mie calculations for wavelengths
    outside the node support range, improving performance by avoiding unnecessary
    computation.

    Parameters
    ----------
    wl_val : `~jax.numpy.ndarray`
        Wavelength in microns.
    n_val : `~jax.numpy.ndarray`
        Real part of the refractive index.
    k_val : `~jax.numpy.ndarray`
        Imaginary part of the refractive index.
    r_eff : `~jax.numpy.ndarray`
        Effective particle radius in microns.
    is_in_support : `~jax.numpy.ndarray`
        Boolean indicating if wavelength is within node support range.

    Returns
    -------
    Q_ext : `~jax.numpy.ndarray`
        Extinction efficiency (0.0 if outside support).
    Q_sca : `~jax.numpy.ndarray`
        Scattering efficiency (0.0 if outside support).
    g : `~jax.numpy.ndarray`
        Asymmetry parameter (0.0 if outside support).
    """
    def compute():
        return _compute_mie_madt_efficiencies(wl_val, n_val, k_val, r_eff)

    def skip():
        return (jnp.zeros_like(wl_val), jnp.zeros_like(wl_val), jnp.zeros_like(wl_val))

    return jax.lax.cond(is_in_support, compute, skip)


def compute_cloud_opacity(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_scheme: str = "none",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Main landing function for cloud opacity calculation.

    This function dispatches to specific cloud opacity schemes based on the
    opacity_scheme parameter. It expects the vertical cloud profile (q_c_lay)
    to already be present in the state dictionary (computed by vert_cloud kernels).

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `p_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer pressures in dyne cm⁻².
        - `q_c_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Cloud mass mixing ratio per layer (from vert_cloud kernel).
        - `nlay` : int
            Number of atmospheric layers.
        - `nwl` : int
            Number of wavelength points.

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing scheme-specific parameters.
        Required parameters depend on the chosen opacity_scheme.

    opacity_scheme : str, optional
        Cloud opacity scheme identifier. Options:

        - `"none"` or `"zero"`: No cloud opacity (default)
        - `"grey"`: Wavelength-independent grey opacity
        - `"direct_nk"`: Retrieved refractive index with Mie/MADT scattering
        - `"given_nk"`: Cached global refractive index with Mie/MADT scattering
        - `"F18"`: Fisher & Heng (2018) empirical model
        - `"powerlaw"`: Grey + power-law wavelength dependence

    Returns
    -------
    k_cld : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Cloud extinction coefficient in cm² g⁻¹.
    ssa : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Single-scattering albedo (Q_sca / Q_ext).
    g : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Asymmetry parameter for scattering phase function.
    """
    scheme_lower = opacity_scheme.lower().strip()

    # Dispatch to appropriate scheme
    if scheme_lower in ("none", "zero", "off", "no_cloud"):
        return zero_cloud_opacity(state, params)
    elif scheme_lower in ("grey", "gray"):
        return grey_cloud(state, params)
    elif scheme_lower in ("direct_nk", "nk", "mie"):
        return direct_nk(state, params)
    elif scheme_lower in ("given_nk", "cached_nk", "global_nk"):
        return given_nk(state, params)
    elif scheme_lower in ("f18", "fisher18", "fisher_heng"):
        return F18_cloud(state, params)
    elif scheme_lower in ("powerlaw", "power_law", "deck_and_powerlaw"):
        return deck_and_powerlaw(state, params)
    else:
        raise ValueError(
            f"Unknown cloud opacity scheme: '{opacity_scheme}'. "
            f"Valid options: none, grey, direct_nk, given_nk, F18, powerlaw"
        )


def zero_cloud_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return zero-valued cloud optical properties.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        State dictionary containing:

        - `nlay` : int
            Number of atmospheric layers.
        - `nwl` : int
            Number of wavelength points.

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; kept for API compatibility).

    Returns
    -------
    k_cld : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Cloud extinction coefficient in cm² g⁻¹ (all zeros).
    ssa : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Single-scattering albedo (all zeros).
    g : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Asymmetry parameter (all zeros).
    """
    del params
    # Use shape directly without int() conversion for JIT compatibility
    shape = (state["nlay"], state["nwl"])
    k_cld = jnp.zeros(shape)
    ssa = jnp.zeros(shape)
    g = jnp.zeros(shape)
    return k_cld, ssa, g


def grey_cloud(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute a grey (wavelength-independent) cloud opacity floor.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        State dictionary containing scalar entries `nlay` and `nwl`.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_k_cld_grey` : float
            Log₁₀ of the grey cloud extinction coefficient in cm² g⁻¹.

    Returns
    -------
    k_cld : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Grey cloud extinction coefficient in cm² g⁻¹.
    ssa : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Single-scattering albedo (zeros; pure absorption).
    g : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Asymmetry parameter (zeros).
    """
    # Use shape directly without int() conversion for JIT compatibility
    shape = (state["nlay"], state["nwl"])
    opacity_value = 10.0**params["log_10_k_cld_grey"]
    k_cld = jnp.full(shape, opacity_value)
    ssa = jnp.zeros(shape)
    g = jnp.zeros(shape)
    return k_cld, ssa, g


def deck_and_powerlaw(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    wl = state["wl"]
    nlay = state["nlay"]

    # Constant grey opacity component
    k_grey = 10.0**params["log_10_k_cld_grey"]

    # Power-law amplitude at reference wavelength
    k_powerlaw = 10.0**params["log_10_k_cld_Ray"]

    # Power-law exponent (alpha=4 gives Rayleigh slope)
    alpha = params["alpha_cld"]

    # Reference wavelength
    wl_ref = params["wl_ref_cld"]

    # Two-component opacity: grey + power-law
    # k(λ) = k_grey + k_powerlaw * (λ/λ_ref)^(-alpha)
    k_wl = k_grey + k_powerlaw * (wl / wl_ref)**(-alpha)

    # Broadcast to (nlay, nwl) using implicit broadcasting
    k_cld = jnp.zeros((nlay, 1)) + k_wl[None, :]

    # Pure absorption (no scattering)
    ssa = jnp.zeros_like(k_cld)
    g = jnp.zeros_like(k_cld)

    return k_cld, ssa, g

def F18_cloud(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute a cloud opacity using Fisher and Heng (2018) with vertical profile.

    This function evaluates an extinction efficiency `Qext(x)` as a function of
    size parameter `x = 2πr/λ` and converts it into a wavelength-dependent cloud
    extinction coefficient. The vertical profile is provided via q_c_lay in the
    state dictionary (computed separately by vert_cloud kernels).

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        State dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `q_c_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Cloud mass mixing ratio per layer (from vert_cloud kernel).
        - `nlay` : int
            Number of atmospheric layers.

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_cld_r` : float
            Log₁₀ geometric-mean particle radius `r_g` in microns.
        - `cld_Q0` : float
            Extinction-efficiency scale factor.
        - `cld_a` : float
            Power-law exponent controlling the small-particle regime.
        - `cld_sigma` : float
            Lognormal geometric standard deviation (dimensionless).
        - `cld_rho` : float
            Cloud bulk density in g cm⁻³.
        - `cld_Q1` : float
            Extinction-efficiency scale factor.
            
    Returns
    -------
    k_cld : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Cloud extinction coefficient in cm² g⁻¹.
    ssa : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Single-scattering albedo (zeros; pure absorption).
    g : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Asymmetry parameter (zeros).
    """
    wl = state["wl"]
    q_c_lay = state["q_c_lay"]  # (nlay,)

    r_g = 10.0**params["log_10_cld_r"]
    Q0 = params["cld_Q0"]
    a = params["cld_a"]
    cld_rho = params["cld_rho"]  # Cloud bulk density (g/cm^3)
    Q1 = params["cld_Q1"]
    sig = params["cld_sigma"]

    # Compute size parameter and extinction efficiency
    lnsig2 = jnp.log(sig) ** 2
    r_eff = r_g * jnp.exp(2.5 * lnsig2)

    x = (2.0 * jnp.pi * r_eff) / wl # (nwl,)
    x = jnp.maximum(x, 1e-30)
    Qext = Q1 / (Q0 * x ** (-a) + x**0.2)  # (nwl,)


    # Compute cloud opacity using vertical profile
    # q_c_lay: (nlay,), Qext: (nwl,) -> k_cld: (nlay, nwl)
    k_cld = (3.0 * q_c_lay[:, None] * Qext[None, :]) / (4.0 * cld_rho * (r_eff * 1e-4)) 

    ssa = jnp.zeros_like(k_cld)
    g = jnp.zeros_like(k_cld)
    return k_cld, ssa, g

def direct_nk(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute cloud optical properties from retrieved refractive-index nodes.

    This function retrieves node values describing the complex refractive
    index (n, k) as a function of wavelength, interpolates them onto the model
    wavelength grid, and computes wavelength-dependent optical properties using
    Mie/MADT scattering. The vertical profile is provided via q_c_lay in the
    state dictionary (computed separately by vert_cloud kernels).

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `q_c_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Cloud mass mixing ratio per layer (from vert_cloud kernel).

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `wl_node_0`..`wl_node_12` : float
            Wavelength nodes (microns).
        - `n_0`..`n_12` : float
            Real refractive-index nodes.
        - `log_10_k_0`..`log_10_k_12` : float
            Log₁₀ imaginary refractive-index nodes.
        - `log_10_cld_r` : float
            Log₁₀ particle radius in microns.
        - `cld_rho` : float
            Cloud bulk density in g cm⁻³.

    Returns
    -------
    k_cld : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Cloud extinction coefficient in cm² g⁻¹.
    ssa : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Single-scattering albedo derived from (Q_sca / Q_ext).
    g : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Asymmetry parameter (zeros in this implementation).
    """
    wl = state["wl"]          # (nwl,) in micron
    q_c_lay = state["q_c_lay"]  # (nlay,)

    # -----------------------------
    # Retrieved / configured knobs
    # -----------------------------
    r_eff = 10.0 ** params["log_10_cld_r"]  # particle radius (um)
    cld_rho = params["cld_rho"]  # Cloud bulk density, defaults to 1.0 g/cm³

    # Keep n positive for scattering math sanity (doesn't forbid n<1)
    n_floor = 1e-6

    # -----------------------------
    # Retrieve k(wl) from log-nodes
    # -----------------------------
    # Use jnp.stack instead of list comprehension for efficiency
    wl_nodes = jnp.stack([params[f"wl_node_{i}"] for i in range(13)])
    # Limit nk contribution to the wavelength span covered by the nodes
    wl_support_min = jnp.min(wl_nodes)
    wl_support_max = jnp.max(wl_nodes)
    wl_support_mask = jnp.logical_and(wl >= wl_support_min, wl <= wl_support_max)

    # Retrieve n(wl) / k(wl) node values using jnp.stack
    n_nodes = jnp.stack([params[f"n_{i}"] for i in range(13)])
    log10_k_nodes = jnp.stack([params[f"log_10_k_{i}"] for i in range(13)])

    n_interp = pchip_1d(wl, wl_nodes, n_nodes)
    log10_k_interp = pchip_1d(wl, wl_nodes, log10_k_nodes)
    n = jnp.maximum(n_interp, n_floor)
    k = jnp.maximum(10.0 ** log10_k_interp, 1e-12)
    n = jnp.where(wl_support_mask, n, n_floor)
    k = jnp.where(wl_support_mask, k, 1e-12)

    # -----------------------------
    # Compute optical properties
    # -----------------------------

    # Compute Mie/MADT efficiencies conditionally (skip wavelengths outside node support)
    Q_ext_vals, Q_sca_vals, g_vals = jax.vmap(_compute_mie_or_zero, in_axes=(0, 0, 0, None, 0))(
        wl, n, k, r_eff, wl_support_mask
    )

    # Compute cloud opacity using vertical profile from state
    # q_c_lay: (nlay,), Q_ext_vals: (nwl,) -> k_cld: (nlay, nwl)
    k_cld = (
        (3.0 * q_c_lay[:, None] * Q_ext_vals[None, :])
        / (4.0 * cld_rho * (r_eff * 1e-4))
    )

    ssa_wl = jnp.clip(Q_sca_vals / jnp.maximum(Q_ext_vals, 1e-30), 0.0, 1.0)
    # Use implicit broadcasting instead of broadcast_to
    ssa = ssa_wl[None, :] + jnp.zeros_like(q_c_lay[:, None])
    # Broadcast g values to (nlay, nwl)
    g = g_vals[None, :] + jnp.zeros_like(q_c_lay[:, None])

    return k_cld, ssa, g


def given_nk(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute cloud optical properties from cached global n,k arrays.

    This function retrieves pre-loaded refractive index data from the global
    registry and computes wavelength-dependent optical properties using
    Mie/MADT scattering. The n,k data must be loaded into the registry using
    `set_cloud_nk_data()` before calling this function. The vertical profile
    is provided via q_c_lay in the state dictionary.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `q_c_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Cloud mass mixing ratio per layer (from vert_cloud kernel).

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_cld_r` : float
            Log₁₀ particle radius in microns.
        - `cld_rho` : float
            Cloud bulk density in g cm⁻³.

    Returns
    -------
    k_cld : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Cloud extinction coefficient in cm² g⁻¹.
    ssa : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Single-scattering albedo derived from (Q_sca / Q_ext).
    g : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Asymmetry parameter.

    Raises
    ------
    RuntimeError
        If no cloud n,k data has been loaded (call `set_cloud_nk_data` first).
    """
    wl = state["wl"]          # (nwl,) in micron
    q_c_lay = state["q_c_lay"]  # (nlay,)

    # -----------------------------
    # Retrieve cached n,k data
    # -----------------------------
    nk_data = get_cloud_nk_data()
    n = nk_data['n']  # Already on model wavelength grid
    k = nk_data['k']  # Already on model wavelength grid

    # -----------------------------
    # Retrieved / configured knobs
    # -----------------------------
    r_eff = 10.0 ** params["log_10_cld_r"]  # particle radius (um)
    cld_rho = params["cld_rho"]  # Cloud bulk density, defaults to 1.0 g/cm³

    # -----------------------------
    # Compute optical properties
    # -----------------------------

    # Compute Mie/MADT efficiencies using modular blending function
    Q_ext_vals, Q_sca_vals, g_vals = jax.vmap(_compute_mie_madt_efficiencies, in_axes=(0, 0, 0, None))(
        wl, n, k, r_eff
    )

    # Compute cloud opacity using vertical profile from state
    # q_c_lay: (nlay,), Q_ext_vals: (nwl,) -> k_cld: (nlay, nwl)
    k_cld = (
        (3.0 * q_c_lay[:, None] * Q_ext_vals[None, :])
        / (4.0 * cld_rho * (r_eff * 1e-4))
    )

    ssa_wl = jnp.clip(Q_sca_vals / jnp.maximum(Q_ext_vals, 1e-30), 0.0, 1.0)
    # Use implicit broadcasting instead of broadcast_to
    ssa = ssa_wl[None, :] + jnp.zeros_like(q_c_lay[:, None])
    # Broadcast g values to (nlay, nwl)
    g = g_vals[None, :] + jnp.zeros_like(q_c_lay[:, None])

    return k_cld, ssa, g
