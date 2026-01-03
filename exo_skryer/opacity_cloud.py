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
    "powerlaw_cloud",
    "F18_cloud",
    "F18_cloud_2",
    "direct_nk",
    "direct_nk_slab",
    "given_nk",
    "_compute_mie_madt_efficiencies",
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
            Layer pressures in microbar.
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

    Notes
    -----
    The q_c_lay vertical profile should be computed separately using functions
    from the vert_cloud module (e.g., no_cloud, exponential_decay_profile,
    slab_profile) and added to the state dictionary before calling this function.

    Examples
    --------
    >>> # After computing vertical profile
    >>> q_c_lay = exponential_decay_profile(p_lay, T_lay, mu_lay, rho_lay, nd_lay, params)
    >>> state["q_c_lay"] = q_c_lay
    >>> # Now compute cloud opacity
    >>> k_cld, ssa, g = compute_cloud_opacity(state, params, opacity_scheme="direct_nk")
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


# Alias for backwards compatibility
powerlaw_cloud = deck_and_powerlaw


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
            Log₁₀ particle radius in microns.
        - `cld_Q0` : float
            Extinction-efficiency scale factor.
        - `cld_a` : float
            Power-law exponent controlling the small-particle regime.
        - `cld_rho` : float, optional
            Cloud bulk density in g cm⁻³ (defaults to 1.0).
        - `Q1` : float, optional
            Q1 scale factor (defaults to 1.0).
            
    Returns
    -------
    k_cld : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Cloud extinction coefficient in cm² g⁻¹.
    ssa : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Single-scattering albedo (zeros; pure absorption).
    g : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Asymmetry parameter (zeros).

    Notes
    -----
    The F18 parameterization for extinction efficiency is:
        Qext(x) = Q1 / (Q0 * x^(-a) + x^0.2)

    The cloud mass mixing ratio vertical profile (q_c_lay) should be computed
    separately using vert_cloud functions and added to the state dictionary
    before calling this function.
    """
    wl = state["wl"]
    q_c_lay = state["q_c_lay"]  # (nlay,)

    r = 10.0**params["log_10_cld_r"]
    Q0 = params["cld_Q0"]
    a = params["cld_a"]
    cld_rho = params.get("cld_rho", 1.0)  # Cloud bulk density, defaults to 1.0 g/cm³
    Q1 = params.get("cld_Q1", 1.0)  # Q1 parameter, defaults to 1.0 

    # Compute size parameter and extinction efficiency
    x = (2.0 * jnp.pi * r) / wl  # (nwl,)
    Qext = Q1 / (Q0 * x**-a + x**0.2)  # (nwl,)

    # Compute cloud opacity using vertical profile
    # q_c_lay: (nlay,), Qext: (nwl,) -> k_cld: (nlay, nwl)
    k_cld = (3.0 * q_c_lay[:, None] * Qext[None, :]) / (4.0 * cld_rho * (r * 1e-4))

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
        - `sigma` : float
            Log-normal width parameter (clipped to be ≥ 1).
        - `cld_rho` : float, optional
            Cloud bulk density in g cm⁻³ (defaults to 1.0).

    Returns
    -------
    k_cld : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Cloud extinction coefficient in cm² g⁻¹.
    ssa : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Single-scattering albedo derived from (Q_sca / Q_ext).
    g : `~jax.numpy.ndarray`, shape (nlay, nwl)
        Asymmetry parameter (zeros in this implementation).

    Notes
    -----
    The retrieved node curves are interpolated using `aux_functions.pchip_1d`.
    To avoid extrapolation artifacts, the contribution is limited to the
    wavelength span covered by the nodes, with a small floor outside the node
    support.

    The cloud mass mixing ratio vertical profile (q_c_lay) should be computed
    separately using vert_cloud functions and added to the state dictionary
    before calling this function.
    """
    wl = state["wl"]          # (nwl,) in micron
    q_c_lay = state["q_c_lay"]  # (nlay,)

    # -----------------------------
    # Retrieved / configured knobs
    # -----------------------------
    r = 10.0 ** params["log_10_cld_r"]  # particle radius (um)
    sig = params["sigma"]
    sig = jnp.maximum(sig, 1.0 + 1e-8)  # log-normal width must be >= 1
    cld_rho = params.get("cld_rho", 1.0)  # Cloud bulk density, defaults to 1.0 g/cm³

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
    # Precompute log(sig)^2 once to avoid redundant computation
    log_sig_sq = jnp.log(sig) ** 2

    # Effective radius for lognormal distribution
    r_eff = r * jnp.exp(2.5 * log_sig_sq)

    # Compute Mie/MADT efficiencies conditionally (skip wavelengths outside node support)
    Q_ext_vals, Q_sca_vals, g_vals = jax.vmap(_compute_mie_or_zero, in_axes=(0, 0, 0, None, 0))(
        wl, n, k, r_eff, wl_support_mask
    )

    # Compute cloud opacity using vertical profile from state
    # q_c_lay: (nlay,), Q_ext_vals: (nwl,) -> k_cld: (nlay, nwl)
    k_cld = (
        (3.0 * q_c_lay[:, None] * Q_ext_vals[None, :])
        / (4.0 * cld_rho * (r * 1e-4))
        * jnp.exp(0.5 * log_sig_sq)
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
        - `sigma` : float
            Log-normal width parameter (clipped to be ≥ 1).
        - `cld_rho` : float, optional
            Cloud bulk density in g cm⁻³ (defaults to 1.0).

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

    Notes
    -----
    Before using this function, load refractive index data into the registry:

    >>> from exo_skryer.registry_cloud import set_cloud_nk_data
    >>> set_cloud_nk_data(wl_data, n_data, k_data)

    IMPORTANT: The cached n,k arrays must already be on the same wavelength grid
    as the model (state["wl"]). No interpolation is performed - the arrays are
    used directly. This ensures maximum efficiency for repeated forward model calls.

    The cloud mass mixing ratio vertical profile (q_c_lay) should be computed
    separately using vert_cloud functions and added to the state dictionary
    before calling this function.

    Examples
    --------
    >>> # Load refractive index data
    >>> from exo_skryer.registry_cloud import set_cloud_nk_data
    >>> wl_nk = jnp.array([0.5, 1.0, 2.0, 5.0])
    >>> n_nk = jnp.array([1.5, 1.45, 1.4, 1.35])
    >>> k_nk = jnp.array([0.01, 0.02, 0.03, 0.04])
    >>> set_cloud_nk_data(wl_nk, n_nk, k_nk)
    >>> # Now use in opacity calculation
    >>> k_cld, ssa, g = given_nk(state, params)
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
    r = 10.0 ** params["log_10_cld_r"]  # particle radius (um)
    sig = params["sigma"]
    sig = jnp.maximum(sig, 1.0 + 1e-8)  # log-normal width must be >= 1
    cld_rho = params.get("cld_rho", 1.0)  # Cloud bulk density, defaults to 1.0 g/cm³

    # -----------------------------
    # Compute optical properties
    # -----------------------------
    # Precompute log(sig)^2 once to avoid redundant computation
    log_sig_sq = jnp.log(sig) ** 2

    # Effective radius for lognormal distribution
    r_eff = r * jnp.exp(2.5 * log_sig_sq)

    # Compute Mie/MADT efficiencies using modular blending function
    Q_ext_vals, Q_sca_vals, g_vals = jax.vmap(_compute_mie_madt_efficiencies, in_axes=(0, 0, 0, None))(
        wl, n, k, r_eff
    )

    # Compute cloud opacity using vertical profile from state
    # q_c_lay: (nlay,), Q_ext_vals: (nwl,) -> k_cld: (nlay, nwl)
    k_cld = (
        (3.0 * q_c_lay[:, None] * Q_ext_vals[None, :])
        / (4.0 * cld_rho * (r * 1e-4))
        * jnp.exp(0.5 * log_sig_sq)
    )

    ssa_wl = jnp.clip(Q_sca_vals / jnp.maximum(Q_ext_vals, 1e-30), 0.0, 1.0)
    # Use implicit broadcasting instead of broadcast_to
    ssa = ssa_wl[None, :] + jnp.zeros_like(q_c_lay[:, None])
    # Broadcast g values to (nlay, nwl)
    g = g_vals[None, :] + jnp.zeros_like(q_c_lay[:, None])

    return k_cld, ssa, g


# Aliases for backwards compatibility and alternative naming schemes
F18_cloud_2 = F18_cloud  # Placeholder - implement variant if needed
direct_nk_slab = direct_nk  # Placeholder - implement slab variant if needed
