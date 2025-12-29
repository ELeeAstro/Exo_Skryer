"""
opacity_cloud.py
================
"""

from typing import Dict, Tuple, Optional
import jax
import jax.numpy as jnp

from .aux_functions import pchip_1d
from .registry_cloud import get_or_create_kk_cache, KKGridCache

__all__ = [
    "zero_cloud_opacity",
    "grey_cloud",
    "powerlaw_cloud",
    "F18_cloud",
    "kk_n_from_k_wavenumber_cached",
    "kk_n_from_k_wavenumber_fast",
    "kk_n_from_k_wavenumber",
    "kk_n_from_k_wavelength_um",
    "direct_nk",
    "direct_nk_slab",
    "F18_cloud_2"
]


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


def powerlaw_cloud(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute a grey + power-law cloud opacity.

    The extinction coefficient is computed as:

        k_cld(λ) = k_grey + k_powerlaw × (λ / λ_ref)^(-α)

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        State dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `nlay` : int
            Number of atmospheric layers.

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_k_cld_grey` : float
            Log₁₀ of the grey opacity component in cm² g⁻¹.
        - `log_10_k_cld_Ray` : float
            Log₁₀ amplitude of the power-law component at `wl_ref_cld` in cm² g⁻¹.
        - `alpha_cld` : float
            Power-law exponent α (e.g., α=4 gives a Rayleigh-like λ⁻⁴ slope).
        - `wl_ref_cld` : float
            Reference wavelength in microns.

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
    nlay = state["nlay"]

    # Constant grey opacity component
    k_grey = 10.0**params["log_10_k_cld_grey"]

    # Power-law amplitude at reference wavelength
    k_powerlaw = 10.0**params["log_10_k_cld_Ray"]

    # Power-law exponent (alpha=4 gives Rayleigh slope)
    alpha = params["alpha_cld"]

    # Reference wavelength (default 1.0 micron if not specified)
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
    """Compute a cloud opacity using Fisher and Heng (2018).

    This function evaluates an extinction efficiency `Qext(x)` as a function of
    size parameter `x = 2πr/λ` and converts it into a wavelength-dependent cloud
    extinction coefficient. The result is broadcast over layers.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        State dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `nlay` : int
            Number of atmospheric layers.

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_cld_r` : float
            Log₁₀ particle radius (units assumed consistent with the parameterization).
        - `cld_Q0` : float
            Extinction-efficiency scale factor.
        - `cld_a` : float
            Power-law exponent controlling the small-particle regime.
        - `log_10_q_c` : float
            Log₁₀ cloud mass-mixing parameter controlling overall opacity strength.

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
    nlay = state["nlay"]
    r = 10.0**params["log_10_cld_r"]
    Q0 = params["cld_Q0"]
    a = params["cld_a"]
    q_c = 10.0**params["log_10_q_c"]

    x = (2.0 * jnp.pi * r) / wl
    Qext = 1.0 / (Q0 * x**-a + x**0.2)

    k_cld = (3.0 * q_c * Qext) / (4.0 * (r * 1e-4))

    # Broadcast to (nlay, nwl) using implicit broadcasting
    k_map = jnp.zeros((nlay, 1)) + k_cld[None, :]
    ssa = jnp.zeros_like(k_map)
    g = jnp.zeros_like(k_map)
    return k_map, ssa, g


def kk_n_from_k_wavenumber_cached(
    nu: jnp.ndarray,
    k_nu: jnp.ndarray,
    nu_ref: jnp.ndarray,
    n_ref: jnp.ndarray,
    cache: KKGridCache,
) -> jnp.ndarray:
    """Compute `n(ν)` from `k(ν)` via a singly-subtracted Kramers–Kronig relation.

    This variant is JIT-friendly: the `KKGridCache` is passed explicitly, avoiding
    Python-side cache lookups. Grid-dependent trapezoid weights are reused via
    the cache.

    Parameters
    ----------
    nu : `~jax.numpy.ndarray`, shape (N,)
        Wavenumber grid (strictly increasing), e.g. cm⁻¹.
    k_nu : `~jax.numpy.ndarray`, shape (N,)
        Extinction coefficient on the wavenumber grid (clipped to be non-negative).
    nu_ref : `~jax.numpy.ndarray`
        Reference wavenumber used to anchor the subtraction term.
    n_ref : `~jax.numpy.ndarray`
        Real refractive index at `nu_ref`.
    cache : `~exo_skryer.registry_cloud.KKGridCache`
        Precomputed grid quantities for this `nu` grid (e.g., trapezoid weights).

    Returns
    -------
    n_nu : `~jax.numpy.ndarray`, shape (N,)
        Real refractive index on the wavenumber grid.
    """
    k_nu = jnp.maximum(k_nu, 0.0)

    # Extract cached quantities (only O(N) trapezoid weights)
    trap_weights = cache.trap_weights

    # Compute alpha_inv on-the-fly to save memory
    # For N=33219, storing this would need 8.8 GB!
    # Computing it is fast with JAX JIT fusion
    nu_i = nu[:, None]  # (N,1)
    nu_j = nu[None, :]  # (1,N)
    alpha = nu_j**2 - nu_i**2  # (N,N)
    alpha_inv = jnp.where(alpha != 0.0, 1.0 / alpha, 0.0)

    # k(nu_ref) via interpolation
    k_ref = jnp.interp(nu_ref, nu, k_nu)

    # Key optimization: compute v = nu * k_nu once
    v = nu * k_nu  # (N,)

    # y1[i,j] = (v[j] - v[i]) / alpha[i,j]
    v_diff = v[None, :] - v[:, None]  # (N,N)
    y1 = v_diff * alpha_inv

    # y2[i,j] = (v[j] - nu_ref * k_ref) / beta[j]
    beta = nu**2 - nu_ref**2
    beta_inv = jnp.where(beta != 0.0, 1.0 / beta, 0.0)
    v_ref = nu_ref * k_ref
    y2 = (v[None, :] - v_ref) * beta_inv[None, :]

    # Combined integrand
    y = y1 - y2  # (N,N)

    # Trapezoid integration using precomputed weights
    integ = jnp.sum(y * trap_weights[None, :], axis=1)  # (N,)

    n_nu = n_ref + (2.0 / jnp.pi) * integ
    return n_nu


def kk_n_from_k_wavenumber_fast(
    nu: jnp.ndarray,      # (N,) strictly increasing, e.g. cm^-1
    k_nu: jnp.ndarray,    # (N,) extinction coefficient, >= 0
    nu_ref: jnp.ndarray,  # scalar, same units as nu
    n_ref: jnp.ndarray,   # scalar
    cache: Optional[KKGridCache] = None,
) -> jnp.ndarray:
    """Optimized KK relation using precomputed grid quantities.

    Parameters
    ----------
    nu : `~jax.numpy.ndarray`, shape (N,)
        Wavenumber grid (strictly increasing), e.g. cm⁻¹.
    k_nu : `~jax.numpy.ndarray`, shape (N,)
        Extinction coefficient on the wavenumber grid (clipped to be non-negative).
    nu_ref : `~jax.numpy.ndarray`
        Reference wavenumber used to anchor the subtraction term.
    n_ref : `~jax.numpy.ndarray`
        Real refractive index at `nu_ref`.
    cache : `~exo_skryer.registry_cloud.KKGridCache`, optional
        Precomputed grid quantities for this `nu` grid. If `None`, the cache is
        obtained via `registry_cloud.get_or_create_kk_cache(nu)`.

    Returns
    -------
    n_nu : `~jax.numpy.ndarray`, shape (N,)
        Real refractive index on the wavenumber grid.

    Notes
    -----
    For best performance in JIT-compiled code, precompute the cache and pass it
    explicitly:

        cache = get_or_create_kk_cache(nu)
        n = kk_n_from_k_wavenumber_fast(nu, k_nu, nu_ref, n_ref, cache=cache)
    """
    nu = jnp.asarray(nu)
    k_nu = jnp.maximum(jnp.asarray(k_nu), 0.0)
    nu_ref = jnp.asarray(nu_ref)
    n_ref = jnp.asarray(n_ref)

    # Get cache from registry if not provided
    if cache is None:
        cache = get_or_create_kk_cache(nu)

    return kk_n_from_k_wavenumber_cached(nu, k_nu, nu_ref, n_ref, cache)


def kk_n_from_k_wavenumber(
    nu: jnp.ndarray,      # (N,) strictly increasing, e.g. cm^-1
    k_nu: jnp.ndarray,    # (N,) extinction coefficient, >= 0
    nu_ref: jnp.ndarray,  # scalar, same units as nu
    n_ref: jnp.ndarray,   # scalar
) -> jnp.ndarray:
    """Compute `n(ν)` from `k(ν)` via a singly-subtracted KK relation.

    This is a convenience wrapper around `kk_n_from_k_wavenumber_fast()` that
    looks up the grid cache internally.

    Parameters
    ----------
    nu : `~jax.numpy.ndarray`, shape (N,)
        Wavenumber grid (strictly increasing), e.g. cm⁻¹.
    k_nu : `~jax.numpy.ndarray`, shape (N,)
        Extinction coefficient on the wavenumber grid (clipped to be non-negative).
    nu_ref : `~jax.numpy.ndarray`
        Reference wavenumber used to anchor the subtraction term.
    n_ref : `~jax.numpy.ndarray`
        Real refractive index at `nu_ref`.

    Returns
    -------
    n_nu : `~jax.numpy.ndarray`, shape (N,)
        Real refractive index on the wavenumber grid.
    """
    return kk_n_from_k_wavenumber_fast(nu, k_nu, nu_ref, n_ref, cache=None)


def kk_n_from_k_wavelength_um(
    wl_um: jnp.ndarray,   # (N,) wavelength in micron
    k_wl: jnp.ndarray,    # (N,) extinction coefficient on wl grid
    wl_ref_um: jnp.ndarray,
    n_ref: jnp.ndarray,
    cache: Optional[KKGridCache] = None,
) -> jnp.ndarray:
    """Compute `n(λ)` from `k(λ)` via KK, using wavelength inputs in microns.

    This convenience wrapper converts wavelength to wavenumber via
    `ν[cm⁻¹] = 10⁴ / λ[μm]`, runs `kk_n_from_k_wavenumber_fast()` in wavenumber
    space, and returns `n` on the original wavelength ordering.

    Parameters
    ----------
    wl_um : `~jax.numpy.ndarray`, shape (N,)
        Wavelength grid in microns.
    k_wl : `~jax.numpy.ndarray`, shape (N,)
        Extinction coefficient on the wavelength grid (clipped to be non-negative).
    wl_ref_um : `~jax.numpy.ndarray`
        Reference wavelength in microns used to define `nu_ref`.
    n_ref : `~jax.numpy.ndarray`
        Real refractive index at `wl_ref_um`.
    cache : `~exo_skryer.registry_cloud.KKGridCache`, optional
        Precomputed grid quantities for the wavenumber grid. If `None`, the
        cache is obtained via `registry_cloud.get_or_create_kk_cache(nu)`.

    Returns
    -------
    n_wl : `~jax.numpy.ndarray`, shape (N,)
        Real refractive index on the wavelength grid.
    """
    wl_um = jnp.asarray(wl_um)
    k_wl = jnp.maximum(jnp.asarray(k_wl), 0.0)

    # Safety: avoid division by 0 (physically wl must be > 0 anyway)
    wl_um = jnp.maximum(wl_um, 1e-12)

    # Convert to wavenumber nu [cm^-1]
    nu = 1e4 / wl_um
    nu_ref = 1e4 / jnp.maximum(jnp.asarray(wl_ref_um), 1e-12)

    # Ensure nu is increasing for KK (reverse if needed)
    rev = nu[0] > nu[-1]
    nu_inc = jnp.where(rev, nu[::-1], nu)
    k_inc  = jnp.where(rev, k_wl[::-1], k_wl)

    n_inc = kk_n_from_k_wavenumber_fast(nu_inc, k_inc, nu_ref=nu_ref, n_ref=n_ref, cache=cache)

    # Back to original wl ordering
    n_wl = jnp.where(rev, n_inc[::-1], n_inc)
    return n_wl


def direct_nk(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute cloud optical properties from retrieved refractive-index nodes.

    This prescription retrieves node values describing the complex refractive
    index (n, k) as a function of wavelength, interpolates them onto the model
    wavelength grid, and computes a wavelength-dependent extinction coefficient.
    A simple vertical profile is applied to modulate the cloud strength with
    pressure.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `p_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer pressures (microbar convention used elsewhere in the forward model).

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing (at minimum) the retrieved node values
        and cloud-profile controls used in this function, including:

        - `wl_node_0`..`wl_node_7` : float
            Wavelength nodes (microns).
        - `n_0`..`n_7` : float
            Real refractive-index nodes.
        - `log_10_k_0`..`log_10_k_7` : float
            Log₁₀ imaginary refractive-index nodes.
        - `log_10_cld_r` : float
            Log₁₀ particle radius.
        - `sigma` : float
            Log-normal width parameter (clipped to be ≥ 1).
        - `log_10_q_c_0`, `log_10_H_cld`, `log_10_p_base` : float
            Vertical profile controls for the cloud strength.
        - `width_base_dex` : float, optional
            Transition width around the cloud base (dex).

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
    """
    wl = state["wl"]          # (nwl,) in micron
    p_lay = state["p_lay"]    # (nlay,)

    # -----------------------------
    # Retrieved / configured knobs
    # -----------------------------
    r = 10.0 ** params["log_10_cld_r"]  # particle radius (um)
    sig = params["sigma"]
    sig = jnp.maximum(sig, 1.0 + 1e-8)  # log-normal width must be >= 1

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
    # Cloud vertical profile
    # -----------------------------
    q_c_0 = 10.0 ** params["log_10_q_c_0"]
    H_cld = 10.0 ** params["log_10_H_cld"]
    alpha = 1.0 / jnp.maximum(H_cld, 1e-12)

    p_base = 10.0 ** params["log_10_p_base"] * 1e6 
    width_base_dex = params.get("width_base_dex", 0.25)
    d_base = jnp.maximum(width_base_dex * jnp.log(10.0), 1e-12)

    logP = jnp.log(jnp.maximum(p_lay, 1e-30))
    logPb = jnp.log(jnp.maximum(p_base, 1e-30))

    # gate: ~1 for P <= P_base (aloft), ~0 for P >> P_base (deep)
    S_base = 0.5 * (1.0 - jnp.tanh((logP - logPb) / d_base))

    q_c_lay = q_c_0 * (p_lay / jnp.maximum(p_base, 1e-30)) ** alpha * S_base
    q_c_lay = jnp.clip(q_c_lay, 0.0)

    # Precompute log(sig)^2 once to avoid redundant computation
    log_sig_sq = jnp.log(sig) ** 2

    # Effective radius for lognormal distribution
    r_eff = r * jnp.exp(2.5 * log_sig_sq)

    def _compute_active(args):
        wl_val, n_val, k_val = args
        x = 2.0 * jnp.pi * r_eff / jnp.maximum(wl_val, 1e-12)

        m = n_val + 1j * k_val
        m2 = m * m
        alp = (m2 - 1.0) / (m2 + 2.0)

        term = 1.0 + (x**2 / 15.0) * alp * ((m2 * m2 + 27.0 * m2 + 38.0) / (2.0 * m2 + 3.0))
        Q_abs_ray = 4.0 * x * jnp.imag(alp * term)
        Q_sca_ray = (8.0 / 3.0) * x**4 * jnp.real(alp**2)
        Q_ext_ray = Q_abs_ray + Q_sca_ray

        k_min = 1e-12
        k_eff = jnp.maximum(k_val, k_min)

        dn = n_val - 1.0
        dn_safe = jnp.where(jnp.abs(dn) < 1e-12, jnp.sign(dn + 1e-30) * 1e-12, dn)

        rho = 2.0 * x * dn_safe
        rho_safe = jnp.where(jnp.abs(rho) < 1e-12, jnp.sign(rho + 1e-30) * 1e-12, rho)

        beta = jnp.arctan2(k_eff, dn_safe)
        tan_b = jnp.tan(beta)

        exp_arg = -rho_safe * tan_b
        exp_arg = jnp.clip(exp_arg, -80.0, 80.0)
        exp_rho = jnp.exp(exp_arg)

        cosb_over_rho = jnp.cos(beta) / rho_safe

        Q_ext_madt = (
            2.0
            - 4.0 * exp_rho * cosb_over_rho * jnp.sin(rho - beta)
            - 4.0 * exp_rho * (cosb_over_rho**2) * jnp.cos(rho - 2.0 * beta)
            + 4.0 * (cosb_over_rho**2) * jnp.cos(2.0 * beta)
        )

        z = 4.0 * k_eff * x
        z_safe = jnp.maximum(z, 1e-30)
        exp_z = jnp.exp(jnp.clip(-z_safe, -80.0, 80.0))

        Q_abs_madt = 1.0 + 2.0 * (exp_z / z_safe) + 2.0 * ((exp_z - 1.0) / (z_safe * z_safe))

        C1 = 0.25 * (1.0 + jnp.exp(-1167.0 * k_eff)) * (1.0 - Q_abs_madt)

        eps = 0.25 + 0.61 * (1.0 - jnp.exp(-(8.0 * jnp.pi / 3.0) * k_eff)) ** 2
        C2 = (
            jnp.sqrt(2.0 * eps * (x / jnp.pi))
            * jnp.exp(0.5 - eps * (x / jnp.pi))
            * (0.79393 * n_val - 0.6069)
        )

        Q_abs_madt = (1.0 + C1 + C2) * Q_abs_madt

        Q_edge = (1.0 - jnp.exp(-0.06 * x)) * x ** (-2.0 / 3.0)
        Q_ext_madt = (1.0 + 0.5 * C2) * Q_ext_madt + Q_edge
        Q_sca_madt = Q_ext_madt - Q_abs_madt

        t = jnp.clip((x - 1.0) / 2.0, 0.0, 1.0)
        w = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3

        Q_ext = (1.0 - w) * Q_ext_ray + w * Q_ext_madt
        Q_sca = (1.0 - w) * Q_sca_ray + w * Q_sca_madt
        return Q_ext, Q_sca

    def _skip_active(args):
        del args
        return 0.0, 0.0

    def _per_wavelength(wl_val, n_val, k_val, active):
        return jax.lax.cond(active, _compute_active, _skip_active, (wl_val, n_val, k_val))

    Q_ext_vals, Q_sca_vals = jax.vmap(_per_wavelength)(wl, n, k, wl_support_mask)

    # Reuse precomputed log_sig_sq
    k_cld = (
        (3.0 * q_c_lay[:, None] * Q_ext_vals[None, :])
        / (4.0 * (r * 1e-4))
        * jnp.exp(0.5 * log_sig_sq)
    )

    ssa_wl = jnp.clip(Q_sca_vals / jnp.maximum(Q_ext_vals, 1e-30), 0.0, 1.0)
    # Use implicit broadcasting instead of broadcast_to
    ssa = ssa_wl[None, :] + jnp.zeros_like(q_c_lay[:, None])
    g = jnp.zeros_like(k_cld)

    return k_cld, ssa, g


def direct_nk_slab(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute cloud optical properties from retrieved n-k nodes with a simple slab profile.

    This is a simplified version of `direct_nk()` that uses a pressure-slab vertical
    profile instead of a complex exponential + gate profile. The cloud is present
    between `P_top_slab` and `P_top_slab * 10^(Delta_log_P)`, with hard cutoffs.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `p_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer pressures (microbar convention used elsewhere in the forward model).

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `wl_node_0`..`wl_node_7` : float
            Wavelength nodes (microns).
        - `n_0`..`n_7` : float
            Real refractive-index nodes.
        - `log_10_k_0`..`log_10_k_7` : float
            Log₁₀ imaginary refractive-index nodes.
        - `log_10_cld_r` : float
            Log₁₀ particle radius.
        - `sigma` : float
            Log-normal width parameter (clipped to be ≥ 1).
        - `log_10_q_c` : float
            Log₁₀ cloud mass-mixing ratio inside the slab.
        - `log_p_top_slab` : float
            Log₁₀ pressure at the top of the slab (in bars).
        - `dlog_p_slab` : float
            Extent of the slab in log pressure space (positive downward).

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
    The slab extends from `P_top = 10^log_P_top_slab` to `P_bot = P_top * 10^Delta_log_P`.
    Smooth edges are applied using tanh functions to avoid numerical discontinuities.
    """
    wl = state["wl"]          # (nwl,) in micron
    p_lay = state["p_lay"]    # (nlay,) in microbar

    # -----------------------------
    # Retrieved / configured knobs
    # -----------------------------
    r = 10.0 ** params["log_10_cld_r"]  # particle radius (um)
    sig = params["sigma"]
    sig = jnp.maximum(sig, 1.0 + 1e-8)  # log-normal width must be >= 1

    # Keep n positive for scattering math sanity
    n_floor = 1e-6

    # -----------------------------
    # Retrieve k(wl) from log-nodes
    # -----------------------------
    wl_nodes = jnp.stack([params[f"wl_node_{i}"] for i in range(13)])
    wl_support_min = jnp.min(wl_nodes)
    wl_support_max = jnp.max(wl_nodes)
    wl_support_mask = jnp.logical_and(wl >= wl_support_min, wl <= wl_support_max)

    n_nodes = jnp.stack([params[f"n_{i}"] for i in range(13)])
    log10_k_nodes = jnp.stack([params[f"log_10_k_{i}"] for i in range(13)])

    n_interp = pchip_1d(wl, wl_nodes, n_nodes)
    log10_k_interp = pchip_1d(wl, wl_nodes, log10_k_nodes)
    n = jnp.maximum(n_interp, n_floor)
    k = jnp.maximum(10.0 ** log10_k_interp, 1e-12)
    n = jnp.where(wl_support_mask, n, n_floor)
    k = jnp.where(wl_support_mask, k, 1e-12)

    # -----------------------------
    # Cloud slab vertical profile
    # -----------------------------
    q_c_slab = 10.0 ** params["log_10_q_c"]

    # Slab boundaries in pressure (bars)
    log_P_top = params["log_10_p_top_slab"]
    Delta_log_P = params["log_10_dp_slab"]

    # Convert to microbar (state p_lay is in microbar)
    P_top = 10.0 ** log_P_top * 1e6  # bars -> microbar
    P_bot = 10.0 ** (log_P_top + Delta_log_P) * 1e6  # bars -> microbar

    # Hard slab cutoff: 1 inside [P_top, P_bot], 0 outside
    slab_mask = jnp.logical_and(p_lay >= P_top, p_lay <= P_bot)
    q_c_lay = q_c_slab * slab_mask  # shape (nlay,)

    # Precompute log(sig)^2 once to avoid redundant computation
    log_sig_sq = jnp.log(sig) ** 2

    # Effective radius for lognormal distribution
    r_eff = r * jnp.exp(2.5 * log_sig_sq)

    def _compute_active(args):
        wl_val, n_val, k_val = args
        x = 2.0 * jnp.pi * r_eff / jnp.maximum(wl_val, 1e-12)

        m = n_val + 1j * k_val
        m2 = m * m
        alp = (m2 - 1.0) / (m2 + 2.0)

        term = 1.0 + (x**2 / 15.0) * alp * ((m2 * m2 + 27.0 * m2 + 38.0) / (2.0 * m2 + 3.0))
        Q_abs_ray = 4.0 * x * jnp.imag(alp * term)
        Q_sca_ray = (8.0 / 3.0) * x**4 * jnp.real(alp**2)
        Q_ext_ray = Q_abs_ray + Q_sca_ray

        k_min = 1e-12
        k_eff = jnp.maximum(k_val, k_min)

        dn = n_val - 1.0
        dn_safe = jnp.where(jnp.abs(dn) < 1e-12, jnp.sign(dn + 1e-30) * 1e-12, dn)

        rho = 2.0 * x * dn_safe
        rho_safe = jnp.where(jnp.abs(rho) < 1e-12, jnp.sign(rho + 1e-30) * 1e-12, rho)

        beta = jnp.arctan2(k_eff, dn_safe)
        tan_b = jnp.tan(beta)

        exp_arg = -rho_safe * tan_b
        exp_arg = jnp.clip(exp_arg, -80.0, 80.0)
        exp_rho = jnp.exp(exp_arg)

        cosb_over_rho = jnp.cos(beta) / rho_safe

        Q_ext_madt = (
            2.0
            - 4.0 * exp_rho * cosb_over_rho * jnp.sin(rho - beta)
            - 4.0 * exp_rho * (cosb_over_rho**2) * jnp.cos(rho - 2.0 * beta)
            + 4.0 * (cosb_over_rho**2) * jnp.cos(2.0 * beta)
        )

        z = 4.0 * k_eff * x
        z_safe = jnp.maximum(z, 1e-30)
        exp_z = jnp.exp(jnp.clip(-z_safe, -80.0, 80.0))

        Q_abs_madt = 1.0 + 2.0 * (exp_z / z_safe) + 2.0 * ((exp_z - 1.0) / (z_safe * z_safe))

        C1 = 0.25 * (1.0 + jnp.exp(-1167.0 * k_eff)) * (1.0 - Q_abs_madt)

        eps = 0.25 + 0.61 * (1.0 - jnp.exp(-(8.0 * jnp.pi / 3.0) * k_eff)) ** 2
        C2 = (
            jnp.sqrt(2.0 * eps * (x / jnp.pi))
            * jnp.exp(0.5 - eps * (x / jnp.pi))
            * (0.79393 * n_val - 0.6069)
        )

        Q_abs_madt = (1.0 + C1 + C2) * Q_abs_madt

        Q_edge = (1.0 - jnp.exp(-0.06 * x)) * x ** (-2.0 / 3.0)
        Q_ext_madt = (1.0 + 0.5 * C2) * Q_ext_madt + Q_edge
        Q_sca_madt = Q_ext_madt - Q_abs_madt

        t = jnp.clip((x - 1.0) / 2.0, 0.0, 1.0)
        w = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3

        Q_ext = (1.0 - w) * Q_ext_ray + w * Q_ext_madt
        Q_sca = (1.0 - w) * Q_sca_ray + w * Q_sca_madt
        return Q_ext, Q_sca

    def _skip_active(args):
        del args
        return 0.0, 0.0

    def _per_wavelength(wl_val, n_val, k_val, active):
        return jax.lax.cond(active, _compute_active, _skip_active, (wl_val, n_val, k_val))

    Q_ext_vals, Q_sca_vals = jax.vmap(_per_wavelength)(wl, n, k, wl_support_mask)

    # Extinction coefficient with slab profile
    # q_c_lay: (nlay,), Q_ext_vals: (nwl,) -> k_cld: (nlay, nwl)
    k_cld = (
        (3.0 * q_c_lay[:, None] * Q_ext_vals[None, :])
        / (4.0 * (r * 1e-4))
        * jnp.exp(0.5 * log_sig_sq)
    )

    ssa_wl = jnp.clip(Q_sca_vals / jnp.maximum(Q_ext_vals, 1e-30), 0.0, 1.0)
    ssa = ssa_wl[None, :] + jnp.zeros_like(k_cld)
    g = jnp.zeros_like(k_cld)

    return k_cld, ssa, g


def F18_cloud_2(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute a vertically varying empirical cloud extinction model.

    This is a variant of `F18_cloud()` that applies a pressure-dependent cloud
    mass-mixing profile (`q_c_lay`) to modulate the extinction with altitude.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `p_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer pressures (microbar convention used elsewhere in the forward model).

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:

        - `log_10_cld_r`, `sigma` : float
            Particle size distribution parameters.
        - `cld_Q0`, `cld_a` : float
            Extinction-efficiency parameters.
        - `log_10_q_c_0`, `log_10_H_cld`, `log_10_p_base` : float
            Vertical profile controls for the cloud strength.
        - `width_base_dex` : float, optional
            Transition width around the cloud base (dex).

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
    p_lay = state["p_lay"]

    r = 10.0 ** params["log_10_cld_r"]
    sig = params["sigma"]
    sig = jnp.maximum(sig, 1.0 + 1e-8)

    Q0 = params["cld_Q0"]
    a = params["cld_a"]

    q_c_0 = 10.0 ** params["log_10_q_c_0"]
    H_cld = 10.0 ** params["log_10_H_cld"]
    alpha = 1.0 / jnp.maximum(H_cld, 1e-12)

    p_base = 10.0 ** params["log_10_p_base"] * 1e6
    width_base_dex = params.get("width_base_dex", 0.25)
    d_base = jnp.maximum(width_base_dex * jnp.log(10.0), 1e-12)

    logP = jnp.log(jnp.maximum(p_lay, 1e-30))
    logPb = jnp.log(jnp.maximum(p_base, 1e-30))
    S_base = 0.5 * (1.0 - jnp.tanh((logP - logPb) / d_base))

    q_c_lay = q_c_0 * (p_lay / jnp.maximum(p_base, 1e-30)) ** alpha * S_base
    q_c_lay = jnp.clip(q_c_lay, 0.0)

    # Precompute log(sig)^2 once to avoid redundant computation
    log_sig_sq = jnp.log(sig) ** 2

    r_eff = r * jnp.exp(2.5 * log_sig_sq)
    x = (2.0 * jnp.pi * r_eff) / jnp.maximum(wl, 1e-12)
    Qext = 1.0 / (Q0 * x**-a + x**0.2)

    # Reuse precomputed log_sig_sq
    k_cld = (
        (3.0 * q_c_lay[:, None] * Qext[None, :])
        / (4.0 * (r * 1e-4))
        * jnp.exp(0.5 * log_sig_sq)
    )

    # k_cld already has correct shape, no need for broadcast_to
    ssa = jnp.zeros_like(k_cld)
    g = jnp.zeros_like(k_cld)
    return k_cld, ssa, g
