"""
opacity_cloud.py
================

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""
from typing import Dict, Tuple, Optional
import jax
import jax.numpy as jnp
from aux_funtions import pchip_1d
from registry_cloud import get_or_create_kk_cache, compute_kk_grid_cache, KKGridCache


def zero_cloud_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del params
    layer_count = int(state["nlay"])
    wavelength_count = int(state["nwl"])
    k_cld = jnp.zeros((layer_count, wavelength_count))
    ssa = jnp.zeros_like(k_cld)
    g = jnp.zeros_like(k_cld)
    return k_cld, ssa, g


def grey_cloud(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    layer_count = int(state["nlay"])
    wavelength_count = int(state["nwl"])
    opacity_value = 10.0**jnp.asarray(params["log_10_k_cld_grey"])
    k_cld = jnp.full((layer_count, wavelength_count), opacity_value)
    ssa = jnp.zeros_like(k_cld)
    g = jnp.zeros_like(k_cld)
    return k_cld, ssa, g


def F18_cloud(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    wl = state["wl"]
    layer_count = int(state["nlay"])
    wavelength_count = int(state["nwl"])
    r = 10.0**jnp.asarray(params["log_10_cld_r"])
    Q0 = jnp.asarray(params["cld_Q0"])
    a = jnp.asarray(params["cld_a"])
    q_c = 10.0**jnp.asarray(params["log_10_q_c"])


    x = (2.0 * jnp.pi * r) / wl
    Qext = 1.0 / (Q0 * x**-a + x**0.2)


    k_cld = (3.0 * q_c * Qext)/(4.0 * (r*1e-4))


    k_map = jnp.broadcast_to(k_cld, (layer_count, wavelength_count))
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
    """
    JIT-friendly Kramers-Kronig with explicit cache argument.

    This is the optimal version for JAX: pure functional with no Python dict lookups.
    The cache is passed explicitly, making it JIT-compatible.

    Memory-efficient: Computes alpha_inv on-the-fly instead of caching it.
    For N=33219, this saves ~8.8 GB of memory with minimal performance cost
    due to JAX's JIT fusion.

    Parameters
    ----------
    nu : array (N,)
        Wavenumber grid, strictly increasing
    k_nu : array (N,)
        Extinction coefficient on nu grid
    nu_ref : scalar
        Reference wavenumber for anchoring
    n_ref : scalar
        Refractive index at nu_ref
    cache : KKGridCache
        Precomputed grid quantities (only O(N) data now)

    Returns
    -------
    n_nu : array (N,)
        Real refractive index computed via KK relation
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
    """
    Optimized singly-subtracted Kramers–Kronig using precomputed grid quantities.

    The key optimization is recognizing that y1 can be computed as:
        v = nu * k_nu  (element-wise product)
        y1[i,j] = (v[j] - v[i]) / alpha[i,j]

    This reduces the number of operations significantly.

    Parameters
    ----------
    nu : array (N,)
        Wavenumber grid, strictly increasing
    k_nu : array (N,)
        Extinction coefficient on nu grid
    nu_ref : scalar
        Reference wavenumber for anchoring
    n_ref : scalar
        Refractive index at nu_ref
    cache : KKGridCache, optional
        Precomputed grid quantities. If None, looks up from global registry.
        For JIT-compiled code, pass cache explicitly.

    Returns
    -------
    n_nu : array (N,)
        Real refractive index computed via KK relation

    Notes
    -----
    For best performance in JIT-compiled code, pre-compute the cache and pass it:
        cache = get_or_create_kk_cache(nu)

        @jax.jit
        def my_func(k_nu):
            return kk_n_from_k_wavenumber_fast(nu, k_nu, nu_ref, n_ref, cache=cache)
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
    """
    Singly-subtracted Kramers–Kronig relation (optimized version).

    Computes:
        n(nu_i) = n_ref + (2/pi) * ∫ [ (nu' k(nu') - nu_i k(nu_i))/(nu'^2-nu_i^2)
                                     - (nu' k(nu') - nu_ref k(nu_ref))/(nu'^2-nu_ref^2) ] dnu'

    This function now uses the optimized implementation with caching.
    Principal-value singularities are handled by masking.

    Notes
    -----
    - Still O(N^2) but with reduced constant factors
    - Grid-dependent quantities are cached for reuse via global registry
    - For JIT-compiled functions, pre-compute cache and pass explicitly:
        cache = get_or_create_kk_cache(nu)
        @jax.jit
        def f(k): return kk_n_from_k_wavenumber_fast(nu, k, nu_ref, n_ref, cache=cache)
    """
    return kk_n_from_k_wavenumber_fast(nu, k_nu, nu_ref, n_ref, cache=None)


def kk_n_from_k_wavelength_um(
    wl_um: jnp.ndarray,   # (N,) wavelength in micron
    k_wl: jnp.ndarray,    # (N,) extinction coefficient on wl grid
    wl_ref_um: jnp.ndarray,
    n_ref: jnp.ndarray,
    cache: Optional[KKGridCache] = None,
) -> jnp.ndarray:
    """
    Convenience wrapper: converts wl[um] -> nu[cm^-1], runs KK in nu-space,
    returns n on the original wl ordering.

    Parameters
    ----------
    wl_um : array (N,)
        Wavelength in microns
    k_wl : array (N,)
        Extinction coefficient on wavelength grid
    wl_ref_um : scalar
        Reference wavelength in microns
    n_ref : scalar
        Refractive index at reference wavelength
    cache : KKGridCache, optional
        Precomputed grid quantities for the wavenumber grid.
        If None, will be looked up from registry. For JIT, pass explicitly.

    Returns
    -------
    n_wl : array (N,)
        Real refractive index on wavelength grid
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
    """
    Compute cloud extinction opacity k_cld(lambda, p) from retrieved k(λ) nodes,
    with n(λ) computed by brute-force KK anchored at (wl_ref_um, n_ref).

    Returns
    -------
    k_cld : (nlay, nwl) cloud mass extinction coefficient (your convention)
    ssa   : (nlay, nwl) single scattering albedo
    g     : (nlay, nwl) asymmetry parameter (here zeros)
    """
    wl = jnp.asarray(state["wl"])          # (nwl,) in micron
    p_lay = jnp.asarray(state["p_lay"])    # (nlay,)

    # -----------------------------
    # Retrieved / configured knobs
    # -----------------------------
    r = 10.0 ** jnp.asarray(params["log_10_cld_r"])  # particle radius (your units)
    sig = jnp.asarray(params["sigma"])
    sig = jnp.maximum(sig, 1.0 + 1e-8)               # log-normal width must be >= 1

    # Keep n positive for scattering math sanity (doesn't forbid n<1)
    n_floor = jnp.asarray(params.get("n_floor", 1e-6))

    # -----------------------------
    # Retrieve k(wl) from log-nodes
    # -----------------------------
    node_idx = [0, 1, 2, 3, 4, 5, 6, 7]

    wl_nodes = jnp.asarray([params[f"wl_node_{i}"] for i in node_idx])
    # Limit nk contribution to the wavelength span covered by the nodes
    wl_support_min = jnp.min(wl_nodes)
    wl_support_max = jnp.max(wl_nodes)
    wl_support_mask = jnp.logical_and(wl >= wl_support_min, wl <= wl_support_max)

    # Retrieve n(wl) / k(wl) node values
    n_nodes = jnp.asarray([params[f"n_{i}"] for i in node_idx])
    log10_k_nodes = jnp.asarray([params[f"log_10_k_{i}"] for i in node_idx])

    n_interp = pchip_1d(wl, wl_nodes, n_nodes)
    log10_k_interp = pchip_1d(wl, wl_nodes, log10_k_nodes)
    n = jnp.maximum(n_interp, n_floor)
    k = jnp.maximum(10.0 ** log10_k_interp, 1e-12)
    n = jnp.where(wl_support_mask, n, n_floor)
    k = jnp.where(wl_support_mask, k, 1e-12)

    # -----------------------------
    # Cloud vertical profile
    # -----------------------------
    q_c_0  = 10.0 ** jnp.asarray(params["log_10_q_c_0"])
    H_cld  = 10.0 ** jnp.asarray(params["log_10_H_cld"])
    alpha  = 1.0 / jnp.maximum(H_cld, 1e-12)

    p_base = 10.0 ** jnp.asarray(params["log_10_p_base"]) * 1e6  # [bar -> Pa], as you had
    width_base_dex = jnp.asarray(params.get("width_base_dex", 0.25))
    d_base = jnp.maximum(width_base_dex * jnp.log(10.0), 1e-12)

    logP  = jnp.log(jnp.maximum(p_lay, 1e-30))
    logPb = jnp.log(jnp.maximum(p_base, 1e-30))

    # gate: ~1 for P <= P_base (aloft), ~0 for P >> P_base (deep)
    S_base = 0.5 * (1.0 - jnp.tanh((logP - logPb) / d_base))

    q_c_lay = q_c_0 * (p_lay / jnp.maximum(p_base, 1e-30)) ** alpha * S_base
    q_c_lay = jnp.clip(q_c_lay, 0.0)

    # Effective radius for lognormal distribution
    r_eff = r * jnp.exp(2.5 * (jnp.log(sig) ** 2))

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

    k_cld = (
        (3.0 * q_c_lay[:, None] * Q_ext_vals[None, :])
        / (4.0 * (r * 1e-4))
        * jnp.exp(0.5 * (jnp.log(sig) ** 2))
    )

    ssa_wl = jnp.clip(Q_sca_vals / jnp.maximum(Q_ext_vals, 1e-30), 0.0, 1.0)
    ssa = jnp.broadcast_to(ssa_wl[None, :], k_cld.shape)
    g = jnp.zeros_like(k_cld)

    return k_cld, ssa, g
