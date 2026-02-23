"""
opacity_cloud.py
================
"""

from typing import Dict, Tuple, Optional
import jax
import jax.numpy as jnp

from .aux_functions import pchip_1d
from .mie_schemes import rayleigh, madt
from .lxmie_mod import lxmie_jax

__all__ = [
    "compute_cloud_efficiencies",
    "compute_cloud_opacity",
    "zero_cloud_opacity",
    "grey_cloud",
    "deck_and_powerlaw",
    "F18_cloud",
    "direct_nk",
]

_LXMIE_NMAX = 2000
_LXMIE_CF_MAX_TERMS = 2000
_LXMIE_CF_EPS = 1e-10
_DIV_EPS = 1e-30
_QC_EPS = 1e-30


def _safe_div(num: jnp.ndarray, den: jnp.ndarray) -> jnp.ndarray:
    """Elementwise num/den with 0 where den==0 (avoids NaNs in cloud-free layers)."""
    return jnp.where(den != 0, num / den, 0.0)

def compute_cloud_efficiencies(
    wl: jnp.ndarray,
    r_cm: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    *,
    eff_scheme: str,
    n: Optional[jnp.ndarray] = None,
    k: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Second-stage landing function for (Q_ext, Q_sca, g).

    Parameters
    ----------
    wl : `~jax.numpy.ndarray`, shape (nwl,)
        Wavelength grid in microns.
    r_cm : `~jax.numpy.ndarray`, shape (nr,) or scalar
        Particle radius in cm.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameters needed by the selected scheme.
    n, k : `~jax.numpy.ndarray`, optional
        Refractive-index arrays on the same wl grid (shape (nwl,)). For all nk-based
        schemes, these must be provided (the "physics" pathway uses cached n,k from
        the registry/opac_cache). The node-interpolation pathway is kept only in
        `direct_nk`.
    eff_scheme : str
        Efficiency scheme identifier. Current options:
        - "f18": Fisher & Heng (2018) Qext model (Qsca=0, g=0 for now)
        - "mie_madt": Rayleigh + MADT blend using retrieved n,k nodes
        - "lxmie": full Lorenz-Mie (LX-MIE) using retrieved n,k nodes

    Returns
    -------
    Q_ext, Q_sca, g : arrays
        Efficiencies and asymmetry parameter. Shape is (nr, nwl) if r_cm is a
        vector, else (nwl,) for scalar radius.
    """
    scheme = eff_scheme.lower().strip()

    if scheme in ("f18", "fisher18", "fisher_heng"):
        return F18_cloud(wl, r_cm, params)

    # nk-based schemes: physics pathway only (cached n,k provided by caller).
    if n is None or k is None:
        raise ValueError(
            "compute_cloud_efficiencies: n and k must be provided for nk-based schemes "
            "(use cached n,k from the registry/opac_cache)."
        )
    wl_support_mask = jnp.ones_like(wl, dtype=bool)

    if scheme in ("mie_madt", "madt", "rayleigh_madt"):
        return _efficiencies_mie_madt(wl, r_cm, n, k, wl_support_mask)

    if scheme in ("lxmie", "mie", "mie_full", "full_mie"):
        return _efficiencies_lxmie(
            wl,
            r_cm,
            n,
            k,
            wl_support_mask,
            nmax=_LXMIE_NMAX,
            cf_max_terms=_LXMIE_CF_MAX_TERMS,
            cf_eps=_LXMIE_CF_EPS,
        )

    raise ValueError(
        f"Unknown eff_scheme='{eff_scheme}'. "
        "Valid options: f18, mie_madt, lxmie."
    )


def _efficiencies_mie_madt(
    wl: jnp.ndarray,
    r_cm: jnp.ndarray,
    n: jnp.ndarray,
    k: jnp.ndarray,
    wl_support_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Rayleigh + MADT blend on a (r, wl) grid using broadcasting."""
    r_um = r_cm * 1e4

    if r_um.ndim == 0:
        x = 2.0 * jnp.pi * r_um / wl
    else:
        x = 2.0 * jnp.pi * r_um[:, None] / wl[None, :]

    Q_ext_ray, Q_sca_ray, g_ray = rayleigh(n, k, x)
    Q_ext_madt, Q_sca_madt, g_madt = madt(n, k, x)

    # Smooth blend between Rayleigh (x=1.0) and MADT (x=3.0) using smootherstep
    t = jnp.clip((x - 1.0) / 2.0, 0.0, 1.0)
    w = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
    Q_ext = (1.0 - w) * Q_ext_ray + w * Q_ext_madt
    Q_sca = (1.0 - w) * Q_sca_ray + w * Q_sca_madt
    g = (1.0 - w) * g_ray + w * g_madt

    # Mask out wavelengths outside the nk node span.
    if r_um.ndim == 0:
        Q_ext = jnp.where(wl_support_mask, Q_ext, 0.0)
        Q_sca = jnp.where(wl_support_mask, Q_sca, 0.0)
        g = jnp.where(wl_support_mask, g, 0.0)
    else:
        Q_ext = jnp.where(wl_support_mask[None, :], Q_ext, 0.0)
        Q_sca = jnp.where(wl_support_mask[None, :], Q_sca, 0.0)
        g = jnp.where(wl_support_mask[None, :], g, 0.0)

    return Q_ext, Q_sca, g


def _efficiencies_lxmie(
    wl: jnp.ndarray,
    r_cm: jnp.ndarray,
    n: jnp.ndarray,
    k: jnp.ndarray,
    wl_support_mask: jnp.ndarray,
    *,
    nmax: int,
    cf_max_terms: int,
    cf_eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full Mie efficiencies (LX-MIE) on a (r, wl) grid."""
    r_um = r_cm * 1e4
    ri = (n + 1.0j * k).astype(jnp.complex128)  # lxmie_mod expects n + i k

    def _one_radius(r_um_val: jnp.ndarray):
        x_wl = (2.0 * jnp.pi * r_um_val) / wl

        def _one_wl(x_val, ri_val, in_support):
            def do():
                q_ext, q_sca, _q_abs, g = lxmie_jax(
                    ri_val, x_val, nmax=nmax, cf_max_terms=cf_max_terms, cf_eps=cf_eps
                )
                return q_ext, q_sca, g

            def skip():
                z = jnp.zeros_like(x_val)
                return z, z, z

            return jax.lax.cond(in_support, do, skip)

        Q_ext, Q_sca, g = jax.vmap(_one_wl, in_axes=(0, 0, 0))(x_wl, ri, wl_support_mask)
        return Q_ext, Q_sca, g

    if r_um.ndim == 0:
        return _one_radius(r_um)

    Q_ext, Q_sca, g = jax.vmap(_one_radius, in_axes=(0,))(r_um)
    return Q_ext, Q_sca, g


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


def _compute_mie_madt_efficiencies_masked(
    wl: jnp.ndarray,
    n: jnp.ndarray,
    k: jnp.ndarray,
    r_eff: jnp.ndarray,
    wl_support_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Mie/MADT efficiencies with wavelength support masking."""
    return jax.vmap(_compute_mie_or_zero, in_axes=(0, 0, 0, None, 0))(
        wl, n, k, r_eff, wl_support_mask
    )


def compute_cloud_efficiencies_cached_nk(
    wl: jnp.ndarray,
    r_cm: jnp.ndarray,
    n: jnp.ndarray,
    k: jnp.ndarray,
    *,
    eff_scheme: str = "mie_madt",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convenience wrapper: compute efficiencies using cached n,k arrays."""
    return compute_cloud_efficiencies(wl, r_cm, {}, eff_scheme=eff_scheme, n=n, k=k)


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
        - `"F18"`: Fisher & Heng (2018) empirical model
        - `"madt_rayleigh"`: Mie/MADT blend using cached n,k on master grid
        - `"lxmie"`: Full Lorenz-Mie (LX-MIE) using cached n,k on master grid
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
    # First check if zero cloud, grey or deck and powerlaw cloud
    if scheme_lower in ("none", "zero", "off", "no_cloud"):
        return zero_cloud_opacity(state, params)
    elif scheme_lower in ("grey", "gray"):
        return grey_cloud(state, params)    
    elif scheme_lower in ("powerlaw", "power_law", "deck_and_powerlaw"):
        return deck_and_powerlaw(state, params)
    elif scheme_lower in ("direct_nk", "nk"):
        # Keep this pathway self-contained (legacy-style).
        return direct_nk(state, params)
    elif scheme_lower in ("madt_rayleigh", "madt-rayleigh", "mie_madt"):
        return _cached_nk_mie_cloud(state, params, eff_scheme="mie_madt")
    elif scheme_lower in ("lxmie", "mie_full", "full_mie"):
        return _cached_nk_mie_cloud(state, params, eff_scheme="lxmie")
    elif scheme_lower not in ("f18", "fisher18", "fisher_heng"):
        raise ValueError(
            f"Unknown cloud opacity scheme: '{opacity_scheme}'. "
            "Valid options: none, grey, powerlaw, direct_nk, f18, madt_rayleigh, lxmie"
        )

    # ------------------------------------------------------------
    # Microphysical clouds: distribution -> efficiencies -> opacity (F18)
    # ------------------------------------------------------------
    wl = state["wl"]               # (nwl,) microns
    rho_a = state["rho_lay"]       # (nlay,) g cm^-3
    q_c = state["q_c_lay"]         # (nlay,) dimensionless
    q_c = jnp.where(q_c > _QC_EPS, q_c, 0.0)
    rho_d = params["cld_rho"]      # g cm^-3

    # Particle size distribution code:
    # 1 = monodisperse, 2 = polydisperse (lognormal)
    cloud_dist_code = jnp.asarray(params.get("cloud_dist", 1), dtype=jnp.int32)

    eff_scheme = "f18"

    # Retrieved / configured radius parameter is in microns.
    r_um = 10.0 ** params["log_10_cld_r"]
    r_cm = r_um * 1e-4

    # If there is no cloud mass anywhere, skip all microphysics/Mie work.
    # This avoids expensive lxmie/madt computations in cloud-free atmospheres.
    has_cloud_any = jnp.any(q_c > 0)

    def _poly_case(_):
        # lax.cond traces both branches; use .get default so monodisperse configs
        # don't require lognormal params to be present.
        sig_g = params.get("cld_sigma", jnp.asarray(1.0))
        lnsig2 = jnp.log(sig_g) ** 2
        # Total number density implied by q_c for a lognormal (geometric-mean) radius.
        N0 = (3.0 * rho_a * q_c) / (4.0 * jnp.pi * rho_d * r_cm**3) * jnp.exp(-4.5 * lnsig2)  # (nlay,)

        # Radius grid bounds are provided in microns.
        log_10_r_min = jnp.log10(1e-3 * 1e-4) #jnp.log10(params["r_min"])
        log_10_r_max = jnp.log10(10.0 * 1e-4)#jnp.log10(params["r_max"])
        nr = 20 #params["nr"]
        r_grid_cm = jnp.logspace(log_10_r_min, log_10_r_max, nr) * 1e-4  # (nr,) cm

        # Spectral number density n(r) [cm^-3 cm^-1], evaluated on-the-fly in scan.
        ln_sigma = jnp.log(sig_g)
        prefac = N0 / (jnp.sqrt(2.0 * jnp.pi) * ln_sigma)  # (nlay,)

        # Trapezoid weights on a non-uniform grid: integral y(r) dr ~= sum_i w_i * y_i
        dr = jnp.diff(r_grid_cm)  # (nr-1,)
        w_trap = jnp.concatenate(
            [
                dr[:1] / 2.0,
                (dr[:-1] + dr[1:]) / 2.0,
                dr[-1:] / 2.0,
            ],
            axis=0,
        )  # (nr,)

        def _accum(carry, r_w):
            alpha_ext, alpha_sca, alpha_sca_g = carry
            r_i, w_i = r_w  # cm, cm

            # n(r_i) for each layer [cm^-3 cm^-1]
            log_ratio_i = jnp.log(r_i / r_cm)  # (nlay,)
            exponent_i = -0.5 * (log_ratio_i / ln_sigma) ** 2  # (nlay,)
            f_i = prefac * jnp.exp(exponent_i) / r_i  # (nlay,)

            # Q over wavelength at this radius (nwl,)
            Q_ext_i, Q_sca_i, g_i = compute_cloud_efficiencies(wl, r_i, params, eff_scheme=eff_scheme)

            area_i = jnp.pi * (r_i**2)  # cm^2
            dA = (w_i * area_i)  # cm^3

            # Add contribution: ∫ n(r) Q πr^2 dr  -> units cm^-1
            alpha_ext = alpha_ext + (f_i[:, None] * Q_ext_i[None, :] * dA)
            alpha_sca = alpha_sca + (f_i[:, None] * Q_sca_i[None, :] * dA)
            alpha_sca_g = alpha_sca_g + (f_i[:, None] * Q_sca_i[None, :] * g_i[None, :] * dA)
            return (alpha_ext, alpha_sca, alpha_sca_g), None

        alpha0 = jnp.zeros((rho_a.shape[0], wl.shape[0]), dtype=wl.dtype)
        (alpha_ext, alpha_sca, alpha_sca_g), _ = jax.lax.scan(
            _accum,
            (alpha0, alpha0, alpha0),
            (r_grid_cm, w_trap),
        )

        # Convert to mass opacities [cm^2 g^-1] by dividing by rho_a.
        k_ext = alpha_ext / rho_a[:, None]
        k_sca = alpha_sca / rho_a[:, None]

        # Scattering-weighted asymmetry parameter.
        g = _safe_div(alpha_sca_g, alpha_sca)
        ssa = _safe_div(k_sca, k_ext)
        return k_ext, ssa, g

    def _mono_case(_):
        # Monodisperse: N0 comes from condensate mass per volume with radius r_cm.
        N0 = (3.0 * rho_a * q_c) / (4.0 * jnp.pi * rho_d * r_cm**3)  # (nlay,) cm^-3
        Q_ext_wl, Q_sca_wl, g_wl = compute_cloud_efficiencies(wl, r_cm, params, eff_scheme=eff_scheme)  # (nwl,)

        alpha_ext = N0[:, None] * Q_ext_wl[None, :] * (jnp.pi * r_cm**2)  # (nlay, nwl) cm^-1
        alpha_sca = N0[:, None] * Q_sca_wl[None, :] * (jnp.pi * r_cm**2)  # (nlay, nwl) cm^-1
        k_ext = alpha_ext / rho_a[:, None]
        k_sca = alpha_sca / rho_a[:, None]
        ssa = _safe_div(k_sca, k_ext)
        g = g_wl[None, :] + jnp.zeros_like(rho_a[:, None])
        q_mask = (q_c > 0)[:, None]
        k_ext = jnp.where(q_mask, k_ext, 0.0)
        ssa = jnp.where(q_mask, ssa, 0.0)
        g = jnp.where(q_mask, g, 0.0)
        return k_ext, ssa, g

    def _do_cloud(_):
        return jax.lax.cond(cloud_dist_code == 2, _poly_case, _mono_case, operand=None)

    def _skip_cloud(_):
        zeros = jnp.zeros((state["nlay"], state["nwl"]), dtype=wl.dtype)
        return zeros, zeros, zeros

    return jax.lax.cond(has_cloud_any, _do_cloud, _skip_cloud, operand=None)


def _cached_nk_mie_cloud(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    *,
    eff_scheme: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cloud opacity from cached n,k + size distribution integration."""
    wl = state["wl"]               # (nwl,) microns
    rho_a = state["rho_lay"]       # (nlay,) g cm^-3
    q_c = state["q_c_lay"]         # (nlay,) dimensionless
    q_c = jnp.where(q_c > _QC_EPS, q_c, 0.0)
    rho_d = params["cld_rho"]      # g cm^-3

    n = state["cloud_nk_n"]        # (nwl,)
    k = state["cloud_nk_k"]        # (nwl,)

    cloud_dist_code = jnp.asarray(params.get("cloud_dist", 1), dtype=jnp.int32)

    r_um = 10.0 ** params["log_10_cld_r"]
    r_cm = r_um * 1e-4

    has_cloud_any = jnp.any(q_c > 0)

    def _mono_case(_):
        N0 = (3.0 * rho_a * q_c) / (4.0 * jnp.pi * rho_d * r_cm**3)  # (nlay,) cm^-3
        Q_ext_wl, Q_sca_wl, g_wl = compute_cloud_efficiencies(wl, r_cm, params, eff_scheme=eff_scheme, n=n, k=k)
        alpha_ext = N0[:, None] * Q_ext_wl[None, :] * (jnp.pi * r_cm**2)
        alpha_sca = N0[:, None] * Q_sca_wl[None, :] * (jnp.pi * r_cm**2)
        k_ext = alpha_ext / rho_a[:, None]
        k_sca = alpha_sca / rho_a[:, None]
        ssa = _safe_div(k_sca, k_ext)
        g = g_wl[None, :] + jnp.zeros_like(rho_a[:, None])
        q_mask = (q_c > 0)[:, None]
        k_ext = jnp.where(q_mask, k_ext, 0.0)
        ssa = jnp.where(q_mask, ssa, 0.0)
        g = jnp.where(q_mask, g, 0.0)
        return k_ext, ssa, g

    def _poly_case(_):
        sig_g = params.get("cld_sigma", jnp.asarray(1.0))
        lnsig2 = jnp.log(sig_g) ** 2
        N0 = (3.0 * rho_a * q_c) / (4.0 * jnp.pi * rho_d * r_cm**3) * jnp.exp(-4.5 * lnsig2)  # (nlay,)

        # NOTE: r_grid is currently hard-baked/static elsewhere in your setup.
        log_10_r_min = 1e-3 * 1e-4
        log_10_r_max = 10.0 * 1e-4
        nr = 20
        r_grid_cm = jnp.logspace(log_10_r_min, log_10_r_max, nr) * 1e-4  # (nr,) cm

        ln_sigma = jnp.log(sig_g)
        prefac = N0 / (jnp.sqrt(2.0 * jnp.pi) * ln_sigma)  # (nlay,)

        dr = jnp.diff(r_grid_cm)
        w_trap = jnp.concatenate([dr[:1] / 2.0, (dr[:-1] + dr[1:]) / 2.0, dr[-1:] / 2.0], axis=0)

        def _accum(carry, r_w):
            alpha_ext, alpha_sca, alpha_sca_g = carry
            r_i, w_i = r_w
            log_ratio_i = jnp.log(r_i / r_cm)
            exponent_i = -0.5 * (log_ratio_i / ln_sigma) ** 2
            f_i = prefac * jnp.exp(exponent_i) / r_i

            Q_ext_i, Q_sca_i, g_i = compute_cloud_efficiencies(wl, r_i, params, eff_scheme=eff_scheme, n=n, k=k)
            dA = w_i * jnp.pi * (r_i**2)
            alpha_ext = alpha_ext + (f_i[:, None] * Q_ext_i[None, :] * dA)
            alpha_sca = alpha_sca + (f_i[:, None] * Q_sca_i[None, :] * dA)
            alpha_sca_g = alpha_sca_g + (f_i[:, None] * Q_sca_i[None, :] * g_i[None, :] * dA)
            return (alpha_ext, alpha_sca, alpha_sca_g), None

        alpha0 = jnp.zeros((rho_a.shape[0], wl.shape[0]), dtype=wl.dtype)
        (alpha_ext, alpha_sca, alpha_sca_g), _ = jax.lax.scan(_accum, (alpha0, alpha0, alpha0), (r_grid_cm, w_trap))
        k_ext = alpha_ext / rho_a[:, None]
        k_sca = alpha_sca / rho_a[:, None]
        ssa = _safe_div(k_sca, k_ext)
        g = _safe_div(alpha_sca_g, alpha_sca)
        q_mask = (q_c > 0)[:, None]
        k_ext = jnp.where(q_mask, k_ext, 0.0)
        ssa = jnp.where(q_mask, ssa, 0.0)
        g = jnp.where(q_mask, g, 0.0)
        return k_ext, ssa, g

    def _do_cloud(_):
        return jax.lax.cond(cloud_dist_code == 2, _poly_case, _mono_case, operand=None)

    def _skip_cloud(_):
        zeros = jnp.zeros((state["nlay"], state["nwl"]), dtype=wl.dtype)
        return zeros, zeros, zeros

    return jax.lax.cond(has_cloud_any, _do_cloud, _skip_cloud, operand=None)


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


def F18_cloud(
    wl: jnp.ndarray,
    r_cm: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fisher & Heng (2018) extinction-efficiency model (optics only).

    This is an "optics engine" that returns (Q_ext, Q_sca, g). For now it
    assumes pure absorption (Q_sca = 0, g = 0).

    Parameters
    ----------
    wl : `~jax.numpy.ndarray`, shape (nwl,)
        Wavelength grid in microns.
    r_cm : `~jax.numpy.ndarray`, shape (nr,) or scalar
        Particle radius in cm.
    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary containing:
        - `cld_Q0`, `cld_Q1`, `cld_a`, `cld_sigma` (sigma currently unused here)

    Returns
    -------
    Q_ext : `~jax.numpy.ndarray`, shape (nr, nwl) or (nwl,) if r_cm is scalar
        Extinction efficiency.
    Q_sca : `~jax.numpy.ndarray`, same shape as Q_ext
        Scattering efficiency (zeros).
    g : `~jax.numpy.ndarray`, same shape as Q_ext
        Asymmetry parameter (zeros).
    """
    Q0 = params["cld_Q0"]
    Q1 = params["cld_Q1"]
    a = params["cld_a"]

    # Convert radius to microns to match wl units for size parameter.
    r_um = r_cm * 1e4

    # Broadcast to (nr, nwl) when r_cm is a vector.
    if r_um.ndim == 0:
        x = (2.0 * jnp.pi * r_um) / jnp.maximum(wl, 1e-30)
    else:
        x = (2.0 * jnp.pi * r_um[:, None]) / jnp.maximum(wl[None, :], 1e-30)

    x = jnp.maximum(x, 1e-30)
    Q_ext = Q1 / (Q0 * x ** (-a) + x**0.2)

    Q_sca = jnp.zeros_like(Q_ext)
    g = jnp.zeros_like(Q_ext)
    return Q_ext, Q_sca, g

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
    q_c_lay = jnp.where(q_c_lay > _QC_EPS, q_c_lay, 0.0)
    has_cloud_any = jnp.any(q_c_lay > 0)

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

    def _do_cloud(_):
        # Compute Mie/MADT efficiencies conditionally (skip wavelengths outside node support)
        Q_ext_vals, Q_sca_vals, g_vals = jax.vmap(_compute_mie_or_zero, in_axes=(0, 0, 0, None, 0))(
            wl, n, k, r_eff, wl_support_mask
        )

        # Compute cloud opacity using vertical profile from state
        k_cld = (
            (3.0 * q_c_lay[:, None] * Q_ext_vals[None, :])
            / (4.0 * cld_rho * (r_eff * 1e-4))
        )

        ssa_wl = jnp.clip(Q_sca_vals / jnp.maximum(Q_ext_vals, 1e-30), 0.0, 1.0)
        ssa = ssa_wl[None, :] + jnp.zeros_like(q_c_lay[:, None])
        g = g_vals[None, :] + jnp.zeros_like(q_c_lay[:, None])
        q_mask = (q_c_lay > 0)[:, None]
        k_cld = jnp.where(q_mask, k_cld, 0.0)
        ssa = jnp.where(q_mask, ssa, 0.0)
        g = jnp.where(q_mask, g, 0.0)
        return k_cld, ssa, g

    def _skip_cloud(_):
        zeros = jnp.zeros((state["nlay"], state["nwl"]), dtype=wl.dtype)
        return zeros, zeros, zeros

    return jax.lax.cond(has_cloud_any, _do_cloud, _skip_cloud, operand=None)
