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
from typing import Dict
import jax.numpy as jnp
from aux_funtions import pchip_1d



def zero_cloud_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]):
    layer_count = int(state["nlay"])
    wavelength_count = int(state["nwl"])
    return jnp.zeros((layer_count, wavelength_count))


def compute_grey_cloud_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]):
    layer_count = int(state["nlay"])
    wavelength_count = int(state["nwl"])
    opacity_value = 10.0**jnp.asarray(params["log_10_k_cld_grey"])
    return jnp.full((layer_count, wavelength_count), opacity_value)


def compute_f18_cloud_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]):
    wavelengths = state["wl"]
    layer_count = int(state["nlay"])
    wavelength_count = int(state["nwl"])
    radius = 10.0**jnp.asarray(params["log_10_cld_r"])
    base_efficiency = jnp.asarray(params["cld_Q0"])
    slope = jnp.asarray(params["cld_a"])
    base_opacity = 10.0**jnp.asarray(params["log_10_cld_k0"])
    size_param = (2.0 * jnp.pi * radius) / wavelengths
    opacity_profile = base_opacity / (base_efficiency * size_param ** (-slope) + size_param ** 0.2)
    return jnp.broadcast_to(opacity_profile, (layer_count, wavelength_count))



def compute_nk_cloud_opacity(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    Compute cloud extinction opacity k_cld(lambda, p) from retrieved
    n(λ), k(λ), particle radius r, and a simple power-law q_c(p).

    Returns
    -------
    k_cld : jnp.ndarray with shape (nlay, nwl)
        Cloud mass extinction coefficient.
    """
    wl = state["wl"]        # (nwl,)
    p_lay = state["p_lay"]  # (nlay,)

    rho_d  = jnp.asarray(params["rho_d"])     # particle material density
    r      = jnp.asarray(params["cld_r"])     # particle radius (scalar or broadcastable)
    n      = jnp.asarray(params["n"])         # refractive index real part (λ)
    node_idx = [0, 1, 2, 3, 4, 5]
    k_nodes = jnp.asarray([params[f"k_node_{i}"] for i in node_idx], dtype=float)
    wl_nodes = jnp.asarray([params[f"wl_node_{i}"] for i in node_idx], dtype=float)
    k = 10.0 ** pchip_1d(wl, wl_nodes, k_nodes)  # interpolate in log-space
    q_c_0  = jnp.asarray(params["q_c_0"])     # base mixing ratio
    alpha  = jnp.asarray(params["alpha"])     # vertical slope
    p_base = jnp.asarray(params["p_base"]) * 1e6  # base pressure [bar → Pa]

    # ------------------------------------------------------------------
    # 1. Size parameter
    # ------------------------------------------------------------------
    # shape: (nwl,) if r scalar, wl (nwl,)
    x = 2.0 * jnp.pi * r / wl

    # ------------------------------------------------------------------
    # 2. Rayleigh Q_ext
    # ------------------------------------------------------------------
    # complex refractive index
    m = n + 1j * k
    m2 = m * m
    alp = (m2 - 1.0) / (m2 + 2.0)

    # this is your higher-order Rayleigh expression
    term = 1.0 + (x**2 / 15.0) * alp * ((m2 * m2 + 27.0 * m2 + 38.0) / (2.0 * m2 + 3.0))
    q_ext_ray = (
        4.0 * x * jnp.imag(alp * term)
        + (8.0 / 3.0) * x**4 * jnp.real(alp * alp)
    )

    # ------------------------------------------------------------------
    # 3. MADT Q_ext (Mitchell / Baran style, as in your Fortran)
    # ------------------------------------------------------------------
    # Avoid singular behaviour at k → 0 by clamping to a small floor.
    k_min = 1e-12
    k_eff = jnp.maximum(k, k_min)

    rho = 2.0 * x * (n - 1.0)
    beta = jnp.arctan(k_eff / (n - 1.0))
    tan_b = jnp.tan(beta)

    # core MADT extinction
    q_ext_madt = (
        2.0
        - 4.0 * jnp.exp(-rho * tan_b) * (jnp.cos(beta) / rho) * jnp.sin(rho - beta)
        - 4.0 * jnp.exp(-rho * tan_b) * (jnp.cos(beta) / rho) ** 2 * jnp.cos(rho - 2.0 * beta)
        + 4.0 * (jnp.cos(beta) / rho) ** 2 * jnp.cos(2.0 * beta)
    )

    # absorption
    z = 4.0 * k_eff * x
    exp_z = jnp.exp(-z)
    q_abs = 1.0 + 2.0 * (exp_z / z) + 2.0 * ((exp_z - 1.0) / (z * z))

    C1 = 0.25 * (1.0 + jnp.exp(-1167.0 * k_eff)) * (1.0 - q_abs)

    eps = 0.25 + 0.61 * (1.0 - jnp.exp(-(8.0 * jnp.pi / 3.0) * k_eff)) ** 2
    C2 = (
        jnp.sqrt(2.0 * eps * (x / jnp.pi))
        * jnp.exp(0.5 - eps * (x / jnp.pi))
        * (0.79393 * n - 0.6069)
    )

    q_abs = (1.0 + C1 + C2) * q_abs

    q_edge = (1.0 - jnp.exp(-0.06 * x)) * x ** (-2.0 / 3.0)

    q_ext_madt = (1.0 + 0.5 * C2) * q_ext_madt + q_edge
    # q_sca_madt = q_ext_madt - q_abs   # available if you need it later

    # ------------------------------------------------------------------
    # 4. Combine Rayleigh and MADT with smootherstep in x between 1 and 3
    # ------------------------------------------------------------------
    t = jnp.clip((x - 1.0) / (3.0 - 1.0), 0.0, 1.0)
    w = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3  # smootherstep

    Q_ext = (1.0 - w) * q_ext_ray + w * q_ext_madt  # (nwl,)

    # ------------------------------------------------------------------
    # 5. q_c(p) vertical profile (power law, no Python loop)
    # ------------------------------------------------------------------
    # Your original logic: q_c = 0 for p > p_base, power-law otherwise.
    q_c = jnp.where(
        p_lay > p_base,
        0.0,
        q_c_0 * (p_lay / p_base) ** alpha,
    )  # (nlay,)

    # ------------------------------------------------------------------
    # 6. Convert to cloud mass extinction k_cld(λ, p)
    # ------------------------------------------------------------------
    # k_cld has shape (nlay, nwl) via outer product
    # k_cld = 3 * q_c / (4π ρ_d r) * Q_ext
    k_cld = (3.0 * q_c[:, None] * Q_ext[None, :]) / (4.0 * jnp.pi * rho_d * r)

    return k_cld
