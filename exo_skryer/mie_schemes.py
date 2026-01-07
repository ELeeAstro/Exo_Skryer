"""
mie_schemes.py
==============

Modular implementations of Mie scattering approximations and exact solutions.
All schemes take the same inputs: real refractive index n, imaginary refractive
index k, and size parameter x.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from functools import partial

__all__ = [
    "rayleigh",
    "madt",
    "lxmie",
]


def rayleigh(n: jnp.ndarray, k: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute extinction and scattering efficiencies using Rayleigh approximation.

    Valid for small particles (x << 1). Uses the polarizability approximation
    with first-order size-parameter corrections.

    Parameters
    ----------
    n : `~jax.numpy.ndarray`
        Real part of the refractive index.
    k : `~jax.numpy.ndarray`
        Imaginary part of the refractive index.
    x : `~jax.numpy.ndarray`
        Size parameter (x = 2πr/λ).

    Returns
    -------
    Q_ext : `~jax.numpy.ndarray`
        Extinction efficiency.
    Q_sca : `~jax.numpy.ndarray`
        Scattering efficiency.
    g : `~jax.numpy.ndarray`
        Asymmetry parameter (scattering anisotropy).
    """

    # Complex refractive index and polarizability
    m = n + 1j * k
    m2 = m * m
    alp = (m2 - 1.0) / (m2 + 2.0)

    # Rayleigh regime with first-order corrections
    term = 1.0 + (x**2 / 15.0) * alp * ((m2 * m2 + 27.0 * m2 + 38.0) / (2.0 * m2 + 3.0))
    Q_abs_ray = 4.0 * x * jnp.imag(alp * term)
    Q_sca_ray = (8.0 / 3.0) * x**4 * jnp.real(alp**2)
    Q_ext_ray = Q_abs_ray + Q_sca_ray

    # Asymmetry parameter: zero for Rayleigh (isotropic scattering
    g_ray = 0.0

    return Q_ext_ray, Q_sca_ray, g_ray


def madt(n: jnp.ndarray, k: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute extinction and scattering efficiencies using Modified Anomalous Diffraction Theory (MADT.

    Most for larger particles (x >= 1) and soft particles (n ~< 3).

    Parameters
    ----------
    n : `~jax.numpy.ndarray`
        Real part of the refractive index.
    k : `~jax.numpy.ndarray`
        Imaginary part of the refractive index.
    x : `~jax.numpy.ndarray`
        Size parameter (x = 2πr/λ).

    Returns
    -------
    Q_ext : `~jax.numpy.ndarray`
        Extinction efficiency.
    Q_sca : `~jax.numpy.ndarray`
        Scattering efficiency.
    g : `~jax.numpy.ndarray`
        Asymmetry parameter (scattering anisotropy).
    """

    # MADT regime setup
    k_min = 1e-12
    k_eff = jnp.maximum(k, k_min)

    dn = n - 1.0
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
        * (0.79393 * n - 0.6069)
    )

    Q_abs_madt = (1.0 + C1 + C2) * Q_abs_madt

    Q_edge = (1.0 - jnp.exp(-0.06 * x)) * x ** (-2.0 / 3.0)
    Q_ext_madt = (1.0 + 0.5 * C2) * Q_ext_madt + Q_edge
    Q_sca_madt = Q_ext_madt - Q_abs_madt

    # Asymmetry parameter from Rayleigh formula (valid for small-to-moderate x)
    numerator = (
        -2.0 * k**6
        + k**4 * (13.0 - 2.0 * n**2)
        + k**2 * (2.0 * n**4 + 2.0 * n**2 - 27.0)
        + 2.0 * n**6 + 13.0 * n**4 + 27.0 * n**2 + 18.0
    )
    denominator = 15.0 * (
        4.0 * k**4
        + 4.0 * k**2 * (2.0 * n**2 - 3.0)
        + (2.0 * n**2 + 3.0)**2
    )
    Cm = numerator / jnp.maximum(denominator, 1e-30)

    # Limit to 0.9 to capture constant region
    g_madt = jnp.minimum(Cm * x**2, 0.9)

    return Q_ext_madt, Q_sca_madt, g_madt


def lxmie(n: jnp.ndarray, k: jnp.ndarray, x: jnp.ndarray,
          nmax: int = 4096, cf_max_terms: int = 4096, cf_eps: float = 1e-10) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute extinction and scattering efficiencies using exact Mie theory.

    Valid for all size parameters. Uses the full Lorenz-Mie solution with
    continued fractions for numerical stability (Kitzmann et al. 2018).

    Parameters
    ----------
    n : `~jax.numpy.ndarray`
        Real part of the refractive index.
    k : `~jax.numpy.ndarray`
        Imaginary part of the refractive index.
    x : `~jax.numpy.ndarray`
        Size parameter (x = 2πr/λ).
    nmax : int, optional
        Maximum number of Mie coefficients to compute (default: 4096).
    cf_max_terms : int, optional
        Maximum number of continued fraction terms (default: 4096).
    cf_eps : float, optional
        Convergence tolerance for continued fractions (default: 1e-10).

    Returns
    -------
    Q_ext : `~jax.numpy.ndarray`
        Extinction efficiency.
    Q_sca : `~jax.numpy.ndarray`
        Scattering efficiency.
    g : `~jax.numpy.ndarray`
        Asymmetry parameter (scattering anisotropy).

    """
    # Import the full Mie solver
    from .lxmie_mod import lxmie_jax

    # Construct complex refractive index
    m = n - 1j * k

    # Call the full Mie solver
    Q_ext, Q_sca, Q_abs, g = lxmie_jax(m, x, nmax=nmax, cf_max_terms=cf_max_terms, cf_eps=cf_eps)

    return Q_ext, Q_sca, g
