"""
[TODO: add documentation]
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from .rate_jax import RateJAX, get_gibbs_cache

solar_h2 = 0.5
solar_he = 10.0**(10.914-12.0)
solar_h2_he = solar_h2 + solar_he

# Solar reference abundances (relative to H) - Asplund et al. (2021)
solar_O = 10.0**(8.69-12.0)
solar_C = 10.0**(8.46-12.0)
solar_N = 10.0**(7.83-12.0)

__all__ = [
    "constant_vmr",
    "build_constant_vmr_kernel",
    "chemical_equilibrium",
    "CE_rate_jax",
    "quench_approx"
]


def constant_vmr(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """Generate constant volume mixing ratio profiles from parameters.

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`
        Layer pressures in bars.
    T_lay : `~jax.numpy.ndarray`
        Layer temperatures in Kelvin.
    params : dict
        Chemical abundance parameters.
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    dict
        A dictionary mapping species names to their VMR profiles.
    """
    del p_lay, T_lay  # unused but kept for consistent signature

    vmr: Dict[str, jnp.ndarray] = {}
    for k, v in params.items():
        if k.startswith("log_10_f_"):
            species = k[len("log_10_f_"):]
            # Parameter values are already JAX arrays, no need to wrap
            vmr[species] = 10.0 ** v

    trace_values = list(vmr.values())
    if trace_values:
        total_trace_vmr = jnp.sum(jnp.stack(trace_values))
    else:
        total_trace_vmr = 0.0
    background_vmr = 1.0 - total_trace_vmr

    vmr["H2"] = background_vmr * solar_h2 / solar_h2_he
    vmr["He"] = background_vmr * solar_he / solar_h2_he

    vmr_lay = {species: jnp.full((nlay,), value) for species, value in vmr.items()}
    return vmr_lay


def build_constant_vmr_kernel(species_order: tuple[str, ...]):
    """Builds a jitted function for constant VMR profiles.

    Parameters
    ----------
    species_order : tuple
        The order of species.

    Returns
    -------
    function
        A jitted function for constant VMR profiles.
    """
    param_keys = tuple(f"log_10_f_{s}" for s in species_order)

    def _constant_vmr_fixed(p_lay, T_lay, params, nlay):
        del p_lay, T_lay

        # Parameter values are already JAX arrays, no need to wrap
        values = [10.0 ** params[k] for k in param_keys]
        trace = jnp.stack(values, axis=0) if values else jnp.zeros((0,), dtype=jnp.float32)
        background = 1.0 - jnp.sum(trace) if values else 1.0

        vmr = {s: jnp.full((nlay,), trace[i]) for i, s in enumerate(species_order)}
        vmr["H2"] = jnp.full((nlay,), background * (solar_h2 / solar_h2_he))
        vmr["He"] = jnp.full((nlay,), background * (solar_he / solar_h2_he))
        return vmr

    return _constant_vmr_fixed


def chemical_equilibrium(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """Placeholder for general chemical equilibrium calculation.

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`
        Layer pressures in bars.
    T_lay : `~jax.numpy.ndarray`
        Layer temperatures in Kelvin.
    params : dict
        Chemical abundance parameters.
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    dict
        A dictionary mapping species names to their VMR profiles.
    """
    del p_lay, T_lay, params, nlay
    raise NotImplementedError("chemical_equilibrium is not implemented yet.")


def CE_rate_jax(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """Computes chemical equilibrium profiles using RateJAX solver.

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`
        Layer pressures in bars.
    T_lay : `~jax.numpy.ndarray`
        Layer temperatures in Kelvin.
    params : dict
        Chemical abundance parameters.
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    dict
        A dictionary mapping species names to their VMR profiles.
    """
    del nlay  # Unused but kept for API compatibility with other vert_chem functions

    # Get cached Gibbs tables (will raise RuntimeError if not loaded)
    gibbs = get_gibbs_cache()

    # Extract metallicity and C/O ratio from params (keep as JAX arrays for JIT compatibility)
    metallicity = params['M/H']  # [dex]
    CO_ratio = params['C/O']  # dimensionless

    # Convert M/H and C/O to elemental abundances
    # Scale oxygen and nitrogen by metallicity
    O = solar_O * (10.0 ** metallicity)
    N = solar_N * (10.0 ** metallicity)

    # Carbon set by C/O ratio
    C = CO_ratio * O

    # Create RateJAX solver
    rate = RateJAX(gibbs=gibbs, C=C, N=N, O=O, fHe=solar_he)

    # Solve chemical equilibrium profile
    vmr_lay = rate.solve_profile(T_lay, p_lay/1e6)

    return vmr_lay


def _chemical_timescale(species: str, T_K: jnp.ndarray, p_bar: jnp.ndarray) -> jnp.ndarray:
    """Computes chemical timescale for quenched species.

    Parameters
    ----------
    species : str
        The species name.
    T_K : `~jax.numpy.ndarray`
        Layer temperatures in Kelvin.
    p_bar : `~jax.numpy.ndarray`
        Layer pressures in bars.

    Returns
    -------
    `~jax.numpy.ndarray`
        The chemical timescale in seconds.
    """
    if species == "CO" or species == "CH4":
        # CO and CH4 use same timescale (coupled via CO + 3H2 <-> CH4 + H2O)
        # m = metallicity factor (default 3.0 for ~solar, could be parameterized)
        m = 3.0
        tq1 = 1.5e-6 * (p_bar ** -1.0) * (m ** -0.7) * jnp.exp(42000.0 / T_K)
        tq2 = 40.0 * (p_bar ** -2.0) * jnp.exp(25000.0 / T_K)
        return 1.0 / (1.0 / tq1 + 1.0 / tq2)

    if species == "NH3":
        return 1.0e-7 * (p_bar ** -1.0) * jnp.exp(52000.0 / T_K)

    if species == "HCN":
        m = 3.0
        return 1.5e-4 * (p_bar ** -1.0) * (m ** -0.7) * jnp.exp(36000.0 / T_K)

    if species == "CO2":
        return 1.0e-10 * (p_bar ** -0.5) * jnp.exp(38000.0 / T_K)

    # Non-quenched species: return zeros
    return jnp.zeros_like(T_K)


def _mixing_timescale(
    T_K: jnp.ndarray,
    p_bar: jnp.ndarray,
    Kzz: jnp.ndarray,
    mu_bar: jnp.ndarray,
    g: float,
) -> jnp.ndarray:
    """Computes eddy mixing timescale.

    Parameters
    ----------
    T_K : `~jax.numpy.ndarray`
        Layer temperatures in Kelvin.
    p_bar : `~jax.numpy.ndarray`
        Layer pressures in bars.
    Kzz : `~jax.numpy.ndarray`
        The eddy diffusion coefficient in cm^2/s.
    mu_bar : `~jax.numpy.ndarray`
        The mean molecular weight in amu.
    g : float
        The surface gravity in cm/s^2.

    Returns
    -------
    `~jax.numpy.ndarray`
        The mixing timescale in seconds.
    """
    k_B = 1.380649e-16   # Boltzmann constant [erg/K]
    m_H = 1.6735575e-24  # Hydrogen mass [g]

    mu = mu_bar * m_H  # Mean molecular mass [g]
    H = (k_B * T_K) / (mu * g)  # Scale height [cm]
    tau_mix = (H ** 2) / jnp.maximum(Kzz, 1e-30)  # Avoid division by zero [s]

    return tau_mix


def _apply_quench_single(
    vmr_eq: jnp.ndarray,
    tau_chem: jnp.ndarray,
    tau_mix: jnp.ndarray,
) -> jnp.ndarray:
    """Applies quenching to a single species profile.

    Parameters
    ----------
    vmr_eq : `~jax.numpy.ndarray`
        The equilibrium VMR profile.
    tau_chem : `~jax.numpy.ndarray`
        The chemical timescale in seconds.
    tau_mix : `~jax.numpy.ndarray`
        The mixing timescale in seconds.

    Returns
    -------
    `~jax.numpy.ndarray`
        The quenched VMR profile.
    """
    # Quench where chemistry is slower than mixing
    quench_mask = tau_chem > tau_mix

    # Find first quenched level (returns JAX array, not Python int)
    quench_idx = jnp.argmax(quench_mask)

    # Check if any quenching occurs
    has_quench = jnp.any(quench_mask)

    # Build quenched profile: freeze VMR at quench_idx for layers >= quench_idx
    layer_indices = jnp.arange(vmr_eq.size)
    vmr_frozen = jnp.where(
        layer_indices >= quench_idx,
        vmr_eq[quench_idx],  # Freeze at quench level value
        vmr_eq,              # Below quench level: use equilibrium
    )

    # Return equilibrium if no quenching, otherwise return frozen profile
    return jnp.where(has_quench, vmr_frozen, vmr_eq)


def quench_approx(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """Computes quenched chemical abundance profiles.

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`
        Layer pressures in bars.
    T_lay : `~jax.numpy.ndarray`
        Layer temperatures in Kelvin.
    params : dict
        Chemical abundance parameters.
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    dict
        A dictionary mapping species names to their quenched VMR profiles.
    """
    del nlay  # Unused but kept for API compatibility

    # Get cached Gibbs tables (will raise RuntimeError if not loaded)
    gibbs = get_gibbs_cache()

    # Extract metallicity and C/O ratio from params
    metallicity = params['M/H']  # [dex]
    CO_ratio = params['C/O']  # dimensionless

    Kzz = params['Kzz']  # Eddy diffusion coefficient [cm²/s]
    g = 10.0**params['log_10_g']  # Surface gravity [cm/s²]

    # Convert M/H and C/O to elemental abundances
    O = solar_O * (10.0 ** metallicity)
    N = solar_N * (10.0 ** metallicity)
    C = CO_ratio * O

    # Create RateJAX solver and compute chemical equilibrium
    rate = RateJAX(gibbs=gibbs, C=C, N=N, O=O, fHe=solar_he)
    vmr_eq = rate.solve_profile(T_lay, p_lay / 1e6)

    # Compute mean molecular weight (needed for mixing timescale)
    from vert_mu import compute_mu
    mu_bar = compute_mu(vmr_eq)

    # Compute mixing timescale (same for all species)
    tau_mix = _mixing_timescale(T_lay, p_lay, Kzz, mu_bar, g)

    # Apply quenching to relevant species
    # Species that undergo quenching: CO, CH4, NH3, HCN, CO2
    # Non-quenched species: H2O, C2H2, C2H4, N2, H2, H, He
    quenched_species = ["CO", "CH4", "NH3", "HCN", "CO2"]

    vmr_quenched = {}
    for species in vmr_eq.keys():
        if species in quenched_species:
            # Compute chemical timescale and apply quenching
            tau_chem = _chemical_timescale(species, T_lay, p_lay)
            vmr_quenched[species] = _apply_quench_single(vmr_eq[species], tau_chem, tau_mix)
        else:
            # Non-quenched species: use equilibrium values
            vmr_quenched[species] = vmr_eq[species]

    return vmr_quenched
