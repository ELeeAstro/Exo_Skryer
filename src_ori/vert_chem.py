"""
vert_chem.py
============

Overview:
    Vertical chemistry profiles for atmospheric models.

Notes:
    Species names in config parameters (after stripping f_ or log_10_f_ prefix)
    must match opacity table species names exactly (including capitalization).

Sections to complete:
    - Usage
    - Key Functions
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from rate_jax import RateJAX, get_gibbs_cache

solar_h2 = 0.5
solar_he = 10.0**(10.914-12.0)
solar_h2_he = solar_h2 + solar_he

# Solar reference abundances (relative to H) - Asplund et al. (2021)
solar_O = 10.0**(8.69-12.0)
solar_C = 10.0**(8.46-12.0)
solar_N = 10.0**(7.83-12.0)


def constant_vmr(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """
    Create constant VMR profiles from parameters.

    Species names are taken directly from parameter keys after stripping prefixes.
    User must ensure these match opacity table species names exactly.
    """
    del p_lay, T_lay  # unused but kept for consistent signature

    vmr: Dict[str, jnp.ndarray] = {}
    for k, v in params.items():
        if k.startswith("log_10_f_"):
            species = k[len("log_10_f_"):]
            vmr[species] = 10.0 ** jnp.asarray(v)
        elif k.startswith("f_"):
            species = k[len("f_"):]
            vmr[species] = jnp.asarray(v)

    trace_values = list(vmr.values())
    if trace_values:
        total_trace_vmr = jnp.sum(jnp.stack(trace_values))
    else:
        total_trace_vmr = jnp.asarray(0.0)
    background_vmr = 1.0 - total_trace_vmr

    vmr["H2"] = background_vmr * solar_h2 / solar_h2_he
    vmr["He"] = background_vmr * solar_he / solar_h2_he

    vmr_lay = {species: jnp.full((nlay,), value) for species, value in vmr.items()}
    return vmr_lay


def chemical_equilibrium(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    del p_lay, T_lay, params, nlay
    raise NotImplementedError("chemical_equilibrium is not implemented yet.")


def CE_rate_jax(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """
    Chemical equilibrium profiles using RateJAX solver with cached Gibbs data.

    Computes thermochemical equilibrium VMR profiles for 12 species
    (H2O, CH4, CO, CO2, NH3, C2H2, C2H4, HCN, N2, H2, H, He) using
    Gibbs free energy minimization.

    Parameters
    ----------
    p_lay : array, shape (nlay,)
        Layer pressures [bar]
    T_lay : array, shape (nlay,)
        Layer temperatures [K]
    params : dict
        Chemical abundance parameters:
        - 'M/H': Metallicity [dex], log10 scale factor relative to solar (default: 0.0)
        - 'C/O': Carbon to oxygen ratio (default: 0.5, solar)
    nlay : int
        Number of atmospheric layers (unused but kept for API compatibility)

    Returns
    -------
    vmr_lay : dict[str, array]
        Dictionary mapping species names to VMR profiles, shape (nlay,)
        Species: H2O, CH4, CO, CO2, NH3, C2H2, C2H4, HCN, N2, H2, H, He

    Raises
    ------
    RuntimeError
        If Gibbs cache has not been initialized with load_gibbs_cache()

    Notes
    -----
    - Requires Gibbs cache to be pre-loaded via load_gibbs_cache()
    - Uses Lodders & Fegley (2002) polynomial fits for chemistry regimes
    - Automatically selects HCO or HCNO chemistry based on conditions
    - Includes H2 dissociation equilibrium at high temperatures
    - GPU-compatible when JIT-compiled
    - Helium fraction is fixed at solar value (fHe = 0.085114)

    Abundance Scaling:
        Solar reference values (relative to H2):
            O_solar = 5.0e-4
            C_solar = 2.5e-4
            N_solar = 1.0e-4
            Solar C/O = 0.5

        Conversion from M/H and C/O:
            O = O_solar × 10^[M/H]
            C = (C/O) × O
            N = N_solar × 10^[M/H]

    Examples
    --------
    >>> # First, load Gibbs cache (once during initialization):
    >>> from rate_jax import load_gibbs_cache
    >>> load_gibbs_cache("JANAF_data/")
    >>>
    >>> # Then use in forward model:
    >>> # Solar metallicity, solar C/O
    >>> params = {'M/H': 0.0, 'C/O': 0.5}
    >>> vmr = CE_rate_jax(p_lay, T_lay, params, nlay)
    >>>
    >>> # Super-solar metallicity, high C/O (carbon-rich)
    >>> params = {'M/H': 0.5, 'C/O': 0.9}
    >>> vmr = CE_rate_jax(p_lay, T_lay, params, nlay)
    """
    del nlay  # Unused but kept for API compatibility with other vert_chem functions

    # Get cached Gibbs tables (will raise RuntimeError if not loaded)
    gibbs = get_gibbs_cache()

    # Extract metallicity and C/O ratio from params (keep as JAX arrays for JIT compatibility)
    metallicity = params.get('M/H', 0.0)  # [dex]
    CO_ratio = params.get('C/O', solar_C/solar_O)  # dimensionless

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


def quench_approx(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    
    # Get cached Gibbs tables (will raise RuntimeError if not loaded)
    gibbs = get_gibbs_cache()

    # Extract metallicity and C/O ratio from params (keep as JAX arrays for JIT compatibility)
    metallicity = params.get('M/H', 0.0)  # [dex]
    CO_ratio = params.get('C/O', solar_C/solar_O)  # dimensionless

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

    # Now solve the quench level for each quenched species - first calculate the chemical timescale of each species
    # For vectorisation reasons, just do the full profile


    Species = [
            "H2O", "CH4", "CO", "CO2", "NH3",
            "C2H2", "C2H4", "HCN", "N2",
            "H2", "H", "He",
    ]

    return vmr_lay
