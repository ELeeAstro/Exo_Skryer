"""
vert_chem.py
============
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from .data_constants import amu, kb, bar
from .rate_jax import RateJAX, get_nasa9_cache
from .vert_mu import compute_mu

# Solar reference abundances (relative to H) - Asplund et al. (2021)
solar_H = 1.0
solar_He = 10.0**(10.914-12.0)
solar_O = 10.0**(8.69-12.0)
solar_C = 10.0**(8.46-12.0)
solar_N = 10.0**(7.83-12.0)

solar_H2 = solar_H/2.0
solar_He_H2 = solar_He/solar_H2 

__all__ = [
    "constant_vmr",
    "constant_vmr_clr",
    "build_constant_vmr_kernel",
    "CE_fastchem_jax",
    "CE_rate_jax",
    "quench_approx"
]


def constant_vmr(species_order: tuple[str, ...]):
    """Build a JIT-optimized function for constant VMR profiles.

    This function creates a chemistry kernel that generates constant (vertically
    uniform) volume mixing ratio profiles from logarithmic abundance parameters.
    The returned kernel is optimized for JAX JIT compilation by using a fixed
    species list determined at build time.

    Parameters
    ----------
    species_order : tuple of str
        Ordered tuple of trace species names (e.g., ('H2O', 'CH4', 'CO')).
        For each species, the kernel will expect a parameter named 'log_10_f_<species>'
        in the params dictionary.

    Returns
    -------
    callable
        A chemistry kernel function with signature:
        `kernel(p_lay, T_lay, params, nlay) -> Dict[str, jnp.ndarray]`

        The kernel takes:
        - p_lay : Layer pressures (unused but kept for API compatibility)
        - T_lay : Layer temperatures (unused but kept for API compatibility)
        - params : Dictionary containing 'log_10_f_<species>' values
        - nlay : Number of atmospheric layers

        And returns a dictionary mapping species names to their VMR profiles.
    """
    param_keys = tuple(f"log_10_f_{s}" for s in species_order)

    def _constant_vmr_kernel(p_lay, T_lay, params, nlay):
        del p_lay, T_lay

        # Convert log10 abundances to VMR values
        values = [10.0 ** params[k] for k in param_keys]
        trace = jnp.stack(values, axis=0) if values else jnp.zeros((0,), dtype=jnp.float64)
        background = 1.0 - jnp.sum(trace) if values else 1.0

        # Build VMR dictionary with constant profiles for each species
        vmr = {s: jnp.full((nlay,), trace[i]) for i, s in enumerate(species_order)}
        H2 =  background  / (1.0 +  solar_He_H2)
        vmr["H2"] = jnp.full((nlay,), H2)
        He = H2 * solar_He_H2
        vmr["He"] = jnp.full((nlay,), He)
        return vmr

    return _constant_vmr_kernel


def constant_vmr_clr(species_order: tuple[str, ...], use_log10_vmr: bool = False):
    """Build a JIT-optimized function for constant VMR profiles using
    centered-log-ratio (CLR) parameterization.

    This function creates a chemistry kernel that generates constant (vertically
    uniform) volume mixing ratio profiles from abundance parameters using a
    softmax transform. The filler (H2+He) coordinate is fixed at zero,
    guaranteeing that all VMRs are non-negative and sum to unity.

    The kernel accepts either native CLR parameters (``clr_*``) or traditional
    log10 VMR parameters (``log_10_f_*``). When log10 VMR parameters are used,
    they are converted to CLR coordinates internally before applying softmax,
    which acts as a soft constraint ensuring valid atmospheric composition.

    Parameters
    ----------
    species_order : tuple of str
        Ordered tuple of trace species names (e.g., ('H2O', 'CH4', 'CO')).
    use_log10_vmr : bool, optional
        If True, kernel expects 'log_10_f_<species>' parameters and converts
        them to CLR coordinates internally. If False (default), expects
        'clr_<species>' parameters directly.

    Returns
    -------
    callable
        A chemistry kernel function with signature:
        ``kernel(p_lay, T_lay, params, nlay) -> Dict[str, jnp.ndarray]``

        The kernel takes:
        - p_lay : Layer pressures (unused but kept for API compatibility)
        - T_lay : Layer temperatures (unused but kept for API compatibility)
        - params : Dictionary containing abundance parameters
        - nlay : Number of atmospheric layers

        And returns a dictionary mapping species names to their VMR profiles.
    """
    if use_log10_vmr:
        param_keys = tuple(f"log_10_f_{s}" for s in species_order)
    else:
        param_keys = tuple(f"clr_{s}" for s in species_order)

    n_trace = len(species_order)

    def _constant_vmr_clr_kernel(p_lay, T_lay, params, nlay):
        del p_lay, T_lay

        if n_trace == 0:
            background = 1.0
            vmr = {}
        else:
            if use_log10_vmr:
                # Convert log10(VMR) to CLR coordinates
                log10_vmrs = jnp.array([params[k] for k in param_keys])
                vmrs = 10.0 ** log10_vmrs

                # Compute filler fraction (clamped to avoid log(0))
                filler = jnp.maximum(1.0 - jnp.sum(vmrs), 1e-10)

                # CLR transform: z_i = log(VMR_i / VMR_filler)
                z_vals = jnp.log(vmrs) - jnp.log(filler)
            else:
                # Direct CLR parameters
                z_vals = jnp.array([params[k] for k in param_keys])

            # Numerically stable softmax with z_filler = 0:
            # log_denom = log(1 + sum(exp(z_j))) via logaddexp (softplus)
            log_sum_exp_z = jax.scipy.special.logsumexp(z_vals)
            log_denom = jnp.logaddexp(0.0, log_sum_exp_z)

            # VMR for each trace species: x_i = exp(z_i - log_denom)
            trace = jnp.exp(z_vals - log_denom)

            # Filler fraction: x_filler = exp(-log_denom)
            background = jnp.exp(-log_denom)

            # Build VMR dictionary with constant profiles for each species
            vmr = {s: jnp.full((nlay,), trace[i]) for i, s in enumerate(species_order)}

        # Split filler into H2 and He at solar ratio
        H2 = background / (1.0 + solar_He_H2)
        vmr["H2"] = jnp.full((nlay,), H2)
        He = H2 * solar_He_H2
        vmr["He"] = jnp.full((nlay,), He)
        return vmr

    return _constant_vmr_clr_kernel


def build_constant_vmr_kernel(species_order: tuple[str, ...]):
    """Build a constant-VMR chemistry kernel for an explicit species ordering.

    This is a thin wrapper around `constant_vmr` kept for backwards compatibility
    with older documentation and configs.

    Parameters
    ----------
    species_order : tuple[str, ...]
        Ordered tuple of trace species names.

    Returns
    -------
    kernel : callable
        Chemistry kernel function returning VMR profiles.
    """
    return constant_vmr(species_order)


def CE_fastchem_jax(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """Placeholder for a FastChem-based chemical equilibrium backend (not implemented).

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer pressures. Units are arbitrary as long as consistent with the
        solver backend (in this codebase `p_lay` is typically in dyne cm⁻²).
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer temperatures in Kelvin.
    params : dict[str, `~jax.numpy.ndarray`]
        Chemical abundance parameters (e.g., metallicity, C/O).
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    vmr_lay : dict[str, `~jax.numpy.ndarray`]
        Dictionary mapping species names to VMR profiles with shape (nlay,).
    """
    del p_lay, T_lay, params, nlay
    raise NotImplementedError("CE_fastchem_jax is not implemented yet.")


# Backwards-compat alias (do not export)
chemical_equilibrium = CE_fastchem_jax


def CE_rate_jax(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """Compute chemical equilibrium profiles using the `RateJAX` solver.

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer pressures. In the forward model this is typically in dyne cm⁻² and
        is converted internally to bar for the solver.
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer temperatures in Kelvin.
    params : dict[str, `~jax.numpy.ndarray`]
        Chemical abundance parameters containing:

        - `M/H` : float
            Metallicity relative to solar in dex.
        - `C/O` : float
            Carbon-to-oxygen ratio (dimensionless).
    nlay : int
        Number of atmospheric layers (unused; kept for API compatibility).

    Returns
    -------
    vmr_lay : dict[str, `~jax.numpy.ndarray`]
        Dictionary mapping species names to VMR profiles with shape (nlay,).
    """
    del nlay  # Unused but kept for API compatibility with other vert_chem functions

    # Get cached Gibbs tables (will raise RuntimeError if not loaded)
    thermo = get_nasa9_cache()

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
    rate = RateJAX(thermo=thermo, C=C, N=N, O=O, fHe=solar_He)

    # Solve chemical equilibrium profile
    vmr_lay = rate.solve_profile(T_lay, p_lay / bar)

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
        The mean molecular weight in g mol⁻¹.
    g : float
        The surface gravity in cm/s^2.

    Returns
    -------
    `~jax.numpy.ndarray`
        The mixing timescale in seconds.
    """
    del p_bar
    H = (kb * T_K) / (mu_bar * amu * g)  # Scale height [cm]
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
    """Compute quenched chemical abundance profiles.

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer pressures. In the forward model this is typically in dyne cm⁻².
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer temperatures in Kelvin.
    params : dict[str, `~jax.numpy.ndarray`]
        Chemical abundance parameters containing:

        - `M/H` : float
            Metallicity relative to solar in dex.
        - `C/O` : float
            Carbon-to-oxygen ratio (dimensionless).
        - `Kzz` : float
            Eddy diffusion coefficient in cm² s⁻¹.
        - `log_10_g` : float
            Log₁₀ surface gravity in cm s⁻².
    nlay : int
        Number of atmospheric layers (unused; kept for API compatibility).

    Returns
    -------
    vmr_lay : dict[str, `~jax.numpy.ndarray`]
        Dictionary mapping species names to quenched VMR profiles with shape (nlay,).
    """
    del nlay  # Unused but kept for API compatibility

    # Get cached Gibbs tables (will raise RuntimeError if not loaded)
    thermo = get_nasa9_cache()

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
    rate = RateJAX(thermo=thermo, C=C, N=N, O=O, fHe=solar_He)
    vmr_eq = rate.solve_profile(T_lay, p_lay / bar)

    # Compute mean molecular weight (needed for mixing timescale)
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
            tau_chem = _chemical_timescale(species, T_lay, p_lay / bar)
            vmr_quenched[species] = _apply_quench_single(vmr_eq[species], tau_chem, tau_mix)
        else:
            # Non-quenched species: use equilibrium values
            vmr_quenched[species] = vmr_eq[species]

    return vmr_quenched
