"""
vert_chem.py
============
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import equinox as eqx

from .data_constants import amu, kb, bar, R
from .rate_jax import RateJAX, get_nasa9_cache
from .chem_easychem_short_jax import (
    EasyChemShortModel,
    solve_profile_scan,
    solve_profile_vmap,
)
from .chem_fastchem_grid_jax import (
    FastChemGridModel,
    load_fastchem_grid,
    resolve_species_indices,
    interpolate_profile_scan as interpolate_fc_profile_scan,
    interpolate_profile_vmap as interpolate_fc_profile_vmap,
)
from .vert_mu import compute_mu


# Solar reference abundances (relative to H) - Asplund et al. (2021)
solar_H = 1.0
solar_He = 10.0 ** (10.914 - 12.0)
solar_N  = 10.0 ** (7.83  - 12.0)
solar_C  = 10.0 ** (8.46  - 12.0)
solar_O  = 10.0 ** (8.69  - 12.0)
solar_Na  = 10.0 ** (6.22  - 12.0)
solar_Si  = 10.0 ** (7.51  - 12.0)
solar_S  = 10.0 ** (7.12  - 12.0)
solar_Cl  = 10.0 ** (7.12  - 12.0)
solar_K  = 10.0 ** (5.07  - 12.0)
solar_Fe  = 10.0 ** (7.46  - 12.0)

solar_H2 = solar_H/2.0
solar_He_H2 = solar_He/solar_H2 

__all__ = [
    "constant_vmr",
    "constant_vmr_clr",
    "build_constant_vmr_kernel",
    "CE_fastchem_jax",
    "CE_fastchem_grid_jax",
    "CE_rate_jax",
    "CE_easychem_jax",
    "quench_approx",
    "CE_atmodeller",
    "load_element_potentials_cache",
    "is_element_potentials_cache_loaded",
    "load_atmodeller_cache",
    "is_atmodeller_cache_loaded",
    "load_fastchem_grid_cache",
    "is_fastchem_grid_cache_loaded",
    "get_fastchem_grid_cache_info",
]

# ---------------------------------------------------------------------------
# Global atmodeller cache (mirrors _NASA9_CACHE pattern in rate_jax.py)
# ---------------------------------------------------------------------------
_ATMODELLER_MODEL = None          # EquilibriumModel instance (built once at init)
_ATMODELLER_GAS_KEYS: tuple = ()  # atmodeller output keys, e.g. ('H2O_g', 'CH4_g', ...)
_ATMODELLER_SPECIES: tuple = ()   # bare retrieval names, e.g. ('H2O', 'CH4', ...)
_ATMODELLER_ELEM_KEYS: tuple = ()        # sorted element symbols, e.g. ('C', 'H', 'He', ...)
_ATMODELLER_ELEM_MASSES: np.ndarray = None  # atomic masses [g/mol] in same order as ELEM_KEYS

# ---------------------------------------------------------------------------
# Global element-potentials cache
# ---------------------------------------------------------------------------
_EP_MODEL: EasyChemShortModel | None = None
_EP_SPECIES: tuple[str, ...] = ()
_EP_ELEMENTS: tuple[str, ...] = ()
_EP_SOLVER_CFG: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Global FastChem-grid cache
# ---------------------------------------------------------------------------
_FC_GRID_MODEL: FastChemGridModel | None = None
_FC_GRID_SPECIES_OUT: tuple[str, ...] = ()
_FC_GRID_SPECIES_IDX: jnp.ndarray | None = None
_FC_GRID_SOLVER_MODE: str = "vmap"
_FC_GRID_UNMAPPED: tuple[str, ...] = ()


def load_fastchem_grid_cache(
    grid_path: str,
    species_out: list[str] | tuple[str, ...],
    solver_cfg: dict | None,
    species_map_override: dict[str, str] | None = None,
) -> None:
    """Load and cache a FastChem 5D interpolation grid."""
    global _FC_GRID_MODEL, _FC_GRID_SPECIES_OUT, _FC_GRID_SPECIES_IDX, _FC_GRID_SOLVER_MODE, _FC_GRID_UNMAPPED

    model = load_fastchem_grid(grid_path)
    resolved, missing = resolve_species_indices(model, species_out, species_map_override=species_map_override)
    if not resolved:
        raise ValueError(
            "No requested opacity species could be mapped to FastChem grid species. "
            f"Requested={list(species_out)}"
        )

    mode = str((solver_cfg or {}).get("mode", "vmap")).lower()
    if mode not in ("scan", "vmap"):
        raise ValueError("fastchem_grid_jax.solver.mode must be one of: scan, vmap")

    _FC_GRID_MODEL = model
    _FC_GRID_SPECIES_OUT = tuple(resolved.keys())
    _FC_GRID_SPECIES_IDX = jnp.asarray([resolved[s] for s in _FC_GRID_SPECIES_OUT], dtype=jnp.int32)
    _FC_GRID_SOLVER_MODE = mode
    _FC_GRID_UNMAPPED = tuple(missing)


def is_fastchem_grid_cache_loaded() -> bool:
    """Return True if FastChem grid interpolation cache is initialised."""
    return _FC_GRID_MODEL is not None


def get_fastchem_grid_cache_info() -> dict[str, Any]:
    """Return lightweight diagnostics for the cached FastChem grid backend."""
    if _FC_GRID_MODEL is None:
        return {}
    return {
        "species_out": _FC_GRID_SPECIES_OUT,
        "unmapped_species": _FC_GRID_UNMAPPED,
        "solver_mode": _FC_GRID_SOLVER_MODE,
        "use_log_axes": bool(_FC_GRID_MODEL.use_log_axes),
        "shape": tuple(_FC_GRID_MODEL.mixing_ratios.shape),
        "T_range": (float(_FC_GRID_MODEL.temperature[0]), float(_FC_GRID_MODEL.temperature[-1])),
        "P_range": (float(_FC_GRID_MODEL.pressure[0]), float(_FC_GRID_MODEL.pressure[-1])),
        "MH_range": (float(_FC_GRID_MODEL.M_H[0]), float(_FC_GRID_MODEL.M_H[-1])),
        "CO_range": (float(_FC_GRID_MODEL.C_O[0]), float(_FC_GRID_MODEL.C_O[-1])),
    }


def load_element_potentials_cache(
    species_list: list[str],
    elements: list[str] | tuple[str, ...] | None,
    nlay: int,
    solver_kwargs: dict | None,
    nasa9_dir: str,
    *,
    p0_bar: float = 1.0,
    e_ref: str = "H",
) -> None:
    """Build and cache the production CE model for retrieval use."""
    del nlay  # Reserved for API symmetry with atmodeller init.
    global _EP_MODEL, _EP_SPECIES, _EP_ELEMENTS, _EP_SOLVER_CFG

    if not species_list:
        raise ValueError("easychem_jax.species must be a non-empty list.")

    element_seq = tuple(elements) if elements else None
    if element_seq is not None and len(element_seq) == 0:
        element_seq = None

    if element_seq is not None:
        unsupported = [e for e in element_seq if e not in {"H", "He", "C", "N", "O", "S", "Na", "K", "Si"}]
        if unsupported:
            raise ValueError(
                "Unsupported elements for easychem_jax budgets: "
                f"{unsupported}. Supported: H, He, C, N, O, S, Na, K, Si."
            )

    if e_ref and element_seq is not None and e_ref not in element_seq:
        raise ValueError(f"easychem_jax.e_ref='{e_ref}' is not in elements list {element_seq}.")

    b_seed = {e: 1.0 for e in (element_seq if element_seq is not None else ("H", "He", "C", "N", "O"))}
    _EP_MODEL = EasyChemShortModel.from_nasa9_dir(
        nasa9_dir,
        species=tuple(species_list),
        elements=element_seq,
        element_budgets=b_seed,
        P0_bar=float(p0_bar),
    )
    _EP_SPECIES = tuple(_EP_MODEL.species)
    _EP_ELEMENTS = tuple(_EP_MODEL.elements)

    solver_defaults = {
        "mode": "scan",
        "max_steps": 64,
        "tol": 1.0e-11,
        "throw": True,
        "relax_limit": 0.75,
    }
    _EP_SOLVER_CFG = {**solver_defaults, **(solver_kwargs or {})}
    _EP_SOLVER_CFG["mode"] = str(_EP_SOLVER_CFG.get("mode", "scan")).lower()
    if _EP_SOLVER_CFG["mode"] not in ("scan", "vmap"):
        raise ValueError("easychem_jax.solver.mode must be one of: scan, vmap")


def is_element_potentials_cache_loaded() -> bool:
    """Return True if the global element-potentials cache is initialised."""
    return _EP_MODEL is not None


def _element_budgets_from_params(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Build element budget vector matching `_EP_ELEMENTS` from M_to_H and C_to_O."""
    metallicity = params["M_to_H"]
    co_ratio = params["C_to_O"]

    zscale = 10.0 ** metallicity
    O = solar_O * zscale
    C = co_ratio * O

    vals = []
    for e in _EP_ELEMENTS:
        if e == "H":
            vals.append(jnp.asarray(solar_H, dtype=jnp.float64))
        elif e == "He":
            vals.append(jnp.asarray(solar_He, dtype=jnp.float64))
        elif e == "C":
            vals.append(C)
        elif e == "O":
            vals.append(O)
        elif e == "N":
            vals.append(solar_N * zscale)
        elif e == "S":
            vals.append(solar_S * zscale)
        elif e == "Na":
            vals.append(solar_Na * zscale)
        elif e == "K":
            vals.append(solar_K * zscale)
        elif e == "Si":
            vals.append(solar_Si * zscale)
        else:
            raise ValueError(
                f"Element {e!r} has no budget rule in CE_easychem_jax. "
                "Supported: H, He, C, N, O, S, Na, K, Si."
            )
    return jnp.asarray(vals, dtype=jnp.float64)

# Common-name → Hill-notation mapping for species whose database key differs from the
# conventional name.  atmodeller stores species under Hill notation; the output dict
# uses Hill-notation keys.  Entries here let the YAML use readable names (e.g. NH3_g)
# while _solve looks up the correct Hill-notation key (H3N_g) in the output.
_COMMON_TO_HILL: dict[str, str] = {
    "NH3": "H3N",   # no carbon: H before N alphabetically
    "HCN": "CHN",   # carbon first, then H, then N
}

def load_atmodeller_cache(
    species_list: list[str],
    nlay: int,
    solver_kwargs: dict | None = None,
) -> None:
    """Build an :class:`atmodeller.EquilibriumModel` from *species_list* and cache it globally.

    Parameters
    ----------
    species_list : list of str
        Species strings in atmodeller notation, e.g. ``['H2O_g', 'CH4_g', 'H2_g', 'He_g']``.
        All gas-phase entries (suffix ``_g``) are registered as retrievable VMR outputs.
    nlay : int
        Number of atmospheric layers. Sets the batch size of the cached
        :class:`~atmodeller.containers.Parameters` pytree so that ``eqx.tree_at``
        updates during retrieval match the correct shape.
    solver_kwargs : dict, optional
        Keyword arguments forwarded to :class:`~atmodeller.containers.SolverParameters`.
        Supported keys: ``atol``, ``rtol``, ``max_steps``, ``multistart``,
        ``multistart_perturbation``, ``jac``.  Defaults (atmodeller's own) are used
        for any key not supplied.  Reduce ``multistart`` and loosen ``atol``/``rtol``
        for faster GPU throughput.
    """
    global _ATMODELLER_MODEL, _ATMODELLER_STATE, _ATMODELLER_SOLVER, _ATMODELLER_SPECIES_NETWORK, _ATMODELLER_GAS_KEYS, _ATMODELLER_SPECIES, _ATMODELLER_ELEM_KEYS, _ATMODELLER_ELEM_MASSES
    try:
        from atmodeller import SpeciesNetwork, ThermodynamicState
        from atmodeller.containers import Parameters, SolverParameters
        from atmodeller.solvers import make_independent_solver
        from molmass import Formula
    except ImportError as exc:
        raise ImportError(
            "The 'atmodeller' chemistry backend is optional and is not installed. "
            "Install the optional dependencies for the atmodeller backend, then retry."
        ) from exc

    _ATMODELLER_SPECIES_NETWORK = SpeciesNetwork.create(tuple(species_list))

    T_dum = np.full(nlay, 1000.0)
    p_dum = np.ones(nlay)

    _ATMODELLER_STATE = ThermodynamicState(temperature=T_dum, pressure=p_dum, melt_fraction=0)

    mole_fractions = {
        "H":  solar_H,
        "He": solar_He,
        "C": solar_C,
        "N": solar_N,
        "O": solar_O,
        "Na": solar_Na,
        "K":  solar_K,
    }

    _ATMODELLER_ELEM_KEYS = tuple(sorted(mole_fractions.keys()))
    _ATMODELLER_ELEM_MASSES = np.array([Formula(k).mass for k in _ATMODELLER_ELEM_KEYS])

    mass_constraints = {k: mole_fractions[k] * Formula(k).mass for k in _ATMODELLER_ELEM_KEYS}

    solver_params = SolverParameters(**(solver_kwargs or {}))

    _ATMODELLER_MODEL = Parameters.create(
        _ATMODELLER_SPECIES_NETWORK,
        _ATMODELLER_STATE,
        mass_constraints=mass_constraints,
        solver_parameters=solver_params,
    )
    # _ATMODELLER_SPECIES: common names used as VMR dict keys (e.g. 'NH3', 'HCN')
    # _ATMODELLER_GAS_KEYS: Hill-notation keys used to look up atmodeller output (e.g. 'H3N_g', 'CHN_g')
    _ATMODELLER_SPECIES = tuple(s.removesuffix("_g") for s in species_list if s.endswith("_g"))
    _ATMODELLER_GAS_KEYS = tuple(
        _COMMON_TO_HILL.get(bare, bare) + "_g"
        for bare in _ATMODELLER_SPECIES
    )

    _ATMODELLER_SOLVER = make_independent_solver(_ATMODELLER_MODEL)

    # print(_ATMODELLER_MODEL)
    # print(_ATMODELLER_GAS_KEYS)
    # print(_ATMODELLER_SPECIES)
    # print(_ATMODELLER_SOLVER)
    # quit()


def is_atmodeller_cache_loaded() -> bool:
    """Return True if the global atmodeller cache has been initialised."""
    return _ATMODELLER_MODEL is not None


def get_atmodeller_cache():
    """Return ``(model, gas_keys, species_names)``; raises :exc:`RuntimeError` if not loaded."""
    if _ATMODELLER_MODEL is None:
        raise RuntimeError(
            "Atmodeller cache not loaded. Call load_atmodeller_cache() before using CE_atmodeller."
        )
    return _ATMODELLER_MODEL, _ATMODELLER_GAS_KEYS, _ATMODELLER_SPECIES


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
        # Optional atomic-hydrogen split: retrieve log10(H/H2) and solve for
        # H2, He, H such that:
        #   H/H2 = r,  He/H = solar_He (by H nuclei), and H2 + He + H = background
        #
        # This keeps the total hydrogen budget (H nuclei) consistent while moving
        # hydrogen between H2 and H, rather than implicitly changing He/H when H is added.
        if "log_10_H_over_H2" in params:
            r = 10.0 ** params["log_10_H_over_H2"]
        else:
            r = jnp.asarray(0.0, dtype=background.dtype)
        r = jnp.maximum(r, 0.0)

        # Let N_H be the (dimensionless) abundance of hydrogen nuclei in the filler.
        # Then:
        #   H2 = N_H/(2+r),  H = r*H2,  He = solar_He*N_H,
        # and enforce H2+H+He = background to solve for N_H.
        denom = solar_He + (1.0 + r) / (2.0 + r)
        N_H = background / denom
        H2 = N_H / (2.0 + r)
        H = r * H2
        He = solar_He * N_H
        vmr["H2"] = jnp.full((nlay,), H2)
        vmr["He"] = jnp.full((nlay,), He)
        if "log_10_H_over_H2" in params:
            vmr["H"] = jnp.full((nlay,), H)
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

        # Split filler into H2/He, optionally with atomic H using retrieved H/H2 ratio.
        # Enforce He/H = solar_He (by H nuclei) when H is present.
        if "log_10_H_over_H2" in params:
            r = 10.0 ** params["log_10_H_over_H2"]
        else:
            r = jnp.asarray(0.0, dtype=background.dtype)
        r = jnp.maximum(r, 0.0)

        denom = solar_He + (1.0 + r) / (2.0 + r)
        N_H = background / denom
        H2 = N_H / (2.0 + r)
        H = r * H2
        He = solar_He * N_H
        vmr["H2"] = jnp.full((nlay,), H2)
        vmr["He"] = jnp.full((nlay,), He)
        if "log_10_H_over_H2" in params:
            vmr["H"] = jnp.full((nlay,), H)
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


def CE_fastchem_grid_jax(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """Interpolate FastChem 5D grid over (T, P, M/H, C/O)."""
    del nlay  # Kept for API compatibility.
    if _FC_GRID_MODEL is None or _FC_GRID_SPECIES_IDX is None:
        raise RuntimeError(
            "FastChem-grid cache not loaded. "
            "Call load_fastchem_grid_cache() before using CE_fastchem_grid_jax."
        )

    metallicity = params["M_to_H"]
    co_ratio = params["C_to_O"]
    p_lay_bar = p_lay / bar

    if _FC_GRID_SOLVER_MODE == "scan":
        vmr_matrix, mmw_lay = interpolate_fc_profile_scan(
            _FC_GRID_MODEL, T_lay, p_lay_bar, metallicity, co_ratio, _FC_GRID_SPECIES_IDX
        )
    else:
        vmr_matrix, mmw_lay = interpolate_fc_profile_vmap(
            _FC_GRID_MODEL, T_lay, p_lay_bar, metallicity, co_ratio, _FC_GRID_SPECIES_IDX
        )

    vmr_matrix = jnp.clip(vmr_matrix, 0.0, jnp.inf)

    out = {sp: vmr_matrix[:, i] for i, sp in enumerate(_FC_GRID_SPECIES_OUT)}
    out["__mu_lay__"] = jnp.clip(mmw_lay, 1e-30, jnp.inf)
    return out


def CE_fastchem_jax(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """Compatibility alias for FastChem-grid interpolation backend."""
    return CE_fastchem_grid_jax(p_lay, T_lay, params, nlay)


# Backwards-compat alias (do not export)
chemical_equilibrium = CE_fastchem_grid_jax


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

        - `M_to_H` : float
            Metallicity relative to solar in dex.
        - `C_to_O` : float
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

    # Extract metallicity and C_to_O ratio from params (keep as JAX arrays for JIT compatibility)
    metallicity = params['M_to_H']  # [dex]
    CO_ratio = params['C_to_O']  # dimensionless

    # Convert M_to_H and C_to_O to elemental abundances
    # Scale oxygen and nitrogen by metallicity
    O = solar_O * (10.0 ** metallicity)
    N = solar_N * (10.0 ** metallicity)

    # Carbon set by C/O ratio
    C = CO_ratio * O

    # Create RateJAX solver
    rate = RateJAX(thermo=thermo, C=C, N=N, O=O, fHe=solar_He)

    # Solve chemical equilibrium profile
    vmr_lay = rate.solve_profile(T_lay, p_lay / bar)

    # Scale Na and K by metallicity, add as constant profiles, then renormalise
    vmr_Na = solar_Na * (10.0 ** metallicity)
    vmr_K  = solar_K  * (10.0 ** metallicity)
    n_lay = T_lay.shape[0]
    vmr_lay['Na'] = jnp.full((n_lay,), vmr_Na)
    vmr_lay['K']  = jnp.full((n_lay,), vmr_K)
    total = jnp.sum(jnp.stack(list(vmr_lay.values())), axis=0)  # (nlay,)
    vmr_lay = {sp: v / total for sp, v in vmr_lay.items()}

    return vmr_lay


def CE_easychem_jax(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """Compute equilibrium profiles using the production CE JAX backend."""
    del nlay  # Kept for API compatibility.
    if _EP_MODEL is None:
        raise RuntimeError(
            "EasyChem cache not loaded. "
            "Call load_element_potentials_cache() before using CE_easychem_jax."
        )

    b = _element_budgets_from_params(params)
    inp = replace(_EP_MODEL.inputs, b=b)

    p_bar = p_lay / bar
    mode = str(_EP_SOLVER_CFG.get("mode", "scan")).lower()
    max_steps = int(_EP_SOLVER_CFG.get("max_steps", 64))
    tol = float(_EP_SOLVER_CFG.get("tol", 1.0e-11))
    throw = bool(_EP_SOLVER_CFG.get("throw", False))
    relax_limit = float(_EP_SOLVER_CFG.get("relax_limit", 0.75))

    if mode == "vmap":
        _packed_prof, y_prof, _n_prof, result_prof = solve_profile_vmap(
            T_lay, p_bar, inp, max_steps=max_steps, tol=tol, relax_limit=relax_limit
        )
    else:
        _packed_prof, y_prof, _n_prof, result_prof = solve_profile_scan(
            T_lay, p_bar, inp, state_init=None, max_steps=max_steps, tol=tol, relax_limit=relax_limit
        )

    failed = result_prof != 0
    if bool(jnp.any(failed)):
        n_failed = int(jnp.sum(failed))
        if throw:
            raise RuntimeError(
                "EasyChem SHORT CE solve failed to converge for "
                f"{n_failed}/{int(result_prof.shape[0])} layers."
            )

    y_prof = jnp.clip(y_prof, 0.0, jnp.inf)
    y_sum = jnp.sum(y_prof, axis=1, keepdims=True)
    vmr_arr = y_prof / jnp.maximum(y_sum, 1e-300)
    vmr_arr = jnp.where(failed[:, None], jnp.nan, vmr_arr)
    return {sp: vmr_arr[:, i] for i, sp in enumerate(_EP_SPECIES)}




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

        - `M_to_H` : float
            Metallicity relative to solar in dex.
        - `C_to_O` : float
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

    # Extract metallicity and C_to_O ratio from params
    metallicity = params['M_to_H']  # [dex]
    CO_ratio = params['C_to_O']  # dimensionless

    Kzz = params['Kzz']  # Eddy diffusion coefficient [cm²/s]
    g = 10.0**params['log_10_g']  # Surface gravity [cm/s²]

    # Convert M_to_H and C_to_O to elemental abundances
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


def CE_atmodeller(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """Compute chemical equilibrium profiles using the :mod:`atmodeller` backend.

    Parameters
    ----------
    p_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer pressures. In the forward model this is typically in dyne cm⁻² and
        is converted internally to bar for the solver.
    T_lay : `~jax.numpy.ndarray`, shape (nlay,)
        Layer temperatures in Kelvin.
    params : dict[str, `~jax.numpy.ndarray`]
        Chemical abundance parameters containing:

        - ``M_to_H`` : float
            Metallicity relative to solar in dex.
        - ``C_to_O`` : float
            Carbon-to-oxygen ratio (dimensionless).
    nlay : int
        Number of atmospheric layers. Used to broadcast mass constraints and
        initial guess to shape ``(nlay, ...)``.

    Returns
    -------
    vmr_lay : dict[str, `~jax.numpy.ndarray`]
        Dictionary mapping bare species names (e.g. ``'H2O'``, ``'CH4'``) to
        VMR profiles with shape ``(nlay,)``.

    Notes
    -----
    H and He abundances are **fixed at solar values** and are not scaled by
    metallicity. All other elements present in the species network (C, N, O,
    Na, K) are scaled by ``10 ** M_to_H``; carbon is further set by
    ``C_to_O * O``.
    """

    _, _, species_names = get_atmodeller_cache()

    nsp = len(species_names)

    metallicity = params['M_to_H']  # [dex]
    CO_ratio = params['C_to_O']  # dimensionless

    O = solar_O * (10.0 ** metallicity)
    N = solar_N * (10.0 ** metallicity)
    Na = solar_Na * (10.0 ** metallicity)
    K = solar_K * (10.0 ** metallicity)
    C = CO_ratio * O                          # C/O sets carbon

    mole_fractions = {
        "H":  solar_H,
        "He": solar_He,
        "C": C,
        "N": N,
        "O": O,
        "Na": Na,
        "K":  K,
    }

    # update temperature in pytree
    parameter_update = eqx.tree_at(
        lambda p: p.state.temperature,_ATMODELLER_MODEL,T_lay
    )

    # update pressure in pytree
    p_lay_bar = p_lay/bar
    parameter_update = eqx.tree_at(
        lambda p: p.state.pressure,parameter_update,p_lay_bar
    )

    # update mass constraints in pytree — shape must be (1, n_elements)
    # atmodeller transposes this to (n_elements, 1) internally; shape[0]=1 keeps
    # vmap_axes_spec treating it as non-batched (same constraints for all layers)
    mf_vals = jnp.stack([mole_fractions[k] for k in _ATMODELLER_ELEM_KEYS])
    mass_constraints = (mf_vals * _ATMODELLER_ELEM_MASSES)[None, :]  # (1, n_elements)
    parameter_update = eqx.tree_at(
        lambda p: p.mass_constraints.abundance, parameter_update, mass_constraints
    )

    # init (initial guess values) - units ln moles (absolute value) 
    init = jnp.ones(2 * nsp)
    init = jnp.broadcast_to(init, (nlay,2*nsp))

    result = _ATMODELLER_SOLVER(init*50.0,parameter_update)

    solution = result.value
    solution = jnp.split(solution,2,axis=-1)[0]

    logtotal = logsumexp(solution,axis=-1, keepdims=True)

    vmrs = jnp.exp(solution-logtotal)

    return {species_names[i]: vmrs[:, i] for i in range(nsp)}
