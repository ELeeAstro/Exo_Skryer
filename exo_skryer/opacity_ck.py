"""
opacity_ck.py
=============
"""

from typing import Dict

import jax
import jax.numpy as jnp
from jax import lax

from . import build_opacities as XS
from .data_constants import amu, bar
from .ck_mix_RORR import mix_k_tables_rorr
from .ck_mix_PRAS import mix_k_tables_pras
from .aux_functions import interp_table_axis, t_interp_scheme, interp_dtype

# exp(_LN10 * x) is several times faster than 10.0 ** x on these large arrays.
_LN10 = 2.302585092994046

__all__ = [
    "zero_ck_opacity",
    "compute_ck_opacity",
    "compute_ck_opacity_perspecies",
    "ck_vmr_perspecies",
    "make_ck_species_kappa",
]


def _interp_species_log(sigma_4d, log_temp_grid, p_idx, p_weight, log_t, t_interp="linear", dtype=None):
    """Interpolate one species' (nT, nP, nwl, ng) log10 k-table to one layer.

    Linear in log P (the layer grid is fixed), then along log T with
    ``t_interp`` ("linear" or "pchip"), clipped to the species' own T range.
    ``dtype`` is the arithmetic dtype (see ``interp_dtype``).
    """
    if dtype is not None:
        p_weight = p_weight.astype(dtype)

    def _at_t(t_idx):
        return (1.0 - p_weight) * sigma_4d[t_idx, p_idx, :, :] + p_weight * sigma_4d[t_idx, p_idx + 1, :, :]

    return interp_table_axis(log_t, log_temp_grid, _at_t, t_interp, dtype=dtype)


def _interpolate_sigma_log(layer_pressures_bar: jnp.ndarray, layer_temperatures: jnp.ndarray,
                           t_interp: str = "linear") -> jnp.ndarray:
    """Bilinear interpolation of correlated-k cross-sections on (log P, log T) grids.

    This function retrieves pre-loaded correlated-k opacity tables from the opacity
    registry and interpolates them to the specified atmospheric layer conditions using
    bilinear interpolation in log₁₀(P)-log₁₀(T) space. The interpolation is performed
    separately for each species and returns cross-sections still in log₁₀ space.

    Parameters
    ----------
    layer_pressures_bar : `~jax.numpy.ndarray`, shape (nlay,)
        Atmospheric layer pressures in bar.
    layer_temperatures : `~jax.numpy.ndarray`, shape (nlay,)
        Atmospheric layer temperatures in Kelvin.

    Returns
    -------
    sigma_interp : `~jax.numpy.ndarray`, shape (nspecies, nlay, nwl, ng)
        Interpolated cross-sections in log₁₀ space with units of log₁₀(cm² molecule⁻¹).
        The axes represent:
        - nspecies: Number of absorbing species
        - nlay: Number of atmospheric layers
        - nwl: Number of wavelength bins
        - ng: Number of g-points per wavelength bin
    """
    # NOTE: This helper pulls tables from the global registry. If used inside a large
    # jitted forward model, it may increase compile-time constants. Prefer
    # compute_ck_opacity() which supports passing tables via `state`.
    sigma_cube = XS.ck_sigma_cube()
    log_p_grid = XS.ck_log10_pressure_grid()
    log_temperature_grids = XS.ck_log10_temperature_grids()

    log_p_layers = jnp.log10(layer_pressures_bar)
    log_t_layers = jnp.log10(layer_temperatures)

    def _interp_one_layer(log_p: jnp.ndarray, log_t: jnp.ndarray) -> jnp.ndarray:
        # Pressure bracket indices and weights (shared across species)
        p_idx = jnp.searchsorted(log_p_grid, log_p) - 1
        p_idx = jnp.clip(p_idx, 0, log_p_grid.shape[0] - 2)
        p_weight = (log_p - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
        p_weight = jnp.clip(p_weight, 0.0, 1.0)

        def _interp_one_species(sigma_4d: jnp.ndarray, log_temp_grid: jnp.ndarray) -> jnp.ndarray:
            return _interp_species_log(sigma_4d, log_temp_grid, p_idx, p_weight, log_t, t_interp)

        sigma_log_layer = jax.vmap(_interp_one_species)(sigma_cube, log_temperature_grids)  # (nspec, nwl, ng)
        return sigma_log_layer

    # NOTE: This returns a large (nspec, nlay, nwl, ng) array; callers that care about peak
    # memory should perform layer-wise interpolation + mixing instead of using this helper.
    return jax.vmap(_interp_one_layer)(log_p_layers, log_t_layers)

def _get_ck_quadrature(opac: Dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract g-points and quadrature weights for correlated-k integration.

    This function retrieves the g-points and their associated quadrature weights
    used for integrating over the k-distribution. The g-points represent cumulative
    probability values in [0, 1], and the weights are typically Gaussian quadrature
    weights that sum to 1.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        State dictionary that may contain pre-loaded 'g_weights' array.
        If not present, weights are retrieved from the opacity registry.

    Returns
    -------
    g_points : `~jax.numpy.ndarray`, shape (ng,)
        Cumulative probability points where k-distribution is sampled, in [0, 1].
    weights : `~jax.numpy.ndarray`, shape (ng,)
        Quadrature weights for numerical integration over g-space. Sum to 1.0.
    """
    return opac["g_points"], opac["g_weights"]


def zero_ck_opacity(state: Dict[str, jnp.ndarray], opac: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return a zero correlated-k opacity array.

    This function is used as a fallback when correlated-k opacities are disabled
    in the configuration. It maintains API compatibility with `compute_ck_opacity()`
    so the forward model can seamlessly switch between CK enabled/disabled.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        State dictionary containing:

        - `p_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer pressures (used only to determine array size).
        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid (used only to determine array size).

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; kept for API compatibility).

    Returns
    -------
    zeros : `~jax.numpy.ndarray`, shape (nlay, nwl, ng)
        Zero-valued correlated-k opacity array in cm² g⁻¹.
    """
    layer_pressures = state["p_lay"]
    wavelengths = state["wl"]
    layer_count = layer_pressures.shape[0]
    wavelength_count = wavelengths.shape[0]

    # Get number of g-points from opac cache.
    g_weights = opac["g_weights"]
    n_g = g_weights.shape[-1]

    return jnp.zeros((layer_count, wavelength_count, n_g))


def compute_ck_opacity(state: Dict[str, jnp.ndarray], opac: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute correlated-k opacity with multi-species mixing.

    This function calculates the total atmospheric opacity using the correlated-k
    approximation. It performs the following steps:
    1. Interpolates pre-loaded k-tables to atmospheric (P, T) conditions
    2. Mixes k-distributions from multiple species using RORR or PRAS scheme
    3. Converts from cross-section (cm² molecule⁻¹) to mass opacity (cm² g⁻¹)

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state dictionary containing:

        - `p_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer pressures in dyne cm⁻².
        - `T_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Layer temperatures in Kelvin.
        - `mu_lay` : `~jax.numpy.ndarray`, shape (nlay,)
            Mean molecular weight per layer in amu.
        - `vmr_lay` : dict[str, `~jax.numpy.ndarray`]
            Volume mixing ratios for each species. Keys must match species
            names in the loaded CK tables. Values can be scalars or arrays
            with shape (nlay,).
        - `wl` : `~jax.numpy.ndarray`, shape (nwl,)
            Wavelength grid in microns.
        - `ck_mix` : str or int, optional
            Mixing method selector. Either 'RORR' (default, code=1) or 'PRAS'
            (code=2). Can be specified as string or integer code.
        - `g_weights` : `~jax.numpy.ndarray`, optional
            Quadrature weights for g-point integration. If not provided,
            retrieved from opacity registry.

    params : dict[str, `~jax.numpy.ndarray`]
        Parameter dictionary (unused; VMRs come from state['vmr_lay']).
        Kept for API compatibility with other opacity functions.

    Returns
    -------
    kappa_ck : `~jax.numpy.ndarray`, shape (nlay, nwl, ng)
        Total atmospheric mass opacity in cm² g⁻¹ at each layer, wavelength
        bin, and g-point.
    """
    layer_pressures = state["p_lay"]
    layer_temperatures = state["T_lay"]
    layer_mu = state["mu_lay"]
    layer_count = layer_pressures.shape[0]

    # Get species names and mixing ratios
    species_names = XS.ck_runtime_species_order()
    layer_vmr = state["vmr_lay"]

    # Direct lookup - species names must match VMR keys exactly
    # VMR values are already JAX arrays, no need to wrap
    mixing_ratios = jnp.stack(
        [jnp.broadcast_to(layer_vmr[name], (layer_count,)) for name in species_names],
        axis=0,
    )

    g_points, g_weights = _get_ck_quadrature(opac)

    # Get mixing method from state (default to RORR).
    # Backwards-compatible: accept either a string ("RORR"/"PRAS") or an int code.
    ck_mix_raw = state.get("ck_mix", 1)
    if isinstance(ck_mix_raw, str):
        ck_mix_code = 2 if ck_mix_raw.upper() == "PRAS" else 1
    else:
        # Avoid int() conversion for JIT compatibility
        ck_mix_code = ck_mix_raw

    # Layer-wise interpolation + mixing to avoid materializing (n_species, n_layers, n_wl, n_g).
    sigma_cube = opac["ck_sigma_cube"]
    log_p_grid = opac["ck_log10_pressure_grid"]
    log_temperature_grids = opac["ck_log10_temperature_grids"]
    t_interp = t_interp_scheme(state)
    cdt = interp_dtype(state)

    log_p_layers = jnp.log10(layer_pressures / bar)
    log_t_layers = jnp.log10(layer_temperatures)

    n_species = sigma_cube.shape[0]
    n_wl = sigma_cube.shape[-2]
    n_g = sigma_cube.shape[-1]

    def _interp_sigma_log_layer(log_p: jnp.ndarray, log_t: jnp.ndarray) -> jnp.ndarray:
        p_idx = jnp.searchsorted(log_p_grid, log_p) - 1
        p_idx = jnp.clip(p_idx, 0, log_p_grid.shape[0] - 2)
        p_weight = (log_p - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx])
        p_weight = jnp.clip(p_weight, 0.0, 1.0)

        def _interp_one_species(sigma_4d: jnp.ndarray, log_temp_grid: jnp.ndarray) -> jnp.ndarray:
            return _interp_species_log(sigma_4d, log_temp_grid, p_idx, p_weight, log_t, t_interp, cdt)

        return jax.vmap(_interp_one_species)(sigma_cube, log_temperature_grids)

    # One layer at a time: lax.map stacks each layer's output (no full-size carry,
    # which reverse-mode AD would store at every layer), and jax.checkpoint keeps
    # only the layer inputs for the backward pass, recomputing the interpolation
    # and mixing (incl. the RORR sorts) on the way back.
    @jax.checkpoint
    def _mix_one_layer(x) -> jnp.ndarray:
        layer_idx, log_p, log_t, vmr_layer = x
        sigma_log_layer = _interp_sigma_log_layer(log_p, log_t)  # (nspec, nwl, ng)

        if ck_mix_code == 2:
            mixed = mix_k_tables_pras(
                sigma_log_layer[:, None, :, :],
                vmr_layer[:, None],
                g_points,
                g_weights,
                layer_offset=layer_idx,
            )[0]
        else:
            mixed = mix_k_tables_rorr(
                jnp.exp(_LN10 * sigma_log_layer[:, None, :, :].astype(jnp.float64)),
                vmr_layer[:, None],
                g_points,
                g_weights,
            )[0]
        return mixed

    mixed_sigma = lax.map(
        _mix_one_layer,
        (jnp.arange(layer_count), log_p_layers, log_t_layers, mixing_ratios.T),
    )  # (nlay, nwl, ng)

    # Convert to mass opacity (cm^2 / g)
    total_opacity = mixed_sigma / (layer_mu[:, None, None] * amu)

    return total_opacity


def compute_ck_opacity_perspecies(
    state: Dict[str, jnp.ndarray],
    opac: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-species correlated-k mass opacities, all species at once (diagnostics).

    The forward model does not use this: with ``ck_mix: TRANS`` the RT
    interpolates one species at a time (:func:`make_ck_species_kappa`), so the
    (n_species, nlay, nwl, ng) array is never held. This builds that array from
    the same interpolation, e.g. for per-species component spectra.

    Parameters
    ----------
    state : dict[str, `~jax.numpy.ndarray`]
        Atmospheric state with `p_lay` (dyne cm⁻²), `T_lay` (K), `mu_lay` (amu)
        and `vmr_lay` (dict of volume mixing ratios).
    opac : dict[str, `~jax.numpy.ndarray`]
        Runtime k-tables (`ck_sigma_cube`, `ck_log10_pressure_grid`,
        `ck_log10_temperature_grids`).
    params : dict[str, `~jax.numpy.ndarray`]
        Unused; kept for API compatibility.

    Returns
    -------
    sigma_perspecies : `~jax.numpy.ndarray`, shape (n_species, nlay, nwl, ng)
        Per-species mass opacities in cm² g⁻¹, NOT weighted by VMR.
    vmr_perspecies : `~jax.numpy.ndarray`, shape (n_species, nlay)
        Volume mixing ratios for each species at each layer.
    """
    del params
    vmr_perspecies = ck_vmr_perspecies(state)
    kappa_of = make_ck_species_kappa(state, opac)
    n_species = vmr_perspecies.shape[0]
    sigma_perspecies = jnp.stack([kappa_of(s) for s in range(n_species)], axis=0)
    return sigma_perspecies, vmr_perspecies


def ck_vmr_perspecies(state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Volume mixing ratios of the ck species, (n_species, nlay), in table order."""
    layer_count = state["p_lay"].shape[0]
    layer_vmr = state["vmr_lay"]
    return jnp.stack(
        [jnp.broadcast_to(layer_vmr[name], (layer_count,)) for name in XS.ck_runtime_species_order()],
        axis=0,
    )


def make_ck_species_kappa(state: Dict[str, jnp.ndarray], opac: Dict[str, jnp.ndarray]):
    """Per-species correlated-k mass opacity on demand (``ck_mix: TRANS``).

    Returns ``kappa_of(spec_idx) -> (nlay, nwl, ng)`` in cm^2 g^-1 for a traced
    ``spec_idx``. Only one species is held at a time, so the RT can interpolate
    inside its species loop instead of keeping (n_species, nlay, nwl, ng).
    """
    sigma_cube = opac["ck_sigma_cube"]
    log_p_grid = opac["ck_log10_pressure_grid"]
    log_temperature_grids = opac["ck_log10_temperature_grids"]
    t_interp = t_interp_scheme(state)
    cdt = interp_dtype(state)

    log_p_layers = jnp.log10(state["p_lay"] / bar)
    log_t_layers = jnp.log10(state["T_lay"])
    mass = state["mu_lay"] * amu

    # Pressure bracket per layer (species independent)
    p_idx = jnp.clip(jnp.searchsorted(log_p_grid, log_p_layers) - 1, 0, log_p_grid.shape[0] - 2)
    p_weight = jnp.clip(
        (log_p_layers - log_p_grid[p_idx]) / (log_p_grid[p_idx + 1] - log_p_grid[p_idx]), 0.0, 1.0
    )

    def kappa_of(spec_idx):
        sigma_4d = lax.dynamic_index_in_dim(sigma_cube, spec_idx, keepdims=False)
        log_temp_grid = lax.dynamic_index_in_dim(log_temperature_grids, spec_idx, keepdims=False)
        sigma_log = jax.vmap(
            lambda pi, pw, lt: _interp_species_log(sigma_4d, log_temp_grid, pi, pw, lt, t_interp, cdt)
        )(p_idx, p_weight, log_t_layers)                       # (nlay, nwl, ng)
        return jnp.exp(_LN10 * sigma_log.astype(jnp.float64)) / mass[:, None, None]

    return kappa_of
