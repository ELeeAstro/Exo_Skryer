"""
opacity_cia.py
==================
"""

from typing import Dict

import jax.numpy as jnp
from jax import vmap

from . import build_opacities as XS

__all__ = [
    "compute_cia_opacity",
    "zero_cia_opacity"
]


def zero_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return a zero CIA opacity array.

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary containing scalar entries `nlay` (number of layers)
        and `nwl` (number of wavelengths).
    params : dict[str, jnp.ndarray]
        Unused; kept for API compatibility.

    Returns
    -------
    `~jax.numpy.ndarray`
        Array of zeros with shape `(nlay, nwl)`.
    """
    # Use shape directly without jnp.size() for JIT compatibility
    shape = (state["nlay"], state["nwl"])
    return jnp.zeros(shape)


def _interpolate_sigma(layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """Interpolate CIA cross-sections to layer temperatures.

    Parameters
    ----------
    layer_temperatures : `~jax.numpy.ndarray`, shape `(nlay,)`
        Layer temperatures in K.

    Returns
    -------
    `~jax.numpy.ndarray`
        CIA cross-sections in linear space with shape `(nspecies, nlay, nwl)`.

        The cross-sections are expected to be in units of
        `cm^5 molecule^-2` (or an equivalent pair-absorption unit) so that
        `nd_lay**2 * sigma / rho_lay` yields a mass opacity in `cm^2 g^-1`.
    """
    sigma_cube = XS.cia_sigma_cube()
    temperature_grids = XS.cia_temperature_grids()

    # Convert to log10 space for interpolation
    log_t_layers = jnp.log10(layer_temperatures)

    def _interp_one_species(sigma_2d, temp_grid):
        """Interpolate log10 cross-sections for a single CIA species.

        Parameters
        ----------
        sigma_2d : `~jax.numpy.ndarray`
            Log10 CIA cross-sections with shape `(nT, nwl)`.
        temp_grid : `~jax.numpy.ndarray`
            Temperature grid (K) with shape `(nT,)`.

        Returns
        -------
        jnp.ndarray
            Log10 cross-sections interpolated to `layer_temperatures` with shape
            `(nlay, nwl)`.
        """
        # sigma_2d: (n_temp, n_wavelength)
        # temp_grid: (n_temp,)

        # Convert temperature grid to log space
        log_t_grid = jnp.log10(temp_grid)

        # Find temperature bracket indices and weights in log space
        t_idx = jnp.searchsorted(log_t_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, log_t_grid.shape[0] - 2)
        t_weight = (log_t_layers - log_t_grid[t_idx]) / (log_t_grid[t_idx + 1] - log_t_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)

        # Get lower and upper temperature brackets
        # Indexing: sigma_2d[temp, wavelength]
        s_t0 = sigma_2d[t_idx, :]          # shape: (n_layers, n_wavelength)
        s_t1 = sigma_2d[t_idx + 1, :]

        # Linear interpolation in temperature
        s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1

        # Set cross sections to very small value below minimum temperature
        min_temp = temp_grid[0]
        below_min = layer_temperatures < min_temp
        tiny = jnp.array(-199.0, dtype=s_interp.dtype)
        s_interp = jnp.where(below_min[:, None], tiny, s_interp)

        return s_interp

    # Vectorize over all CIA species
    sigma_log = vmap(_interp_one_species)(sigma_cube, temperature_grids)
    return 10.0 ** sigma_log


def _compute_pair_weight(
    name: str,
    layer_count: int,
    layer_vmr: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Compute a per-layer CIA pair weight from the VMR mapping.

    Parameters
    ----------
    name : str
        CIA species name in `A-B` form (e.g., `H2-He`).
    layer_count : int-like
        Number of atmospheric layers (`nlay`).
    layer_vmr : dict[str, jnp.ndarray]
        Mapping from species name to VMR values (scalar or `(nlay,)`).

    Returns
    -------
    `~jax.numpy.ndarray`
        Per-layer pair weight with shape `(nlay,)`, equal to
        `VMR[A] * VMR[B]` after broadcasting any scalar VMRs to `(nlay,)`.
    """
    name_clean = name.strip()

    # Normal CIA pair: "H2-He" -> product of H2 and He VMRs
    parts = name_clean.split("-")
    if len(parts) != 2:
        raise ValueError(f"CIA species '{name}' must be in 'A-B' format")

    species_a, species_b = parts[0], parts[1]
    # VMR values are already JAX arrays, no need to wrap
    ratio_a = layer_vmr[species_a]
    ratio_b = layer_vmr[species_b]
    ratio_a = jnp.broadcast_to(ratio_a, (layer_count,))
    ratio_b = jnp.broadcast_to(ratio_b, (layer_count,))
    return ratio_a * ratio_b


def compute_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute CIA mass opacity.

    Parameters
    ----------
    state : dict[str, jnp.ndarray]
        State dictionary with at least:

        - `nlay` : int-like
            Number of layers (Python int or a 0-d JAX array).
        - `nwl` : int-like
            Number of wavelengths (Python int or a 0-d JAX array). Only used to
            size the output when no CIA species are enabled.
        - `wl` : `~jax.numpy.ndarray`, shape `(nwl,)`
            Forward-model wavelength grid (typically microns).
        - `T_lay` : `~jax.numpy.ndarray`, shape `(nlay,)`
            Layer temperatures in K.
        - `nd_lay` : `~jax.numpy.ndarray`, shape `(nlay,)`
            Layer number density in `molecule cm^-3`.
        - `rho_lay` : `~jax.numpy.ndarray`, shape `(nlay,)`
            Layer mass density in `g cm^-3`.
        - `vmr_lay` : Mapping[str, Any]
            Mapping from species name to VMR values (scalar or `(nlay,)`).
    params : dict[str, jnp.ndarray]
        Unused; kept for API compatibility.

    Returns
    -------
    `~jax.numpy.ndarray`
        CIA opacity in cm^2 g^-1 with shape `(nlay, nwl)`.
    """
    # Use JAX array directly without int() for JIT compatibility
    layer_count = state["nlay"]
    wavelengths = state["wl"]
    layer_temperatures = state["T_lay"]
    number_density = state["nd_lay"]   # (nlay,)
    density = state["rho_lay"]         # (nlay,)
    layer_vmr = state["vmr_lay"]

    master_wavelength = XS.cia_master_wavelength()
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("CIA wavelength grid must match the forward-model master grid.")

    sigma_values = _interpolate_sigma(layer_temperatures)  # (nspecies, nlay, nwl)
    species_names = XS.cia_species_names()
    keep_indices = [i for i, name in enumerate(species_names) if name.strip() != "H-"]
    if not keep_indices:
        return zero_cia_opacity(state, params)

    # Compute pair weights for each CIA species (string ops happen once at trace time)
    pair_weights = jnp.stack(
        [_compute_pair_weight(species_names[i], layer_count, layer_vmr) for i in keep_indices],
        axis=0,
    )  # (nspecies_keep, nlay)

    normalization = pair_weights * (number_density**2 / density)[None, :]  # (nspecies_keep, nlay)
    sigma_keep = sigma_values[keep_indices, :, :]  # (nspecies_keep, nlay, nwl)
    return jnp.sum(normalization[:, :, None] * sigma_keep, axis=0)  # (nlay, nwl)
