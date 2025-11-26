"""
registry_line.py
================

Overview:
    

    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional

import jax.numpy as jnp
import numpy as np
import h5py

# Dataclass for each of the line opacity tables
@dataclass(frozen=True)
class LineRegistryEntry:
    name: str
    idx: int
    pressures: jnp.ndarray
    temperatures: jnp.ndarray
    wavelengths: jnp.ndarray
    cross_sections: jnp.ndarray


# Global registries and caches for forward model
_LINE_ENTRIES: Tuple[LineRegistryEntry, ...] = ()
_LINE_SIGMA_CACHE: jnp.ndarray | None = None
_LINE_TEMPERATURE_CACHE: jnp.ndarray | None = None

# Clear all the cache entries
def _clear_cache():
    line_species_names.cache_clear()
    line_master_wavelength.cache_clear()
    line_pressure_grid.cache_clear()
    line_temperature_grid.cache_clear()
    line_temperature_grids.cache_clear()
    line_sigma_cube.cache_clear()
    line_pick_arrays.cache_clear()

# Reset all the global registries
def reset_registry():
    global _LINE_ENTRIES, _LINE_SIGMA_CACHE, _LINE_TEMPERATURE_CACHE
    _LINE_ENTRIES = ()
    _LINE_SIGMA_CACHE = None
    _LINE_TEMPERATURE_CACHE = None
    _clear_cache()

# Check if the registries are set or not
def has_line_data() -> bool:
    return bool(_LINE_ENTRIES)

# Function to load TauREx HDF5 opacity data
def _load_line_h5(index: int, path: str, target_wavelengths: np.ndarray) -> LineRegistryEntry:
    """
    Load TauREx HDF5 format opacity tables.

    TauREx format:
    - mol_name: molecule name (string)
    - p: pressure grid in bar (nP,)
    - t: temperature grid in K (nT,)
    - bin_edges: wavenumber bin edges in cm^-1 (nwl+1,)
    - xsecarr: cross sections in cm^2/molecule (nP, nT, nwl)

    Returns data in registry format:
    - pressures in bar (nP,)
    - temperatures in K (nT,)
    - wavelengths in microns (target_wavelengths,)
    - cross_sections in log10(cm^2) (nT, nP, target_wavelengths)
    """

    with h5py.File(path, 'r') as f:
        # Read molecule name
        name = f["mol_name"][0]
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        name = str(name)

        # Read grids
        pressures = np.asarray(f["p"][:], dtype=float)  # bar
        temperatures = np.asarray(f["t"][:], dtype=float)  # K
        bin_edges = np.asarray(f["bin_edges"][:], dtype=float)  # cm^-1
        native_xs = np.asarray(f["xsecarr"][:], dtype=float)  # (nP, nT, nwl) cm^2/molecule

    # Dimensions
    n_pressures = pressures.size
    n_temperatures = temperatures.size

    # Convert wavenumber bin edges to wavelength bin centers
    # λ[μm] = 10000 / ν[cm^-1]
    # Bin centers from edges
    wavenumber_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    native_wavelengths = 10000.0 / wavenumber_centers  # μm

    # Sort wavelengths (wavenumbers are descending, so wavelengths will be ascending after conversion)
    sort_idx = np.argsort(native_wavelengths)
    native_wavelengths = native_wavelengths[sort_idx]

    # Transpose from (nP, nT, nwl) to (nT, nP, nwl) and apply wavelength sorting
    native_xs_transposed = np.transpose(native_xs, (1, 0, 2))[:, :, sort_idx]

    # Convert to log10 (handle zeros by setting minimum value)
    # Use maximum to avoid log10(0) warning
    min_xs = 1e-99  # corresponds to log10 = -99
    native_xs_log = np.log10(np.maximum(native_xs_transposed, min_xs))

    # Dimensions of the master wavelength
    wavelength_count = target_wavelengths.size

    # Interpolate to target wavelength grid
    xs_interp = np.empty((n_temperatures, n_pressures, wavelength_count), dtype=float)
    for iT in range(n_temperatures):
        for iP in range(n_pressures):
            xs_interp[iT, iP, :] = np.interp(
                target_wavelengths,
                native_wavelengths,
                native_xs_log[iT, iP, :],
                left=-99.0,
                right=-99.0
            )

    # Return a dataclass with all the required entries
    return LineRegistryEntry(
        name=name,
        idx=index,
        pressures=jnp.asarray(pressures),
        temperatures=jnp.asarray(temperatures),
        wavelengths=jnp.asarray(target_wavelengths),
        cross_sections=jnp.asarray(xs_interp),
    )

# Pad the tables to a rectangle (in dimension) - usually only in T as wavelength and pressure grids are the same lengths
def _rectangularize_entries(entries: List[LineRegistryEntry]) -> Tuple[LineRegistryEntry, ...]:

    # Return if zero lbl table
    if not entries:
        return ()
    
    # Find the wavelength and pressure grid from the first tables (should be the same across all species)
    base_wavelengths = entries[0].wavelengths
    base_pressures = entries[0].pressures
    expected_wavelengths = base_wavelengths.shape[0]
    for entry in entries[1:]:
        if entry.wavelengths.shape != base_wavelengths.shape or not np.allclose(entry.wavelengths, base_wavelengths):
            raise ValueError(f"Line opacity wavelength grids differ between {entries[0].name} and {entry.name}.")
        if entry.pressures.shape != base_pressures.shape or not np.allclose(entry.pressures, base_pressures):
            raise ValueError(f"Line opacity pressure grids differ between {entries[0].name} and {entry.name}.")
        
    # Find the max number of pressure points    
    max_pressures = max(entry.pressures.shape[0] for entry in entries)
    # Find the max number of temperature points
    max_temperatures = max(entry.temperatures.shape[0] for entry in entries)
    # Start a new list for the padded cross section tables and pad the temperature arrays with the extra dimensions
    padded_entries: List[LineRegistryEntry] = []
    for entry in entries:
        pressures = jnp.asarray(entry.pressures)
        temperatures = jnp.asarray(entry.temperatures)
        xs = jnp.asarray(entry.cross_sections)
        current_temperatures, current_pressures, wavelength_count = xs.shape
        if wavelength_count != expected_wavelengths:
            raise ValueError(f"Species {entry.name} has λ grid length {wavelength_count}, expected {expected_wavelengths}.")
        if current_pressures != max_pressures:
            raise ValueError(f"Species {entry.name} has nP={current_pressures}, expected {max_pressures} for common grid.")
        pad_temperatures = max_temperatures - current_temperatures
        if pad_temperatures > 0:
            temperatures = jnp.pad(temperatures, (0, pad_temperatures), mode="edge")
            xs = jnp.pad(xs, ((0, pad_temperatures), (0, 0), (0, 0)), mode="edge")
        padded_entries.append(
            LineRegistryEntry(
                name=entry.name,
                idx=entry.idx,
                pressures=pressures,
                temperatures=temperatures,
                wavelengths=base_wavelengths,
                cross_sections=xs,
            )
        )
    return tuple(padded_entries)


# Read in and prepare the line data
def load_line_registry(cfg, obs, lam_master: Optional[np.ndarray] = None):

    # Allocate the global scope caches
    global _LINE_ENTRIES, _LINE_SIGMA_CACHE, _LINE_TEMPERATURE_CACHE

    entries: List[LineRegistryEntry] = []

    config = getattr(cfg.opac, "line", None)
    if not config:
        reset_registry()
        return
    
    # Use the observational wavelengths to interpolate to if no master grid is present
    wavelengths = np.asarray(obs["wl"], dtype=float) if lam_master is None else np.asarray(lam_master, dtype=float)

    # Read in the line data for each species given by the YAML file - add to the entries list
    for index, spec in enumerate(cfg.opac.line):
        path = spec.path
        print("[Line] Reading line xs for", spec.species, "@", path)

        # Check file format
        if not (path.endswith('.h5') or path.endswith('.hdf5')):
            raise ValueError(f"Unsupported file format for {path}. Expected .h5 or .hdf5")

        entry = _load_line_h5(index, path, wavelengths)
        entries.append(entry)

    # Now need to pad in the temperature dimension to make all grids to the same size (for JAX)
    _LINE_ENTRIES = _rectangularize_entries(entries)
    if not _LINE_ENTRIES:
        reset_registry()
        return
    
    # Store all the data as global scope caches - stack using JAX
    _LINE_SIGMA_CACHE = jnp.stack([entry.cross_sections for entry in _LINE_ENTRIES], axis=0)
    _LINE_TEMPERATURE_CACHE = jnp.stack([entry.temperatures for entry in _LINE_ENTRIES], axis=0)

    _clear_cache()

### -- lru cached helper functions below --- ###

@lru_cache(None)
def line_species_names() -> Tuple[str, ...]:
    return tuple(entry.name for entry in _LINE_ENTRIES)


@lru_cache(None)
def line_master_wavelength() -> jnp.ndarray:
    if not _LINE_ENTRIES:
        raise RuntimeError("Line registry empty; call build_opacities() first.")
    return _LINE_ENTRIES[0].wavelengths


@lru_cache(None)
def line_pressure_grid() -> jnp.ndarray:
    if not _LINE_ENTRIES:
        raise RuntimeError("Line registry empty; call build_opacities() first.")
    return _LINE_ENTRIES[0].pressures


@lru_cache(None)
def line_temperature_grids() -> jnp.ndarray:
    if _LINE_TEMPERATURE_CACHE is None:
        raise RuntimeError("Line temperature grids not built; call build_opacities() first.")
    return _LINE_TEMPERATURE_CACHE


@lru_cache(None)
def line_temperature_grid() -> jnp.ndarray:
    return line_temperature_grids()[0]


@lru_cache(None)
def line_sigma_cube() -> jnp.ndarray:
    if _LINE_SIGMA_CACHE is None:
        raise RuntimeError("Line σ cube not built; call build_opacities() first.")
    return _LINE_SIGMA_CACHE


@lru_cache(None)
def line_pick_arrays():
    if not _LINE_ENTRIES:
        raise RuntimeError("Line registry empty; call build_opacities() first.")
    picks_pressures = tuple((lambda _=None, pressures=entry.pressures: pressures) for entry in _LINE_ENTRIES)
    picks_temperatures = tuple((lambda _=None, temperatures=entry.temperatures: temperatures) for entry in _LINE_ENTRIES)
    picks_sigma = tuple((lambda _=None, sigma=entry.cross_sections: sigma) for entry in _LINE_ENTRIES)
    return picks_pressures, picks_temperatures, picks_sigma


__all__ = [
    "LineRegistryEntry",
    "reset_registry",
    "has_line_data",
    "load_line_registry",
    "line_species_names",
    "line_master_wavelength",
    "line_pressure_grid",
    "line_temperature_grid",
    "line_temperature_grids",
    "line_sigma_cube",
    "line_pick_arrays",
]
