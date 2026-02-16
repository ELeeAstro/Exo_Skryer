"""
registry_line.py
================
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import h5py


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
    "line_log10_pressure_grid",
    "line_log10_temperature_grids",
]


# Dataclass for each of the line opacity tables
# Note: During preprocessing, all arrays are NumPy (CPU)
# They get converted to JAX (device) only at the final cache creation step
# Mixed precision: float64 for grids (better accuracy), float32 for cross sections (memory savings)
@dataclass(frozen=True)
class LineRegistryEntry:
    name: str
    idx: int
    pressures: np.ndarray      # NumPy during preprocessing (float64)
    temperatures: np.ndarray    # NumPy during preprocessing (float64)
    wavelengths: np.ndarray     # NumPy during preprocessing (float64)
    cross_sections: np.ndarray  # NumPy during preprocessing (float32 to save memory)


# Global registries and caches for forward model
_LINE_SPECIES_NAMES: Tuple[str, ...] = ()  # Lightweight: only species names (few bytes)
_LINE_SIGMA_CACHE: jnp.ndarray | None = None
_LINE_TEMPERATURE_CACHE: jnp.ndarray | None = None
_LINE_WAVELENGTH_CACHE: jnp.ndarray | None = None
_LINE_PRESSURE_CACHE: jnp.ndarray | None = None
_LINE_LOG10_TEMPERATURE_CACHE: jnp.ndarray | None = None
_LINE_LOG10_PRESSURE_CACHE: jnp.ndarray | None = None

# Clear all the cache entries
def _clear_cache():
    line_species_names.cache_clear()
    line_master_wavelength.cache_clear()
    line_pressure_grid.cache_clear()
    line_temperature_grid.cache_clear()
    line_temperature_grids.cache_clear()
    line_sigma_cube.cache_clear()
    line_log10_pressure_grid.cache_clear()
    line_log10_temperature_grids.cache_clear()

# Reset all the global registries
def reset_registry() -> None:
    global _LINE_SPECIES_NAMES, _LINE_SIGMA_CACHE, _LINE_TEMPERATURE_CACHE, _LINE_WAVELENGTH_CACHE, _LINE_PRESSURE_CACHE
    global _LINE_LOG10_TEMPERATURE_CACHE, _LINE_LOG10_PRESSURE_CACHE
    _LINE_SPECIES_NAMES = ()
    _LINE_SIGMA_CACHE = None
    _LINE_TEMPERATURE_CACHE = None
    _LINE_WAVELENGTH_CACHE = None
    _LINE_PRESSURE_CACHE = None
    _LINE_LOG10_TEMPERATURE_CACHE = None
    _LINE_LOG10_PRESSURE_CACHE = None
    _clear_cache()

# Check if the registries are set or not
def has_line_data() -> bool:
    return _LINE_SIGMA_CACHE is not None

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

    # Interpolate to target wavelength grid.
    # Keep NumPy-side preprocessing in float64; downcast happens at JAX cache transfer.
    xs_interp = np.empty((n_temperatures, n_pressures, wavelength_count), dtype=np.float64)
    for iT in range(n_temperatures):
        for iP in range(n_pressures):
            xs_interp[iT, iP, :] = np.interp(
                target_wavelengths,
                native_wavelengths,
                native_xs_log[iT, iP, :],
                left=-99.0,
                right=-99.0
            )

    # Return a dataclass with NumPy arrays (will be converted to JAX later).
    # Keep NumPy-side arrays float64 for preprocessing consistency.
    return LineRegistryEntry(
        name=name,
        idx=index,
        pressures=pressures.astype(np.float64),
        temperatures=temperatures.astype(np.float64),
        wavelengths=target_wavelengths.astype(np.float64),
        cross_sections=xs_interp.astype(np.float64),
    )


def _load_line_npz(index: int, path: str, target_wavelengths: np.ndarray) -> LineRegistryEntry:
    """
    Load opacity tables stored in the custom NPZ format generated by Gen_lbl_table_nu.py.

    Expected NPZ contents:
        - molecule: species name (string or bytes)
        - temperature: temperature grid in K (nT,)
        - pressure: pressure grid in bar (nP,)
        - wavelength: wavelength grid in microns (nwl,)
        - cross_section: log10 cross-sections (nT, nP, nwl)
    """

    with np.load(path) as data:
        name_raw = data["molecule"]
        temperatures = np.asarray(data["temperature"], dtype=float)
        pressures = np.asarray(data["pressure"], dtype=float)
        native_wavelengths = np.asarray(data["wavelength"], dtype=float)
        xs = np.asarray(data["cross_section"], dtype=float)

    if isinstance(name_raw, np.ndarray):
        name = name_raw.tolist()
        if isinstance(name, list):
            name = name[0]
    else:
        name = name_raw
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    name = str(name)

    if xs.ndim != 3:
        raise ValueError(f"Invalid cross_section shape {xs.shape} in {path}; expected 3D array.")

    n_temperatures, n_pressures, native_wl_count = xs.shape
    if native_wavelengths.size != native_wl_count:
        raise ValueError(
            f"Wavelength grid length {native_wavelengths.size} does not match cross-section axis {native_wl_count} in {path}."
        )

    target_wavelengths = np.asarray(target_wavelengths, dtype=float)
    if target_wavelengths.ndim != 1:
        raise ValueError("Target wavelength grid must be 1D.")

    if native_wavelengths.shape == target_wavelengths.shape and np.allclose(native_wavelengths, target_wavelengths):
        xs_interp = xs
    else:
        # Keep NumPy-side preprocessing in float64; downcast happens at JAX cache transfer.
        xs_interp = np.empty((n_temperatures, n_pressures, target_wavelengths.size), dtype=np.float64)
        for iT in range(n_temperatures):
            for iP in range(n_pressures):
                xs_interp[iT, iP, :] = np.interp(
                    target_wavelengths,
                    native_wavelengths,
                    xs[iT, iP, :],
                    left=-99.0,
                    right=-99.0,
                )

    # Return NumPy arrays (will be converted to JAX later).
    # Keep NumPy-side arrays float64 for preprocessing consistency.
    return LineRegistryEntry(
        name=name,
        idx=index,
        pressures=pressures.astype(np.float64),
        temperatures=temperatures.astype(np.float64),
        wavelengths=target_wavelengths.astype(np.float64),
        cross_sections=xs_interp.astype(np.float64),
    )

# Pad the tables to a rectangle (in dimension) - usually only in T as wavelength and pressure grids are the same lengths
# Uses NumPy for preprocessing (CPU-based padding before sending to device)
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
        # Keep as NumPy arrays for preprocessing
        pressures = entry.pressures
        temperatures = entry.temperatures
        xs = entry.cross_sections
        current_temperatures, current_pressures, wavelength_count = xs.shape
        if wavelength_count != expected_wavelengths:
            raise ValueError(f"Species {entry.name} has λ grid length {wavelength_count}, expected {expected_wavelengths}.")
        if current_pressures != max_pressures:
            raise ValueError(f"Species {entry.name} has nP={current_pressures}, expected {max_pressures} for common grid.")
        pad_temperatures = max_temperatures - current_temperatures
        if pad_temperatures > 0:
            # Use NumPy padding (CPU-based)
            temperatures = np.pad(temperatures, (0, pad_temperatures), mode="edge")
            xs = np.pad(xs, ((0, pad_temperatures), (0, 0), (0, 0)), mode="edge")
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
def load_line_registry(cfg, obs, lam_master: Optional[np.ndarray] = None, base_dir: Optional[Path] = None):

    # Allocate the global scope caches
    global _LINE_SPECIES_NAMES, _LINE_SIGMA_CACHE, _LINE_TEMPERATURE_CACHE, _LINE_WAVELENGTH_CACHE, _LINE_PRESSURE_CACHE
    global _LINE_LOG10_TEMPERATURE_CACHE, _LINE_LOG10_PRESSURE_CACHE

    entries: List[LineRegistryEntry] = []

    config = getattr(cfg.opac, "line", None)
    if not config:
        reset_registry()
        return
    
    # Use the observational wavelengths to interpolate to if no master grid is present
    wavelengths = np.asarray(obs["wl"], dtype=float) if lam_master is None else np.asarray(lam_master, dtype=float)

    # Read in the line data for each species given by the YAML file - add to the entries list
    for index, spec in enumerate(cfg.opac.line):
        path = Path(spec.path).expanduser()
        if not path.is_absolute():
            if base_dir is not None:
                path = (Path(base_dir) / path).resolve()
            else:
                path = path.resolve()
        path_str = str(path)
        print("[Line] Reading line xs for", spec.species, "@", path_str)

        # Check file format
        if path_str.endswith(".npz"):
            entry = _load_line_npz(index, path_str, wavelengths)
        elif path_str.endswith('.h5') or path_str.endswith('.hdf5'):
            entry = _load_line_h5(index, path_str, wavelengths)
        else:
            raise ValueError(f"Unsupported file format for {path_str}. Expected .npz, .h5 or .hdf5")
        entries.append(entry)

    # Now need to pad in the temperature dimension to make all grids to the same size (for JAX)
    rectangularized_entries = _rectangularize_entries(entries)
    if not rectangularized_entries:
        reset_registry()
        return

    # ============================================================================
    # CRITICAL: Convert NumPy arrays to JAX arrays here (ONE transfer to device)
    # ============================================================================
    # All preprocessing is done in NumPy (CPU). Now we send the final data
    # to the device (GPU/CPU as configured) for use in JIT-compiled forward model.
    # Mixed precision strategy:
    #   - float64 for grids (pressures, temperatures, wavelengths) → better interpolation accuracy
    #   - float32 for cross sections → halves memory usage (~4 GB → ~2 GB for large grids)
    # ============================================================================

    print(f"[Line] Transferring {len(rectangularized_entries)} species to device...")

    # Stack cross sections: (n_species, nT, nP, nwl) - already float32 from preprocessing
    sigma_stacked = np.stack([entry.cross_sections for entry in rectangularized_entries], axis=0)
    _LINE_SIGMA_CACHE = jnp.asarray(sigma_stacked, dtype=jnp.float32)

    # Stack temperature grids: (n_species, nT) - keep as float64 for accuracy
    temp_stacked = np.stack([entry.temperatures for entry in rectangularized_entries], axis=0)
    _LINE_TEMPERATURE_CACHE = jnp.asarray(temp_stacked, dtype=jnp.float64)

    # Wavelength and pressure grids: all species share the same grids (rectangularized)
    _LINE_WAVELENGTH_CACHE = jnp.asarray(rectangularized_entries[0].wavelengths, dtype=jnp.float64)
    _LINE_PRESSURE_CACHE = jnp.asarray(rectangularized_entries[0].pressures, dtype=jnp.float64)

    # Pre-compute log10 of grids for efficient interpolation
    _LINE_LOG10_PRESSURE_CACHE = jnp.log10(_LINE_PRESSURE_CACHE)
    _LINE_LOG10_TEMPERATURE_CACHE = jnp.log10(_LINE_TEMPERATURE_CACHE)

    print(f"[Line] Cross section cache: {_LINE_SIGMA_CACHE.shape} (dtype: {_LINE_SIGMA_CACHE.dtype})")
    print(f"[Line] Temperature cache: {_LINE_TEMPERATURE_CACHE.shape} (dtype: {_LINE_TEMPERATURE_CACHE.dtype})")
    print(f"[Line] Cached log10(P) and log10(T) grids for efficient interpolation")

    # Estimate memory usage
    sigma_mb = _LINE_SIGMA_CACHE.size * _LINE_SIGMA_CACHE.itemsize / 1024**2
    temp_mb = _LINE_TEMPERATURE_CACHE.size * _LINE_TEMPERATURE_CACHE.itemsize / 1024**2
    wl_mb = _LINE_WAVELENGTH_CACHE.size * _LINE_WAVELENGTH_CACHE.itemsize / 1024**2
    p_mb = _LINE_PRESSURE_CACHE.size * _LINE_PRESSURE_CACHE.itemsize / 1024**2
    total_mb = sigma_mb + temp_mb + wl_mb + p_mb
    print(f"[Line] Estimated device memory: {total_mb:.1f} MB (σ: {sigma_mb:.1f} MB, T: {temp_mb:.2f} MB, λ: {wl_mb:.2f} MB, P: {p_mb:.2f} MB)")

    # Extract species names (lightweight: just strings)
    _LINE_SPECIES_NAMES = tuple(entry.name for entry in rectangularized_entries)

    # Delete NumPy arrays to free memory (JAX caches now hold the data on device)
    # This saves ~100s of MB for typical opacity tables
    del rectangularized_entries, entries, sigma_stacked, temp_stacked
    print(f"[Line] Freed NumPy temporary arrays from CPU memory")

    _clear_cache()

### -- lru cached helper functions below --- ###

@lru_cache(None)
def line_species_names() -> Tuple[str, ...]:
    if not _LINE_SPECIES_NAMES:
        raise RuntimeError("Line registry empty; call build_opacities() first.")
    return _LINE_SPECIES_NAMES


@lru_cache(None)
def line_master_wavelength() -> jnp.ndarray:
    if _LINE_WAVELENGTH_CACHE is None:
        raise RuntimeError("Line registry empty; call build_opacities() first.")
    return _LINE_WAVELENGTH_CACHE


@lru_cache(None)
def line_pressure_grid() -> jnp.ndarray:
    if _LINE_PRESSURE_CACHE is None:
        raise RuntimeError("Line registry empty; call build_opacities() first.")
    return _LINE_PRESSURE_CACHE


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
def line_log10_pressure_grid() -> jnp.ndarray:
    if _LINE_LOG10_PRESSURE_CACHE is None:
        raise RuntimeError("Line log10(P) grid not built; call build_opacities() first.")
    return _LINE_LOG10_PRESSURE_CACHE


@lru_cache(None)
def line_log10_temperature_grids() -> jnp.ndarray:
    if _LINE_LOG10_TEMPERATURE_CACHE is None:
        raise RuntimeError("Line log10(T) grids not built; call build_opacities() first.")
    return _LINE_LOG10_TEMPERATURE_CACHE
