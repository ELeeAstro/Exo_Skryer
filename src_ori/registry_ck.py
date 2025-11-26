"""
registry_ck.py
==============

Overview:
    Registry for correlated-k (c-k) opacity tables.

    Similar to registry_line.py but handles 4D opacity tables:
    (temperature, pressure, wavelength, g-point)

    The g-points represent quadrature points for the correlated-k method,
    which approximates line-by-line calculations by sorting absorption
    coefficients within each spectral bin.

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

# Dataclass for each of the correlated-k opacity tables
@dataclass(frozen=True)
class CKRegistryEntry:
    name: str
    idx: int
    pressures: jnp.ndarray       # (n_pressure,)
    temperatures: jnp.ndarray    # (n_temperature,)
    wavelengths: jnp.ndarray     # (n_wavelength,)
    g_points: jnp.ndarray        # (n_g,)
    g_weights: jnp.ndarray       # (n_g,) - quadrature weights
    cross_sections: jnp.ndarray  # (n_temperature, n_pressure, n_wavelength, n_g)


# Global registries and caches for forward model
_CK_ENTRIES: Tuple[CKRegistryEntry, ...] = ()
_CK_SIGMA_CACHE: jnp.ndarray | None = None
_CK_TEMPERATURE_CACHE: jnp.ndarray | None = None
_CK_G_POINTS_CACHE: jnp.ndarray | None = None
_CK_G_WEIGHTS_CACHE: jnp.ndarray | None = None

# Clear all the cache entries
def _clear_cache():
    ck_species_names.cache_clear()
    ck_master_wavelength.cache_clear()
    ck_pressure_grid.cache_clear()
    ck_temperature_grid.cache_clear()
    ck_temperature_grids.cache_clear()
    ck_sigma_cube.cache_clear()
    ck_g_points.cache_clear()
    ck_g_weights.cache_clear()

# Reset all the global registries
def reset_registry():
    global _CK_ENTRIES, _CK_SIGMA_CACHE, _CK_TEMPERATURE_CACHE, _CK_G_POINTS_CACHE, _CK_G_WEIGHTS_CACHE
    _CK_ENTRIES = ()
    _CK_SIGMA_CACHE = None
    _CK_TEMPERATURE_CACHE = None
    _CK_G_POINTS_CACHE = None
    _CK_G_WEIGHTS_CACHE = None
    _clear_cache()

# Check if the registries are set or not
def has_ck_data() -> bool:
    return bool(_CK_ENTRIES)

# Function to load petitRADTRANS HDF5 correlated-k opacity data
def _load_ck_h5(index: int, path: str, target_wavelengths: np.ndarray) -> CKRegistryEntry:
    """
    Load petitRADTRANS HDF5 format correlated-k opacity tables.

    petitRADTRANS format:
    - mol_name or derive from DOI: molecule name (string)
    - p: pressure grid in bar (nP,)
    - t: temperature grid in K (nT,)
    - bin_centers: wavenumber bin centers in cm^-1 (nwl,)
    - kcoeff: correlated-k coefficients in cm^2/molecule (nP, nT, nwl, ng)
    - ngauss: number of gauss points (scalar)
    - weights: gauss quadrature weights (ng,)
    - samples or derive g-points: g-point locations (ng,)

    Returns data in registry format:
    - pressures in bar (nP,)
    - temperatures in K (nT,)
    - wavelengths in microns (from ck table, must match target_wavelengths)
    - g_points: g-point locations (ng,)
    - g_weights: gauss quadrature weights (ng,)
    - cross_sections in log10(cm^2) (nT, nP, nwl, ng)

    Note: Correlated-k tables are pre-banded and cannot be interpolated in wavelength.
          The wavelength grid from the table must match the master wavelength grid.
    """

    with h5py.File(path, 'r') as f:
        # Read molecule name - try different possible keys
        if 'mol_name' in f:
            name = f['mol_name'][0]
            if isinstance(name, bytes):
                name = name.decode('utf-8')
        elif 'molecule' in f:
            name = f['molecule'][0]
            if isinstance(name, bytes):
                name = name.decode('utf-8')
        else:
            # Try to extract from filename
            import os
            basename = os.path.basename(path)
            # Format: molecule__source__R1000_0.3-50mu.ktable.petitRADTRANS.h5
            name = basename.split('__')[0]
        name = str(name)

        # Read grids
        pressures = np.asarray(f['p'][:], dtype=float)  # bar, shape (nP,)
        temperatures = np.asarray(f['t'][:], dtype=float)  # K, shape (nT,)
        bin_centers_wn = np.asarray(f['bin_centers'][:], dtype=float)  # cm^-1, shape (nwl,)
        native_kcoeff = np.asarray(f['kcoeff'][:], dtype=float)  # cm^2/molecule, shape (nP, nT, nwl, ng)

        # Read gauss quadrature information
        ngauss_dataset = f['ngauss']
        if ngauss_dataset.shape == ():  # Scalar dataset
            ngauss = int(ngauss_dataset[()])
        else:
            ngauss = int(ngauss_dataset[:])

        weights = np.asarray(f['weights'][:], dtype=float)  # shape (ng,)

        # Get g-points - either from 'samples' dataset or create uniform grid
        if 'samples' in f:
            g_points = np.asarray(f['samples'][:], dtype=float)
        else:
            # Create uniform g-point grid from 0 to 1
            g_points = np.linspace(0.0, 1.0, ngauss)

    # Convert wavenumber bin centers to wavelength in microns
    # λ[μm] = 10000 / ν[cm^-1]
    wavelengths = 10000.0 / bin_centers_wn  # μm

    # Sort wavelengths (wavenumbers are typically descending, so wavelengths will be ascending)
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]

    # Transpose from (nP, nT, nwl, ng) to (nT, nP, nwl, ng) and apply wavelength sorting
    kcoeff_transposed = np.transpose(native_kcoeff, (1, 0, 2, 3))[:, :, sort_idx, :]

    # Check that wavelength grids match
    if target_wavelengths is not None:
        if len(wavelengths) != len(target_wavelengths):
            raise ValueError(
                f"Wavelength grid mismatch for {name}: "
                f"ck table has {len(wavelengths)} bins, master grid has {len(target_wavelengths)} bins. "
                f"Correlated-k tables cannot be interpolated in wavelength."
            )
        if not np.allclose(wavelengths, target_wavelengths, rtol=1e-6):
            raise ValueError(
                f"Wavelength grid mismatch for {name}: "
                f"ck table wavelengths do not match master grid. "
                f"Correlated-k tables cannot be interpolated in wavelength."
            )

    # Convert to log10 (handle zeros by setting minimum value)
    min_xs = 1e-99  # corresponds to log10 = -99
    kcoeff_log = np.log10(np.maximum(kcoeff_transposed, min_xs))

    # Return a dataclass with all the required entries
    return CKRegistryEntry(
        name=name,
        idx=index,
        pressures=jnp.asarray(pressures),
        temperatures=jnp.asarray(temperatures),
        wavelengths=jnp.asarray(wavelengths),
        g_points=jnp.asarray(g_points),
        g_weights=jnp.asarray(weights),
        cross_sections=jnp.asarray(kcoeff_log),
    )


# Pad the tables to a rectangle (in dimension) - usually only in T and g as wavelength and pressure grids are the same
def _rectangularize_entries(entries: List[CKRegistryEntry]) -> Tuple[CKRegistryEntry, ...]:

    # Return if zero c-k table
    if not entries:
        return ()

    # Find the wavelength and pressure grid from the first tables (should be the same across all species)
    base_wavelengths = entries[0].wavelengths
    base_pressures = entries[0].pressures
    expected_wavelengths = base_wavelengths.shape[0]
    for entry in entries[1:]:
        if entry.wavelengths.shape != base_wavelengths.shape or not np.allclose(entry.wavelengths, base_wavelengths):
            raise ValueError(f"c-k opacity wavelength grids differ between {entries[0].name} and {entry.name}.")
        if entry.pressures.shape != base_pressures.shape or not np.allclose(entry.pressures, base_pressures):
            raise ValueError(f"c-k opacity pressure grids differ between {entries[0].name} and {entry.name}.")

    # Find the max number of pressure points
    max_pressures = max(entry.pressures.shape[0] for entry in entries)
    # Find the max number of temperature points
    max_temperatures = max(entry.temperatures.shape[0] for entry in entries)
    # Find the max number of g-points
    max_g = max(entry.g_points.shape[0] for entry in entries)

    # Start a new list for the padded cross section tables and pad the temperature and g arrays
    padded_entries: List[CKRegistryEntry] = []
    for entry in entries:
        pressures = jnp.asarray(entry.pressures)
        temperatures = jnp.asarray(entry.temperatures)
        g_points = jnp.asarray(entry.g_points)
        g_weights = jnp.asarray(entry.g_weights)
        xs = jnp.asarray(entry.cross_sections)

        current_temperatures, current_pressures, wavelength_count, current_g = xs.shape

        if wavelength_count != expected_wavelengths:
            raise ValueError(f"Species {entry.name} has λ grid length {wavelength_count}, expected {expected_wavelengths}.")
        if current_pressures != max_pressures:
            raise ValueError(f"Species {entry.name} has nP={current_pressures}, expected {max_pressures} for common grid.")

        # Pad temperatures
        pad_temperatures = max_temperatures - current_temperatures
        if pad_temperatures > 0:
            temperatures = jnp.pad(temperatures, (0, pad_temperatures), mode="edge")
            xs = jnp.pad(xs, ((0, pad_temperatures), (0, 0), (0, 0), (0, 0)), mode="edge")

        # Pad g-points and g-weights
        pad_g = max_g - current_g
        if pad_g > 0:
            g_points = jnp.pad(g_points, (0, pad_g), mode="edge")
            g_weights = jnp.pad(g_weights, (0, pad_g), constant_values=0.0)  # Pad weights with 0
            xs = jnp.pad(xs, ((0, 0), (0, 0), (0, 0), (0, pad_g)), mode="edge")

        padded_entries.append(
            CKRegistryEntry(
                name=entry.name,
                idx=entry.idx,
                pressures=pressures,
                temperatures=temperatures,
                wavelengths=base_wavelengths,
                g_points=g_points,
                g_weights=g_weights,
                cross_sections=xs,
            )
        )
    return tuple(padded_entries)


# Read in and prepare the correlated-k data
def load_ck_registry(cfg, obs, lam_master: Optional[np.ndarray] = None):

    # Allocate the global scope caches
    global _CK_ENTRIES, _CK_SIGMA_CACHE, _CK_TEMPERATURE_CACHE, _CK_G_POINTS_CACHE, _CK_G_WEIGHTS_CACHE

    entries: List[CKRegistryEntry] = []

    config = getattr(cfg.opac, "ck", None)
    if not config:
        reset_registry()
        return

    # Use the observational wavelengths to interpolate to if no master grid is present
    wavelengths = np.asarray(obs["wl"], dtype=float) if lam_master is None else np.asarray(lam_master, dtype=float)

    # Read in the c-k data for each species given by the YAML file - add to the entries list
    for index, spec in enumerate(cfg.opac.ck):
        path = spec.path
        print("[c-k] Reading correlated-k xs for", spec.species, "@", path)

        # Check file format
        if not (path.endswith('.h5') or path.endswith('.hdf5')):
            raise ValueError(f"Unsupported file format for {path}. Expected .h5 or .hdf5")

        entry = _load_ck_h5(index, path, wavelengths)
        entries.append(entry)

    # Now need to pad in the temperature and g dimensions to make all grids to the same size (for JAX)
    _CK_ENTRIES = _rectangularize_entries(entries)
    if not _CK_ENTRIES:
        reset_registry()
        return

    # Store all the data as global scope caches - stack using JAX
    _CK_SIGMA_CACHE = jnp.stack([entry.cross_sections for entry in _CK_ENTRIES], axis=0)
    _CK_TEMPERATURE_CACHE = jnp.stack([entry.temperatures for entry in _CK_ENTRIES], axis=0)
    _CK_G_POINTS_CACHE = jnp.stack([entry.g_points for entry in _CK_ENTRIES], axis=0)
    _CK_G_WEIGHTS_CACHE = jnp.stack([entry.g_weights for entry in _CK_ENTRIES], axis=0)

    _clear_cache()

### -- lru cached helper functions below --- ###

@lru_cache(None)
def ck_species_names() -> Tuple[str, ...]:
    return tuple(entry.name for entry in _CK_ENTRIES)


@lru_cache(None)
def ck_master_wavelength() -> jnp.ndarray:
    if not _CK_ENTRIES:
        raise RuntimeError("c-k registry empty; call build_opacities() first.")
    return _CK_ENTRIES[0].wavelengths


@lru_cache(None)
def ck_pressure_grid() -> jnp.ndarray:
    if not _CK_ENTRIES:
        raise RuntimeError("c-k registry empty; call build_opacities() first.")
    return _CK_ENTRIES[0].pressures


@lru_cache(None)
def ck_temperature_grids() -> jnp.ndarray:
    if _CK_TEMPERATURE_CACHE is None:
        raise RuntimeError("c-k temperature grids not built; call build_opacities() first.")
    return _CK_TEMPERATURE_CACHE


@lru_cache(None)
def ck_temperature_grid() -> jnp.ndarray:
    return ck_temperature_grids()[0]


@lru_cache(None)
def ck_sigma_cube() -> jnp.ndarray:
    if _CK_SIGMA_CACHE is None:
        raise RuntimeError("c-k σ cube not built; call build_opacities() first.")
    return _CK_SIGMA_CACHE


@lru_cache(None)
def ck_g_points() -> jnp.ndarray:
    if _CK_G_POINTS_CACHE is None:
        raise RuntimeError("c-k g-points not built; call build_opacities() first.")
    return _CK_G_POINTS_CACHE


@lru_cache(None)
def ck_g_weights() -> jnp.ndarray:
    if _CK_G_WEIGHTS_CACHE is None:
        raise RuntimeError("c-k g-weights not built; call build_opacities() first.")
    return _CK_G_WEIGHTS_CACHE


__all__ = [
    "CKRegistryEntry",
    "reset_registry",
    "has_ck_data",
    "load_ck_registry",
    "ck_species_names",
    "ck_master_wavelength",
    "ck_pressure_grid",
    "ck_temperature_grid",
    "ck_temperature_grids",
    "ck_sigma_cube",
    "ck_g_points",
    "ck_g_weights",
]
