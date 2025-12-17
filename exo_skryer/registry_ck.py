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
from typing import List, Tuple, Optional, Any
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import h5py

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
    "ck_g_weights"
]


# Dataclass for each of the correlated-k opacity tables
# Note: During preprocessing, all arrays are NumPy (CPU)
# They get converted to JAX (device) only at the final cache creation step
# Mixed precision: float64 for grids (better accuracy), float32 for cross sections (memory savings)
@dataclass(frozen=True)
class CKRegistryEntry:
    name: str
    idx: int
    pressures: np.ndarray        # NumPy during preprocessing (float64) - (n_pressure,)
    temperatures: np.ndarray     # NumPy during preprocessing (float64) - (n_temperature,)
    wavelengths: np.ndarray      # NumPy during preprocessing (float64) - (n_wavelength,)
    g_points: np.ndarray         # NumPy during preprocessing (float64) - (n_g,)
    g_weights: np.ndarray        # NumPy during preprocessing (float64) - (n_g,) - quadrature weights
    cross_sections: np.ndarray   # NumPy during preprocessing (float32) - (n_temperature, n_pressure, n_wavelength, n_g)


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
def _load_ck_h5(index: int, spec, path: str, obs: dict, use_full_grid: bool = False) -> CKRegistryEntry:
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
    - wavelengths in microns (cut to obs bands)
    - g_points: g-point locations (ng,)
    - g_weights: gauss quadrature weights (ng,)
    - cross_sections in log10(cm^2) (nT, nP, nwl_cut, ng)

    Note: Correlated-k tables are pre-banded and cannot be interpolated in wavelength.
          This function cuts the table to only wavelengths within observation bands.
    """

    name = getattr(spec, "species", f"ck_{index}")

    with h5py.File(path, 'r') as f:

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

    # Create mask for wavelengths within observation bands
    if use_full_grid:
        mask = np.ones_like(wavelengths, dtype=bool)
        print(f"[c-k] Using full wavelength grid for {name}: {len(wavelengths)} bins")
    else:
        wl_obs = np.asarray(obs["wl"], dtype=float)
        dwl_obs = np.asarray(obs["dwl"], dtype=float)
        left_edges = wl_obs - dwl_obs
        right_edges = wl_obs + dwl_obs

        # Mask wavelengths that fall within any observation bin
        mask = np.any(
            (wavelengths[None, :] >= left_edges[:, None]) & (wavelengths[None, :] <= right_edges[:, None]),
            axis=0,
        )

        if not np.any(mask):
            raise ValueError(f"No CK wavelengths for {name} lie within observation bins.")

        print(f"[c-k] Cut wavelength grid for {name}: {np.sum(mask)}/{len(wavelengths)} bins retained")

    # Apply mask to wavelengths and cross_sections
    wavelengths_cut = wavelengths[mask]
    kcoeff_cut = kcoeff_transposed[:, :, mask, :]

    # Convert to log10 (handle zeros by setting minimum value)
    # Use float32 to save memory (log10 cross sections don't need float64 precision)
    min_xs = 1e-99  # corresponds to log10 = -99
    kcoeff_log = np.log10(np.maximum(kcoeff_cut, min_xs)).astype(np.float32)

    # Return a dataclass with NumPy arrays (will be converted to JAX later)
    # Mixed precision: float64 for grids (better interpolation accuracy), float32 for cross sections
    return CKRegistryEntry(
        name=name,
        idx=index,
        pressures=pressures.astype(np.float64),
        temperatures=temperatures.astype(np.float64),
        wavelengths=wavelengths_cut.astype(np.float64),
        g_points=g_points.astype(np.float64),
        g_weights=weights.astype(np.float64),
        cross_sections=kcoeff_log,  # Already float32
    )


def _load_ck_npz(index: int, spec, path: str, obs: dict, use_full_grid: bool = False) -> CKRegistryEntry:
    """
    Load correlated-k opacity tables stored in the custom NPZ format.

    Expected NPZ contents:
        - molecule: species name (string or bytes; optional)
        - pressure: pressure grid in bar (nP,)
        - temperature: temperature grid in K (nT,)
        - wavelength: wavelength grid in microns (nwl,)
        - g_points: g-point locations (ng,)
        - g_weights: quadrature weights (ng,)
        - cross_section: log10 cross-sections (nT, nP, nwl, ng)
    """

    with np.load(path) as data:
        cross_section = np.asarray(data["cross_section"], dtype=float)
        name_raw = data.get("molecule", getattr(spec, "species", f"ck_{index}"))
        pressures = np.asarray(data["pressure"], dtype=float)
        temperatures = np.asarray(data["temperature"], dtype=float)
        wavelengths = np.asarray(data["wavelength"], dtype=float)
        nG = cross_section.shape[-1]
        g_points = np.asarray(data.get("g_points", np.linspace(0.0, 1.0, nG)), dtype=float)
        default_weights = np.ones_like(g_points) / g_points.size if g_points.size > 0 else g_points
        g_weights = np.asarray(data.get("g_weights", default_weights), dtype=float)

    if isinstance(name_raw, np.ndarray):
        name = name_raw.tolist()
        if isinstance(name, list):
            name = name[0]
    else:
        name = name_raw
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    name = str(name)

    if cross_section.ndim != 4:
        raise ValueError(f"Invalid cross_section shape {cross_section.shape} in {path}; expected 4D array.")

    nT, nP, nW, nG = cross_section.shape
    if wavelengths.size != nW:
        raise ValueError(f"Wavelength grid length {wavelengths.size} does not match cross-section axis {nW} in {path}.")
    if g_points.size != nG:
        raise ValueError(f"g_point array length {g_points.size} does not match cross-section axis {nG} in {path}.")
    if g_weights.size != nG:
        raise ValueError(f"g_weight array length {g_weights.size} does not match cross-section axis {nG} in {path}.")

    if not use_full_grid:
        wl_obs = np.asarray(obs["wl"], dtype=float)
        dwl_obs = np.asarray(obs["dwl"], dtype=float)
        left_edges = wl_obs - dwl_obs
        right_edges = wl_obs + dwl_obs
        mask = np.any(
            (wavelengths[None, :] >= left_edges[:, None]) & (wavelengths[None, :] <= right_edges[:, None]),
            axis=0,
        )
        if not np.any(mask):
            raise ValueError(f"No CK wavelengths for {name} lie within observation bins.")
        wavelengths = wavelengths[mask]
        cross_section = cross_section[:, :, mask, :]
    else:
        print(f"[c-k] Using full wavelength grid for {name}: {wavelengths.size} bins")

    # Return a dataclass with NumPy arrays (will be converted to JAX later)
    # Mixed precision: float64 for grids (better interpolation accuracy), float32 for cross sections
    return CKRegistryEntry(
        name=name,
        idx=index,
        pressures=pressures.astype(np.float64),
        temperatures=temperatures.astype(np.float64),
        wavelengths=wavelengths.astype(np.float64),
        g_points=g_points.astype(np.float64),
        g_weights=g_weights.astype(np.float64),
        cross_sections=cross_section.astype(np.float32),  # float32 to save memory
    )


# Pad the tables to a rectangle (in dimension) - usually only in T and g as wavelength and pressure grids are the same
# Uses NumPy for preprocessing (CPU-based padding before sending to device)
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
        # Keep as NumPy arrays for preprocessing
        pressures = entry.pressures
        temperatures = entry.temperatures
        g_points = entry.g_points
        g_weights = entry.g_weights
        xs = entry.cross_sections

        current_temperatures, current_pressures, wavelength_count, current_g = xs.shape

        if wavelength_count != expected_wavelengths:
            raise ValueError(f"Species {entry.name} has λ grid length {wavelength_count}, expected {expected_wavelengths}.")
        if current_pressures != max_pressures:
            raise ValueError(f"Species {entry.name} has nP={current_pressures}, expected {max_pressures} for common grid.")

        # Pad temperatures (use NumPy padding)
        pad_temperatures = max_temperatures - current_temperatures
        if pad_temperatures > 0:
            temperatures = np.pad(temperatures, (0, pad_temperatures), mode="edge")
            xs = np.pad(xs, ((0, pad_temperatures), (0, 0), (0, 0), (0, 0)), mode="edge")

        # Pad g-points and g-weights (use NumPy padding)
        pad_g = max_g - current_g
        if pad_g > 0:
            g_points = np.pad(g_points, (0, pad_g), mode="edge")
            g_weights = np.pad(g_weights, (0, pad_g), constant_values=0.0)  # Pad weights with 0
            xs = np.pad(xs, ((0, 0), (0, 0), (0, 0), (0, pad_g)), mode="edge")

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
def load_ck_registry(cfg, obs, lam_master: Optional[np.ndarray] = None, base_dir: Optional[Path] = None):

    # Allocate the global scope caches
    global _CK_ENTRIES, _CK_SIGMA_CACHE, _CK_TEMPERATURE_CACHE, _CK_G_POINTS_CACHE, _CK_G_WEIGHTS_CACHE

    entries: List[CKRegistryEntry] = []

    # When cfg.opac.ck is True (boolean), species are listed in cfg.opac.line
    # When cfg.opac.ck is a list, it contains the species directly
    ck_mode = getattr(cfg.opac, "ck", None)
    if not ck_mode:
        reset_registry()
        return

    # Get species list: if ck is True/False, use cfg.opac.line; otherwise use ck itself
    if isinstance(ck_mode, bool):
        config = getattr(cfg.opac, "line", None)
    else:
        config = ck_mode

    if not config or config in ("None", "none"):
        reset_registry()
        return

    # Check if using full grid (from cfg.opac.full_grid)
    use_full_grid = getattr(cfg.opac, "full_grid", False)

    # Read in the c-k data for each species given by the YAML file - add to the entries list
    for index, spec in enumerate(config):
        path = Path(spec.path).expanduser()
        if not path.is_absolute():
            if base_dir is not None:
                path = (Path(base_dir) / path).resolve()
            else:
                path = path.resolve()
        path_str = str(path)
        print("[c-k] Reading correlated-k xs for", spec.species, "@", path_str)

        # Check file format
        if path_str.endswith(".npz"):
            entry = _load_ck_npz(index, spec, path_str, obs, use_full_grid=use_full_grid)
        elif path_str.endswith('.h5') or path_str.endswith('.hdf5'):
            entry = _load_ck_h5(index, spec, path_str, obs, use_full_grid=use_full_grid)
        else:
            raise ValueError(f"Unsupported file format for {path_str}. Expected .npz, .h5 or .hdf5")
        entries.append(entry)

    # Now need to pad in the temperature and g dimensions to make all grids to the same size (for JAX)
    _CK_ENTRIES = _rectangularize_entries(entries)
    if not _CK_ENTRIES:
        reset_registry()
        return

    # ============================================================================
    # CRITICAL: Convert NumPy arrays to JAX arrays here (ONE transfer to device)
    # ============================================================================
    # All preprocessing is done in NumPy (CPU). Now we send the final data
    # to the device (GPU/CPU as configured) for use in JIT-compiled forward model.
    # Mixed precision strategy:
    #   - float64 for grids (pressures, temperatures, wavelengths, g_points, g_weights) → better interpolation accuracy
    #   - float32 for cross sections → halves memory usage
    # ============================================================================

    print(f"[c-k] Transferring {len(_CK_ENTRIES)} species to device...")

    # Stack cross sections: (n_species, nT, nP, nwl, ng) - already float32 from preprocessing
    sigma_stacked = np.stack([entry.cross_sections for entry in _CK_ENTRIES], axis=0)
    _CK_SIGMA_CACHE = jnp.asarray(sigma_stacked, dtype=jnp.float32)

    # Stack temperature grids: (n_species, nT) - keep as float64 for accuracy
    temp_stacked = np.stack([entry.temperatures for entry in _CK_ENTRIES], axis=0)
    _CK_TEMPERATURE_CACHE = jnp.asarray(temp_stacked, dtype=jnp.float64)

    # Stack g-points: (n_species, ng) - keep as float64 for accuracy
    g_points_stacked = np.stack([entry.g_points for entry in _CK_ENTRIES], axis=0)
    _CK_G_POINTS_CACHE = jnp.asarray(g_points_stacked, dtype=jnp.float64)

    # Stack g-weights: (n_species, ng) - keep as float64 for accuracy
    g_weights_stacked = np.stack([entry.g_weights for entry in _CK_ENTRIES], axis=0)
    _CK_G_WEIGHTS_CACHE = jnp.asarray(g_weights_stacked, dtype=jnp.float64)

    print(f"[c-k] Cross section cache: {_CK_SIGMA_CACHE.shape} (dtype: {_CK_SIGMA_CACHE.dtype})")
    print(f"[c-k] Temperature cache: {_CK_TEMPERATURE_CACHE.shape} (dtype: {_CK_TEMPERATURE_CACHE.dtype})")
    print(f"[c-k] G-points cache: {_CK_G_POINTS_CACHE.shape} (dtype: {_CK_G_POINTS_CACHE.dtype})")
    print(f"[c-k] G-weights cache: {_CK_G_WEIGHTS_CACHE.shape} (dtype: {_CK_G_WEIGHTS_CACHE.dtype})")

    # Estimate memory usage
    sigma_mb = _CK_SIGMA_CACHE.size * _CK_SIGMA_CACHE.itemsize / 1024**2
    temp_mb = _CK_TEMPERATURE_CACHE.size * _CK_TEMPERATURE_CACHE.itemsize / 1024**2
    g_points_mb = _CK_G_POINTS_CACHE.size * _CK_G_POINTS_CACHE.itemsize / 1024**2
    g_weights_mb = _CK_G_WEIGHTS_CACHE.size * _CK_G_WEIGHTS_CACHE.itemsize / 1024**2
    total_mb = sigma_mb + temp_mb + g_points_mb + g_weights_mb
    print(f"[c-k] Estimated device memory: {total_mb:.1f} MB (σ: {sigma_mb:.1f} MB, T: {temp_mb:.2f} MB, g: {g_points_mb:.2f} MB, w: {g_weights_mb:.2f} MB)")

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
