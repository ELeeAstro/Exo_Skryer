"""
registry_cia.py
===============
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional
from pathlib import Path

import jax.numpy as jnp
import numpy as np

__all__ = [
    "CiaRegistryEntry",
    "reset_registry",
    "has_cia_data",
    "load_cia_registry",
    "cia_species_names",
    "cia_master_wavelength",
    "cia_sigma_cube",
    "cia_temperature_grid",
    "cia_temperature_grids",
    "cia_log10_temperature_grids",
]


# Plain Python lists (exact values)
An_ff1 = [518.1021, 472.2636, -482.2089, 115.5291, 0.0, 0.0]
Bn_ff1 = [-734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0]
Cn_ff1 = [1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0]
Dn_ff1 = [-479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0]
En_ff1 = [93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0]
Fn_ff1 = [-6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0]

An_ff2 = [0.0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830]
Bn_ff2 = [0.0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170]
Cn_ff2 = [0.0, -2054.2910, 8746.5230, -13651.1050, 8642.9700, -1863.8640]
Dn_ff2 = [0.0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880]
En_ff2 = [0.0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880]
Fn_ff2 = [0.0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850]

Cn_bf = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]

alf = 1.439e8
lam_0 = 1.6419
lam_min = 0.125


# Dataclass containing the CIA table data
# Note: During preprocessing, all arrays are NumPy (CPU)
# They get converted to JAX (device) only at the final cache creation step
# Float64 throughout for grids and cross sections.
@dataclass(frozen=True)
class CiaRegistryEntry:
    name: str
    idx: int
    temperatures: np.ndarray    # NumPy during preprocessing (float64)
    wavelengths: np.ndarray     # NumPy during preprocessing (float64)
    cross_sections: np.ndarray  # NumPy during preprocessing (float64)


# Global scope cache data array
_CIA_SPECIES_NAMES: Tuple[str, ...] = ()  # Lightweight: only species names (few bytes)
_CIA_SIGMA_CACHE: jnp.ndarray | None = None
_CIA_TEMPERATURE_CACHE: jnp.ndarray | None = None
_CIA_WAVELENGTH_CACHE: jnp.ndarray | None = None
_CIA_LOG10_TEMPERATURE_CACHE: jnp.ndarray | None = None

# Clear cache helper function
def _clear_cache():
    cia_species_names.cache_clear()
    cia_master_wavelength.cache_clear()
    cia_temperature_grids.cache_clear()
    cia_temperature_grid.cache_clear()
    cia_sigma_cube.cache_clear()
    cia_log10_temperature_grids.cache_clear()

# Reset all registry values
def reset_registry():
    global _CIA_SPECIES_NAMES, _CIA_SIGMA_CACHE, _CIA_TEMPERATURE_CACHE
    global _CIA_WAVELENGTH_CACHE, _CIA_LOG10_TEMPERATURE_CACHE
    _CIA_SPECIES_NAMES = ()
    _CIA_SIGMA_CACHE = None
    _CIA_TEMPERATURE_CACHE = None
    _CIA_WAVELENGTH_CACHE = None
    _CIA_LOG10_TEMPERATURE_CACHE = None
    _clear_cache()

# Helper function to check if data is in the global cache
def has_cia_data() -> bool:
    return _CIA_SIGMA_CACHE is not None

# Load the CIA cross section data from the formatted npz files
def _load_cia_npz(index: int, path: str, target_wavelengths: np.ndarray) -> CiaRegistryEntry:

    # Load the table
    data = np.load(path, allow_pickle=True)
    name = data["mol"]
    if isinstance(name, np.ndarray):
        name = name.tolist()
    if not isinstance(name, str):
        name = str(name)

    # Get the temperature array, wavenumbers and cross-sections
    temperatures = np.asarray(data["T"], dtype=float)
    wn = np.asarray(data["wn"], dtype=float)
    xs = np.asarray(data["sig"], dtype=float)
    if not np.all(np.isfinite(xs)):
        bad = np.where(~np.isfinite(xs))
        print(f"[warn] Non-finite CIA xs in {path}: count={bad[0].size}")

    # Convert to wavelength and inverse array
    native_wavelengths = 1.0e4 / wn[::-1]
    native_xs = xs[:, ::-1]

    target_wavelengths = np.asarray(target_wavelengths, dtype=float)
    if target_wavelengths.ndim != 1:
        raise ValueError(f"lam_target must be 1D, got shape {target_wavelengths.shape} for {path}")
    lam_min, lam_max = float(target_wavelengths[0]), float(target_wavelengths[-1])
    wl_min, wl_max = float(native_wavelengths.min()), float(native_wavelengths.max())
    if lam_min < wl_min or lam_max > wl_max:
        print(
            "[warn] Target wavelength grid "
            f"[{lam_min:.6g}, {lam_max:.6g}] extends beyond native CIA grid "
            f"[{wl_min:.6g}, {wl_max:.6g}] in {path}; "
            "filling out-of-range σ with 1e-199."
        )

    # Interpolate to the master wavelength grid
    # Use float64 for log10 cross sections to keep dtype consistent.
    n_temperatures, _ = native_xs.shape
    wavelength_count = target_wavelengths.size
    xs_interp = np.empty((n_temperatures, wavelength_count), dtype=np.float64)
    for idx_temp in range(n_temperatures):
        xs_interp[idx_temp, :] = np.interp(target_wavelengths, native_wavelengths, native_xs[idx_temp, :], left=-199.0, right=-199.0)
    xs_interp = np.maximum(xs_interp, -199.0)

    # Return a CIA table registry entry with NumPy arrays (will be converted to JAX later)
    # Float64 for grids and cross sections.
    return CiaRegistryEntry(
        name=name,
        idx=index,
        temperatures=temperatures.astype(np.float64),
        wavelengths=target_wavelengths.astype(np.float64),
        cross_sections=xs_interp,
    )

# Pad the tables to a rectangle (in dimension) - usually only in T as wavelength grids are the same lengths
# Uses NumPy for preprocessing (CPU-based padding before sending to device)
def _rectangularize_entries(entries: List[CiaRegistryEntry]) -> Tuple[CiaRegistryEntry, ...]:
    if not entries:
        return ()
    base_wavelengths = entries[0].wavelengths
    expected_wavelengths = base_wavelengths.shape[0]
    for entry in entries[1:]:
        if entry.wavelengths.shape != base_wavelengths.shape or not np.allclose(entry.wavelengths, base_wavelengths):
            raise ValueError(f"CIA wavelength grids differ between {entries[0].name} and {entry.name}.")
    max_temperatures = max(entry.temperatures.shape[0] for entry in entries)
    padded_entries: List[CiaRegistryEntry] = []
    for entry in entries:
        # Keep as NumPy arrays for preprocessing
        temperatures = entry.temperatures
        xs = entry.cross_sections
        n_temperatures, wavelength_count = xs.shape
        if wavelength_count != expected_wavelengths:
            raise ValueError(f"Species {entry.name} has λ grid length {wavelength_count}, expected {expected_wavelengths}.")
        pad_temperatures = max_temperatures - n_temperatures
        if pad_temperatures > 0:
            # Use NumPy padding (CPU-based)
            temperatures = np.pad(temperatures, (0, pad_temperatures), mode="edge")
            xs = np.pad(xs, ((0, pad_temperatures), (0, 0)), mode="edge")
        padded_entries.append(
            CiaRegistryEntry(
                name=entry.name,
                idx=entry.idx,
                temperatures=temperatures,
                wavelengths=base_wavelengths,
                cross_sections=xs,
            )
        )
    return tuple(padded_entries)

def _build_hminus_cia_entry(index: int, target_wavelengths: np.ndarray, spec) -> CiaRegistryEntry:
    lam = np.asarray(target_wavelengths, dtype=float)

    # Rectangularise to the expected H2-He grid size
    nT = 334
    T = np.linspace(100.0, 6000.0, nT, dtype=float)

    floor = -199.0

    # Base validity window from configured constants
    lam_min_base = float(lam_min)   # module constant
    lam0_base = float(lam_0)        # module constant
    valid = (lam >= lam_min_base) & (lam <= lam0_base)

    # Start with everything floored in log10 space
    # Use float64 for log10 cross sections to keep dtype consistent.
    log10_sigma = np.full((nT, lam.size), floor, dtype=np.float64)

    if np.any(valid):
        lam_v = lam[valid]
        base = (1.0 / lam_v) - (1.0 / lam0_base)  # >= 0 in valid region

        # fbf(lam) = sum_{n=1..6} Cn_bf[n-1] * base^((n-1)/2)
        fbf = np.zeros_like(lam_v, dtype=float)
        for n in range(1, 7):
            fbf += Cn_bf[n - 1] * (base ** ((n - 1) / 2.0))

        # xbf linear
        xbf_v = 1.0e-18 * (lam_v ** 3) * (base ** 1.5) * fbf

        # Convert to log10 safely and broadcast across T (this snippet is λ-only)
        with np.errstate(divide="ignore", invalid="ignore"):
            log10_v = np.where(xbf_v > 0.0, np.log10(xbf_v), floor).astype(np.float64)

        log10_sigma[:, valid] = log10_v[None, :]

    # enforce floor
    log10_sigma = np.maximum(log10_sigma, floor)

    # Return NumPy arrays (will be converted to JAX later)
    # Float64 for grids and cross sections.
    return CiaRegistryEntry(
        name="H-",
        idx=index,
        temperatures=T.astype(np.float64),
        wavelengths=lam.astype(np.float64),
        cross_sections=log10_sigma,
    )


# Load in the CIA table data - add the data to global scope cache files
def load_cia_registry(cfg, obs, lam_master: Optional[np.ndarray] = None, base_dir: Optional[Path] = None) -> None:

    # Initialise the global caches
    global _CIA_SPECIES_NAMES, _CIA_SIGMA_CACHE, _CIA_TEMPERATURE_CACHE, _CIA_WAVELENGTH_CACHE, _CIA_LOG10_TEMPERATURE_CACHE
    entries: List[CiaRegistryEntry] = []
    config = getattr(cfg.opac, "cia", None)
    if not config:
        reset_registry()
        return
    
    # Use observational wavelengths if no master wavelength grid is availialbe
    wavelengths = np.asarray(obs["wl"], dtype=float) if lam_master is None else np.asarray(lam_master, dtype=float)

    # Read in each CIA table data
    for index, spec in enumerate(cfg.opac.cia):
        name = getattr(spec, "species", spec)
        if name == 'H-':
            print("[CIA] Computing CIA xs for", name, "on master grid")
            entry = _build_hminus_cia_entry(index, wavelengths, spec)
        else:
            cia_path = Path(spec.path).expanduser()
            if not cia_path.is_absolute():
                if base_dir is not None:
                    cia_path = (Path(base_dir) / cia_path).resolve()
                else:
                    cia_path = cia_path.resolve()
            path_str = str(cia_path)
            print("[CIA] Reading cia xs for", name, "@", path_str)
            entry = _load_cia_npz(index, path_str, wavelengths)
        entries.append(entry)

    # For JAX, need to pad to make the tables rectangular with the same nummber of T grids
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
    # Float64 for grids, float32 for cross sections.
    # ============================================================================

    print(f"[CIA] Transferring {len(rectangularized_entries)} species to device...")

    # Stack cross sections: (n_species, nT, nwl) - already float64 from preprocessing
    sigma_stacked = np.stack([entry.cross_sections for entry in rectangularized_entries], axis=0)
    _CIA_SIGMA_CACHE = jnp.asarray(sigma_stacked, dtype=jnp.float32)

    # Stack temperature grids: (n_species, nT) - keep as float64 for accuracy
    temp_stacked = np.stack([entry.temperatures for entry in rectangularized_entries], axis=0)
    _CIA_TEMPERATURE_CACHE = jnp.asarray(temp_stacked, dtype=jnp.float64)

    _CIA_WAVELENGTH_CACHE = jnp.asarray(rectangularized_entries[0].wavelengths, dtype=jnp.float64)

    # Pre-compute log10 of temperature grids for efficient interpolation
    _CIA_LOG10_TEMPERATURE_CACHE = jnp.log10(_CIA_TEMPERATURE_CACHE)

    print(f"[CIA] Cross section cache: {_CIA_SIGMA_CACHE.shape} (dtype: {_CIA_SIGMA_CACHE.dtype})")
    print(f"[CIA] Temperature cache: {_CIA_TEMPERATURE_CACHE.shape} (dtype: {_CIA_TEMPERATURE_CACHE.dtype})")
    print(f"[CIA] Cached log10(T) grids for efficient interpolation")

    # Estimate memory usage
    sigma_mb = _CIA_SIGMA_CACHE.size * _CIA_SIGMA_CACHE.itemsize / 1024**2
    temp_mb = _CIA_TEMPERATURE_CACHE.size * _CIA_TEMPERATURE_CACHE.itemsize / 1024**2
    total_mb = sigma_mb + temp_mb
    print(f"[CIA] Estimated device memory: {total_mb:.1f} MB (σ: {sigma_mb:.1f} MB, T: {temp_mb:.1f} MB)")

    # Extract species names (lightweight: just strings)
    _CIA_SPECIES_NAMES = tuple(entry.name for entry in rectangularized_entries)

    # Delete NumPy arrays to free memory (JAX caches now hold the data on device)
    # This saves ~50 MB for typical CIA tables
    del rectangularized_entries, entries, sigma_stacked, temp_stacked
    print(f"[CIA] Freed NumPy temporary arrays from CPU memory")

    _clear_cache()


### -- lru cached helper functions below --- ###


@lru_cache(None)
def cia_species_names() -> Tuple[str, ...]:
    if not _CIA_SPECIES_NAMES:
        raise RuntimeError("CIA registry empty; call build_opacities() first.")
    return _CIA_SPECIES_NAMES


@lru_cache(None)
def cia_master_wavelength() -> jnp.ndarray:
    if _CIA_WAVELENGTH_CACHE is None:
        raise RuntimeError("CIA registry empty; call build_opacities() first.")
    return _CIA_WAVELENGTH_CACHE


@lru_cache(None)
def cia_sigma_cube() -> jnp.ndarray:
    if _CIA_SIGMA_CACHE is None:
        raise RuntimeError("CIA σ cube not built; call build_opacities() first.")
    return _CIA_SIGMA_CACHE


@lru_cache(None)
def cia_temperature_grids() -> jnp.ndarray:
    if _CIA_TEMPERATURE_CACHE is None:
        raise RuntimeError("CIA temperature grids not built; call build_opacities() first.")
    return _CIA_TEMPERATURE_CACHE


@lru_cache(None)
def cia_temperature_grid() -> jnp.ndarray:
    return cia_temperature_grids()[0]


@lru_cache(None)
def cia_log10_temperature_grids() -> jnp.ndarray:
    if _CIA_LOG10_TEMPERATURE_CACHE is None:
        raise RuntimeError("CIA log10(T) grids not built; call build_opacities() first.")
    return _CIA_LOG10_TEMPERATURE_CACHE
