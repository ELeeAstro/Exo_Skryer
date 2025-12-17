"""
registry_ray.py
===============

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
import jax.numpy as jnp


__all__ = [
    "RayRegistryEntry",
    "reset_registry",
    "has_ray_data",
    "load_ray_registry",
    "ray_species_names",
    "ray_master_wavelength",
    "ray_sigma_table",
    "ray_pick_arrays",
]


# Dataclass for the Rayleigh cross section data
# Note: During preprocessing, all arrays are NumPy (CPU)
# They get converted to JAX (device) only at the final cache creation step
# Mixed precision: float64 for wavelengths (better accuracy), float32 for cross sections (memory savings)
@dataclass(frozen=True)
class RayRegistryEntry:
    name: str
    idx: int
    wavelengths: np.ndarray     # NumPy during preprocessing (float64)
    cross_sections: np.ndarray  # NumPy during preprocessing (float32 to save memory)

# Global Rayleigh cross section caches
_RAY_ENTRIES: Tuple[RayRegistryEntry, ...] = ()
_RAY_SIGMA_CACHE: jnp.ndarray | None = None

# Some required constants
PI = np.pi
C_LIGHT = 2.99792458e10
SIGMA_T = 6.6524587321e-25
WL_LY_CM = 121.567e-7
F_LY = C_LIGHT / WL_LY_CM
W_L = (2.0 * PI * F_LY) / 0.75
CP = np.array([1.26537, 3.73766, 8.8127, 19.1515, 39.919, 81.1018, 161.896, 319.001, 622.229, 1203.82])
N_STP_AIR = 2.68678e19
N_STP_2547 = 2.546899e19
N_STP_H2 = 2.65163e19

# Clear cache helper functions
def _clear_cache():
    ray_species_names.cache_clear()
    ray_master_wavelength.cache_clear()
    ray_sigma_table.cache_clear()
    ray_pick_arrays.cache_clear()


# Clear global data helper function
def reset_registry():
    global _RAY_ENTRIES, _RAY_SIGMA_CACHE
    _RAY_ENTRIES = ()
    _RAY_SIGMA_CACHE = None
    _clear_cache()

# Check if Rayleigh data exists helper functions
def has_ray_data() -> bool:
    return bool(_RAY_ENTRIES)

# Functions to calculate index n and King factor (same as gCMCRT)
def _n_func(wn: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    nm1 = A + B / (C - wn**2)
    return nm1 / 1.0e8 + 1.0


def _n_func2(wl_um: np.ndarray, A: float, B: float) -> np.ndarray:
    nm1 = A * (1.0 + B / (wl_um**2))
    return nm1 + 1.0


def _king_from_Dpol_1(Dpol: float) -> float:
    return (6.0 + 3.0 * Dpol) / (6.0 - 7.0 * Dpol)


def _king_from_Dpol_2(Dpol: float) -> float:
    return (3.0 + 6.0 * Dpol) / (3.0 - 4.0 * Dpol)

# Special H Rayleigh scattering calculation
def _sigma_H(freq: np.ndarray, wl_A: np.ndarray) -> np.ndarray:
    w = 2.0 * PI * freq
    wwl = w / W_L
    xsec = np.zeros_like(wl_A)
    mask_low = wwl <= 0.6
    if np.any(mask_low):
        x = wwl[mask_low]
        poly = np.zeros_like(x)
        for p in range(CP.size):
            poly += CP[p] * x**(2 * p)
        xsec[mask_low] = poly * x**4
    mask_high = ~mask_low
    if np.any(mask_high):
        w_h = w[mask_high]
        wb = (w_h - 0.75 * W_L) / (0.75 * W_L)
        term = 1.0 - 1.792 * wb - 23.637 * wb**2 - 83.1393 * wb**3 - 244.1453 * wb**4 - 699.473 * wb**5
        xsec[mask_high] = 0.0433056 / (wb**2) * term
    xsec *= SIGMA_T
    return xsec

# Go through each species and calculate the Rayleigh scattering cross sections (same as gCMCRT)
def _compute_species_sigma(name: str, wl_um: np.ndarray) -> np.ndarray:
    name = name.strip()
    wl_um = np.asarray(wl_um, dtype=float)
    wn = 1.0e4 / wl_um
    wl_A = wl_um * 1.0e4
    wl_cm = wl_um * 1.0e-4
    freq = C_LIGHT / wl_cm
    if name == "H2":
        n = _n_func2(wl_um, 13.58e-5, 7.52e-3)
        King = 1.0
        nd_stp = N_STP_H2
    elif name == "He":
        n = _n_func(wn, 2283.0, 1.8102e13, 1.5342e10)
        King = 1.0
        nd_stp = N_STP_2547
    elif name in ("e-", "el"):
        return np.full_like(wl_um, SIGMA_T)
    elif name == "H":
        return _sigma_H(freq, wl_A)
    elif name == "CO":
        n = _n_func(wn, 22851.0, 0.456e14, 71427.0**2)
        King = 1.0
        nd_stp = N_STP_2547
    elif name == "CO2":
        n = 1.1427e3 * (
            5799.25 / (128908.9**2 - wn**2)
            + 120.05 / (89223.8**2 - wn**2)
            + 5.3334 / (75037.5**2 - wn**2)
            + 4.3244 / (67837.7**2 - wn**2)
            + 0.1218145e-6 / (2418.136**2 - wn**2)
        )
        n = n + 1.0
        King = 1.1364 + 25.3e-12 * wn**2
        nd_stp = N_STP_2547
    elif name == "CH4":
        n = (46662.0 + 4.02e-6 * wn**2) / 1.0e8 + 1.0
        King = 1.0
        nd_stp = N_STP_2547
    elif name == "O2":
        n = _n_func(wn, 20564.8, 2.480899e13, 4.09e9)
        King = 1.09 + 1.385e-11 * wn**2 + 1.448e-20 * wn**4
        nd_stp = N_STP_AIR
    elif name == "N2":
        A_hi, B_hi, C_hi = 5677.465, 318.81874e12, 14.4e9
        A_lo, B_lo, C_lo = 6498.2, 307.4335e12, 14.4e9
        mask_hi = wn > 21360.0
        n = np.empty_like(wn)
        n[mask_hi] = _n_func(wn[mask_hi], A_hi, B_hi, C_hi)
        n[~mask_hi] = _n_func(wn[~mask_hi], A_lo, B_lo, C_lo)
        King = 1.034 + 3.17e-12 * wn
        nd_stp = N_STP_2547
    elif name == "NH3":
        n = _n_func2(wl_um, 37.0e-5, 12.0e-3)
        King = _king_from_Dpol_1(0.0922)
        nd_stp = N_STP_AIR
    elif name == "Ar":
        n = _n_func(wn, 6432.135, 286.06021e12, 14.4e9)
        King = 1.0
        nd_stp = N_STP_2547
    elif name == "N2O":
        n = (46890.0 + 4.12e-6 * wn**2) / 1.0e8 + 1.0
        Dpol = 0.0577 + 11.8e-12 * wn**2
        King = _king_from_Dpol_2(Dpol)
        nd_stp = N_STP_2547
    elif name == "SF6":
        n = (71517.0 + 4.996e-6 * wn**2) / 1.0e8 + 1.0
        King = 1.0
        nd_stp = N_STP_2547
    elif name == "HCl":
        a_vol = 2.515 / (1.0e8**3)
        King = 1.0
        return (128.0 / 3.0) * PI**5 * a_vol**2 * wn**4 * King
    elif name == "HCN":
        a_vol = 2.593 / (1.0e8**3)
        King = 1.0
        return (128.0 / 3.0) * PI**5 * a_vol**2 * wn**4 * King
    elif name == "H2S":
        a_vol = 3.631 / (1.0e8**3)
        King = 1.0
        return (128.0 / 3.0) * PI**5 * a_vol**2 * wn**4 * King
    elif name == "OCS":
        a_vol = 5.090 / (1.0e8**3)
        King = 1.0
        return (128.0 / 3.0) * PI**5 * a_vol**2 * wn**4 * King
    elif name == "SO2":
        a_vol = 3.882 / (1.0e8**3)
        King = 1.0
        return (128.0 / 3.0) * PI**5 * a_vol**2 * wn**4 * King
    elif name == "C2H2":
        a_vol = 3.487 / (1.0e8**3)
        King = 1.0
        return (128.0 / 3.0) * PI**5 * a_vol**2 * wn**4 * King
    elif name == "PH3":
        a_vol = 4.237 / (1.0e8**3)
        King = 1.0
        return (128.0 / 3.0) * PI**5 * a_vol**2 * wn**4 * King
    elif name == "SO3":
        a_vol = 4.297 / (1.0e8**3)
        King = 1.0
        return (128.0 / 3.0) * PI**5 * a_vol**2 * wn**4 * King
    elif name == "H2O":
        raise NotImplementedError("Layer-dependent H2O Rayleigh must be handled in opacity calculations.")
    else:
        raise ValueError(f"Unsupported Rayleigh species '{name}' in ray registry.")
    xsec = ((24.0 * PI**3 * wn**4) / (nd_stp**2)) * (((n**2 - 1.0) / (n**2 + 2.0))**2) * King
    return np.maximum(xsec, 1.0e-99)

# Calculate and set the global Rayleigh cross section data caches
def load_ray_registry(cfg, obs, lam_master: Optional[np.ndarray] = None) -> None:
    global _RAY_ENTRIES, _RAY_SIGMA_CACHE
    entries: List[RayRegistryEntry] = []
    config = getattr(cfg.opac, "ray", None)
    if not config:
        reset_registry()
        return

    wavelengths = np.asarray(obs["wl"], dtype=float) if lam_master is None else np.asarray(lam_master, dtype=float)

    for index, spec in enumerate(cfg.opac.ray):
        name = getattr(spec, "species", str(spec))
        print("[Ray] Computing Rayleigh xs for", name)
        xs = _compute_species_sigma(name, wavelengths)
        log_xs = np.log10(xs)

        # Create entry with NumPy arrays (will be converted to JAX later)
        # Mixed precision: float64 for wavelengths (better accuracy), float32 for cross sections
        entries.append(
            RayRegistryEntry(
                name=name,
                idx=index,
                wavelengths=wavelengths.astype(np.float64),
                cross_sections=log_xs.astype(np.float32),
            )
        )

    _RAY_ENTRIES = tuple(entries)
    if not _RAY_ENTRIES:
        reset_registry()
        return

    # ============================================================================
    # CRITICAL: Convert NumPy arrays to JAX arrays here (ONE transfer to device)
    # ============================================================================
    # All preprocessing is done in NumPy (CPU). Now we send the final data
    # to the device (GPU/CPU as configured) for use in JIT-compiled forward model.
    # Mixed precision strategy:
    #   - float64 for wavelengths → better interpolation accuracy
    #   - float32 for cross sections → halves memory usage
    # ============================================================================

    print(f"[Ray] Transferring {len(_RAY_ENTRIES)} species to device...")

    # Stack cross sections: (n_species, nwl) - already float32 from preprocessing
    sigma_stacked = np.stack([entry.cross_sections for entry in _RAY_ENTRIES], axis=0)
    _RAY_SIGMA_CACHE = jnp.asarray(sigma_stacked, dtype=jnp.float32)

    print(f"[Ray] Cross section cache: {_RAY_SIGMA_CACHE.shape} (dtype: {_RAY_SIGMA_CACHE.dtype})")

    # Estimate memory usage
    sigma_mb = _RAY_SIGMA_CACHE.size * _RAY_SIGMA_CACHE.itemsize / 1024**2
    print(f"[Ray] Estimated device memory: {sigma_mb:.1f} MB")

    _clear_cache()


### -- lru cached helper functions below --- ###

@lru_cache(None)
def ray_species_names() -> Tuple[str, ...]:
    return tuple(entry.name for entry in _RAY_ENTRIES)


@lru_cache(None)
def ray_master_wavelength() -> jnp.ndarray:
    if not _RAY_ENTRIES:
        raise RuntimeError("Rayleigh registry empty; call build_opacities() first.")
    return _RAY_ENTRIES[0].wavelengths


@lru_cache(None)
def ray_sigma_table() -> jnp.ndarray:
    if _RAY_SIGMA_CACHE is None:
        raise RuntimeError("Rayleigh σ table not built; call build_opacities() first.")
    return _RAY_SIGMA_CACHE


@lru_cache(None)
def ray_pick_arrays():
    if not _RAY_ENTRIES:
        raise RuntimeError("Rayleigh registry empty; call build_opacities() first.")
    picks_wavelengths = tuple((lambda _=None, wl=entry.wavelengths: wl) for entry in _RAY_ENTRIES)
    picks_sigma = tuple((lambda _=None, xs=entry.cross_sections: xs) for entry in _RAY_ENTRIES)
    return picks_wavelengths, picks_sigma
