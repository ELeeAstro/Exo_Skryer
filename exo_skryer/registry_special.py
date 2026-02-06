"""
registry_special.py
===================
Special (non-line, non-Rayleigh, non-CIA) opacity registries.

Currently supported:
- H- bound-free (bf) continuum cross-sections σ_bf(λ, T)
- H- free-free (ff) continuum cross-sections σ_ff(λ, T)

Tables are precomputed on the forward-model master wavelength grid and a fixed
temperature grid, then cached as device arrays for JAX kernels.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .data_constants import kb

__all__ = [
    "reset_registry",
    "has_special_data",
    "load_special_registry",
    "special_master_wavelength",
    "hminus_temperature_grid",
    "hminus_log10_temperature_grid",
    "hminus_bf_log10_sigma_table",
    "hminus_ff_log10_sigma_table",
]


# Coefficients for H- continuum fits.
# Kept as plain Python lists for deterministic values; converted to NumPy as needed.
_CN_BF = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]

_AN_FF1 = [518.1021, 472.2636, -482.2089, 115.5291, 0.0, 0.0]
_BN_FF1 = [-734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0]
_CN_FF1 = [1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0]
_DN_FF1 = [-479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0]
_EN_FF1 = [93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0]
_FN_FF1 = [-6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0]

_AN_FF2 = [0.0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830]
_BN_FF2 = [0.0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170]
_CN_FF2 = [0.0, -2054.2910, 8746.5230, -13651.1050, 8642.9700, -1863.8640]
_DN_FF2 = [0.0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880]
_EN_FF2 = [0.0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880]
_FN_FF2 = [0.0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850]

# Constants used in the bf fit (units consistent with historical H- fit implementation here)
_ALF = 1.439e8
_LAM_0 = 1.6419
_LAM_MIN = 0.125


# Global caches (device arrays)
_SPECIAL_WAVELENGTH_CACHE: jnp.ndarray | None = None
_HM_T_CACHE: jnp.ndarray | None = None
_HM_LOGT_CACHE: jnp.ndarray | None = None
_HM_BF_LOGSIGMA_CACHE: jnp.ndarray | None = None
_HM_FF_LOGSIGMA_CACHE: jnp.ndarray | None = None


def reset_registry() -> None:
    global _SPECIAL_WAVELENGTH_CACHE, _HM_T_CACHE, _HM_LOGT_CACHE
    global _HM_BF_LOGSIGMA_CACHE, _HM_FF_LOGSIGMA_CACHE
    _SPECIAL_WAVELENGTH_CACHE = None
    _HM_T_CACHE = None
    _HM_LOGT_CACHE = None
    _HM_BF_LOGSIGMA_CACHE = None
    _HM_FF_LOGSIGMA_CACHE = None
    _clear_cache()


def has_special_data() -> bool:
    return _HM_BF_LOGSIGMA_CACHE is not None or _HM_FF_LOGSIGMA_CACHE is not None


def _clear_cache() -> None:
    special_master_wavelength.cache_clear()
    hminus_temperature_grid.cache_clear()
    hminus_log10_temperature_grid.cache_clear()
    hminus_bf_log10_sigma_table.cache_clear()
    hminus_ff_log10_sigma_table.cache_clear()


def _special_hminus_flags(cfg) -> Tuple[bool, bool, bool]:
    """Infer whether to enable H- bf/ff special opacity from config.

    Supported config patterns:
    - New:
        cfg.opac.special: iterable of items with species='H-' and optional bf/ff booleans
    - Back-compat:
        cfg.opac.cia includes 'H-' (enables bf only)
    """
    enabled = False
    bf = True
    ff = False

    opac_cfg = getattr(cfg, "opac", None)
    special_cfg = getattr(opac_cfg, "special", None) if opac_cfg is not None else None
    if special_cfg not in (None, "None", "none", False):
        enabled = True
        source = "opac.special"
        # Parse structured special list, if present
        if not isinstance(special_cfg, bool):
            try:
                iterator = iter(special_cfg)
            except TypeError:
                iterator = iter((special_cfg,))
            for item in iterator:
                name = getattr(item, "species", item)
                if str(name).strip() != "H-":
                    continue
                bf = bool(getattr(item, "bf", bf))
                ff = bool(getattr(item, "ff", ff))
                enabled = True
                break
        return enabled, bf, ff

    # Back-compat: if H- is listed under CIA, treat that as enabling bf (only)
    cia_cfg = getattr(opac_cfg, "cia", None) if opac_cfg is not None else None
    if cia_cfg not in (None, "None", "none", False):
        try:
            iterator = iter(cia_cfg)
        except TypeError:
            iterator = iter((cia_cfg,))
        for item in iterator:
            name = getattr(item, "species", item)
            if str(name).strip() == "H-":
                enabled = True
                bf = True
                ff = bool(getattr(item, "ff", ff))
                break

    return enabled, bf, ff


def _build_hminus_temperature_grid() -> np.ndarray:
    # Match previous CIA-based H- grid for backwards consistency
    nT = 100
    return np.linspace(100.0, 6000.0, nT, dtype=np.float64)


def _build_hminus_bf_logsigma_table(lam: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Return log10 σ_bf(T, λ) table on (nT, nwl)."""
    lam = np.asarray(lam, dtype=float)
    T = np.asarray(T, dtype=float)

    floor = -199.0
    log10_sigma = np.full((T.size, lam.size), floor, dtype=np.float64)

    valid = (lam >= float(_LAM_MIN)) & (lam <= float(_LAM_0))
    if not np.any(valid):
        return log10_sigma

    lam_v = lam[valid]
    base = (1.0 / lam_v) - (1.0 / float(_LAM_0))  # >= 0 in valid region

    # fbf(lam) = sum_{n=1..6} Cn_bf[n-1] * base^((n-1)/2)
    fbf = np.zeros_like(lam_v, dtype=float)
    for n in range(1, 7):
        fbf += _CN_BF[n - 1] * (base ** ((n - 1) / 2.0))

    # λ-only part (previous implementation)
    xbf_v = 1.0e-18 * (lam_v**3) * (base**1.5) * fbf

    with np.errstate(divide="ignore", invalid="ignore"):
        log10_v = np.where(xbf_v > 0.0, np.log10(xbf_v), floor).astype(np.float64)

    log10_sigma[:, valid] = log10_v[None, :]
    log10_sigma = np.maximum(log10_sigma, floor)
    # Keep exp10 in range for float32 kernels (10**38 ~ float32 max).
    return np.minimum(log10_sigma, 30.0)


def _build_hminus_ff_logsigma_table(lam: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Return log10 σ_ff_eff(T, λ) table on (nT, nwl).

    This matches the project's Fortran implementation:

      sff = Σ_{n=1..6} T5040^((n+1)/2) * (λ^2*A_n + B_n + C_n/λ + D_n/λ^2 + E_n/λ^3 + F_n/λ^4)
      kff = 1e-29 * sff   [cm^4 dyne^-1]

    In the Python forward model we fold in the extra factor (k_B T), so the
    precomputed table stores:

      σ_ff_eff = kff * k_B T   [cm^2]

    The runtime weighting in `opacity_special.compute_hminus_ff_opacity` is:

      κ_ff = (f_e f_H) * (n_d^2 / ρ) * σ_ff_eff(λ, T)
    """
    lam = np.asarray(lam, dtype=float)
    T = np.asarray(T, dtype=float)

    floor = -199.0
    log10_sigma = np.full((T.size, lam.size), floor, dtype=np.float64)

    lam_safe = np.clip(lam, 1e-12, None)
    T_safe = np.clip(T, 1.0, None)

    # Wavelength regime masks (Fortran conditions)
    m_ff2 = lam >= 0.3645
    m_ff1 = (lam < 0.3645) & (lam > 0.1823)

    T5040 = 5040.0 / T_safe  # (nT,)

    def fill(mask: np.ndarray, An, Bn, Cn, Dn, En, Fn) -> None:
        if not np.any(mask):
            return

        wl = lam_safe[mask]  # (nwl_sub,)
        wl2 = wl**2
        inv1 = 1.0 / wl
        inv2 = inv1**2
        inv3 = inv1**3
        inv4 = inv1**4

        sff = np.zeros((T.size, wl.size), dtype=np.float64)
        for n in range(1, 7):
            p = (n + 1.0) / 2.0
            t_fac = (T5040**p).astype(np.float64)  # (nT,)
            term_wl = (
                wl2 * float(An[n - 1])
                + float(Bn[n - 1])
                + float(Cn[n - 1]) * inv1
                + float(Dn[n - 1]) * inv2
                + float(En[n - 1]) * inv3
                + float(Fn[n - 1]) * inv4
            ).astype(np.float64)  # (nwl_sub,)
            sff = sff + t_fac[:, None] * term_wl[None, :]

        kff = 1.0e-29 * sff
        sigma_eff = kff * (float(kb) * T_safe)[:, None]

        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.where(sigma_eff > 0.0, np.log10(sigma_eff), floor)
        log10_sigma[:, mask] = logs

    fill(m_ff2, _AN_FF2, _BN_FF2, _CN_FF2, _DN_FF2, _EN_FF2, _FN_FF2)
    fill(m_ff1, _AN_FF1, _BN_FF1, _CN_FF1, _DN_FF1, _EN_FF1, _FN_FF1)

    log10_sigma = np.maximum(log10_sigma, floor)
    # Keep exp10 in range for float32 kernels (10**38 ~ float32 max).
    return np.minimum(log10_sigma, 30.0)


def load_special_registry(cfg, obs, lam_master: Optional[np.ndarray] = None, base_dir: Optional[Path] = None) -> None:
    """Load/build special opacity tables and cache them on device."""
    del obs, base_dir

    enabled, bf_on, ff_on = _special_hminus_flags(cfg)
    if not enabled:
        print("[special] No special opacity sources enabled; registry cleared.")
        reset_registry()
        return

    lam = np.asarray(lam_master, dtype=float) if lam_master is not None else None
    if lam is None:
        raise ValueError("load_special_registry requires lam_master.")
    if lam.ndim != 1:
        raise ValueError(f"lam_master must be 1D, got shape {lam.shape}.")

    print("[special] Building special opacity cache on master grid")
    print(f"[special] H- continuum: bf={bool(bf_on)}, ff={bool(ff_on)}")

    T = _build_hminus_temperature_grid()
    logT = np.log10(T)

    bf_table = _build_hminus_bf_logsigma_table(lam, T) if bf_on else None
    ff_table = _build_hminus_ff_logsigma_table(lam, T) if ff_on else None

    global _SPECIAL_WAVELENGTH_CACHE, _HM_T_CACHE, _HM_LOGT_CACHE
    global _HM_BF_LOGSIGMA_CACHE, _HM_FF_LOGSIGMA_CACHE

    print("[special] Transferring special tables to device...")
    _SPECIAL_WAVELENGTH_CACHE = jnp.asarray(lam.astype(np.float64), dtype=jnp.float64)
    _HM_T_CACHE = jnp.asarray(T, dtype=jnp.float64)
    _HM_LOGT_CACHE = jnp.asarray(logT, dtype=jnp.float64)
    _HM_BF_LOGSIGMA_CACHE = None if bf_table is None else jnp.asarray(bf_table, dtype=jnp.float32)
    _HM_FF_LOGSIGMA_CACHE = None if ff_table is None else jnp.asarray(ff_table, dtype=jnp.float32)

    print(f"[special] Master wavelength: {_SPECIAL_WAVELENGTH_CACHE.shape} (dtype: {_SPECIAL_WAVELENGTH_CACHE.dtype})")
    print(f"[special] H- temperature grid: {_HM_T_CACHE.shape} (dtype: {_HM_T_CACHE.dtype})")
    if _HM_BF_LOGSIGMA_CACHE is not None:
        print(f"[special] H- bf log10(σ) table: {_HM_BF_LOGSIGMA_CACHE.shape} (dtype: {_HM_BF_LOGSIGMA_CACHE.dtype})")
    if _HM_FF_LOGSIGMA_CACHE is not None:
        print(f"[special] H- ff log10(σ) table: {_HM_FF_LOGSIGMA_CACHE.shape} (dtype: {_HM_FF_LOGSIGMA_CACHE.dtype})")

    # Estimate memory usage (device arrays)
    total_bytes = 0
    for arr in (_SPECIAL_WAVELENGTH_CACHE, _HM_T_CACHE, _HM_LOGT_CACHE, _HM_BF_LOGSIGMA_CACHE, _HM_FF_LOGSIGMA_CACHE):
        if arr is None:
            continue
        total_bytes += arr.size * arr.itemsize
    print(f"[special] Estimated device memory: {total_bytes / 1024**2:.2f} MB")

    _clear_cache()


@lru_cache(None)
def special_master_wavelength() -> jnp.ndarray:
    if _SPECIAL_WAVELENGTH_CACHE is None:
        raise RuntimeError("Special registry empty; call build_opacities() first.")
    return _SPECIAL_WAVELENGTH_CACHE


@lru_cache(None)
def hminus_temperature_grid() -> jnp.ndarray:
    if _HM_T_CACHE is None:
        raise RuntimeError("Special registry empty; call build_opacities() first.")
    return _HM_T_CACHE


@lru_cache(None)
def hminus_log10_temperature_grid() -> jnp.ndarray:
    if _HM_LOGT_CACHE is None:
        raise RuntimeError("Special registry empty; call build_opacities() first.")
    return _HM_LOGT_CACHE


@lru_cache(None)
def hminus_bf_log10_sigma_table() -> jnp.ndarray:
    if _HM_BF_LOGSIGMA_CACHE is None:
        raise RuntimeError("H- bf table not built/enabled in special registry.")
    return _HM_BF_LOGSIGMA_CACHE


@lru_cache(None)
def hminus_ff_log10_sigma_table() -> jnp.ndarray:
    if _HM_FF_LOGSIGMA_CACHE is None:
        raise RuntimeError("H- ff table not built/enabled in special registry.")
    return _HM_FF_LOGSIGMA_CACHE
