"""
chem_fastchem_grid_jax.py
=========================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

from .data_constants import CHEM_SPECIES_DATA

__all__ = [
    "FastChemGridModel",
    "load_fastchem_grid",
    "resolve_species_indices",
    "interpolate_profile_scan",
    "interpolate_profile_vmap",
]


_REQUIRED_KEYS_LINEAR = (
    "temperature",
    "pressure",
    "M_H",
    "C_O",
    "mixing_ratios",
    "mean_molecular_weight",
    "species_names",
)

_REQUIRED_KEYS_LOG10 = (
    "log10_temperature",
    "log10_pressure",
    "log10_M_H",
    "log10_C_O",
    "log10_mixing_ratios",
    "log10_mean_molecular_weight",
    "species_names",
)


def _decode_species_names(values: np.ndarray) -> tuple[str, ...]:
    out: list[str] = []
    for raw in values:
        if isinstance(raw, (bytes, np.bytes_)):
            out.append(raw.decode("utf-8"))
        else:
            out.append(str(raw))
    return tuple(out)


def _assert_strictly_increasing(name: str, arr: np.ndarray) -> None:
    if arr.ndim != 1:
        raise ValueError(f"FastChem grid axis '{name}' must be 1D; got shape {arr.shape}.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"FastChem grid axis '{name}' must be strictly increasing.")


def _canonical_formula(name: str) -> str:
    """Convert FastChem-like symbol to compact formula (e.g., C1O1 -> CO)."""
    text = str(name).strip()
    if not text:
        return text
    if text in ("e-", "e+"):
        return text
    # Preserve charged-species symbols exactly (e.g., H1-, V1+, Cr+).
    if "+" in text or "-" in text:
        return text

    text = re.sub(r"_[0-9]+$", "", text)  # Drop isotopologue suffixes like _1.
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*)", text)
    if not tokens:
        return text

    parts: list[str] = []
    for elem, count_str in tokens:
        count = int(count_str) if count_str else 1
        if count == 1:
            parts.append(elem)
        else:
            parts.append(f"{elem}{count}")
    return "".join(parts)


def _build_symbol_maps() -> tuple[dict[str, str], dict[str, str]]:
    symbol_to_fastchem: dict[str, str] = {}
    fastchem_to_symbol: dict[str, str] = {}
    for row in CHEM_SPECIES_DATA:
        sym = str(row["symbol"])
        fc = str(row["fastchem_symbol"])
        symbol_to_fastchem[sym] = fc
        fastchem_to_symbol.setdefault(fc, sym)
    return symbol_to_fastchem, fastchem_to_symbol


@dataclass(frozen=True)
class FastChemGridModel:
    temperature: jnp.ndarray
    pressure: jnp.ndarray
    M_H: jnp.ndarray
    C_O: jnp.ndarray
    interp_temperature: jnp.ndarray
    interp_pressure: jnp.ndarray
    interp_M_H: jnp.ndarray
    interp_C_O: jnp.ndarray
    mixing_ratios: jnp.ndarray
    mean_molecular_weight: jnp.ndarray
    use_log_axes: bool
    species_names: tuple[str, ...]
    species_index_raw: dict[str, int]
    species_index_canonical: dict[str, int]
    vmr_interpolator: RegularGridInterpolator
    mmw_interpolator: RegularGridInterpolator


def load_fastchem_grid(npz_path: str | Path) -> FastChemGridModel:
    """Load and validate a FastChem 5D grid NPZ."""
    path = Path(npz_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"FastChem grid file not found: {path}")

    data = np.load(path, allow_pickle=False)
    has_log10 = all(k in data for k in _REQUIRED_KEYS_LOG10)
    has_linear = all(k in data for k in _REQUIRED_KEYS_LINEAR)
    if not has_log10 and not has_linear:
        missing_linear = [k for k in _REQUIRED_KEYS_LINEAR if k not in data]
        missing_log = [k for k in _REQUIRED_KEYS_LOG10 if k not in data]
        raise ValueError(
            "FastChem grid does not contain a supported key set. "
            f"Missing linear keys: {missing_linear}; missing log10 keys: {missing_log}"
        )

    if has_log10:
        log10_T_np = np.asarray(data["log10_temperature"], dtype=np.float64)
        log10_P_np = np.asarray(data["log10_pressure"], dtype=np.float64)
        log10_MH_np = np.asarray(data["log10_M_H"], dtype=np.float64)
        log10_CO_np = np.asarray(data["log10_C_O"], dtype=np.float64)
        log10_vmr_np = np.asarray(data["log10_mixing_ratios"], dtype=np.float64)
        log10_mmw_np = np.asarray(data["log10_mean_molecular_weight"], dtype=np.float64)

        T_np = np.power(10.0, log10_T_np)
        P_np = np.power(10.0, log10_P_np)
        MH_np = np.asarray(log10_MH_np, dtype=np.float64)  # [M/H] is already dex/log10.
        CO_np = np.power(10.0, log10_CO_np)

        interp_T_np = log10_T_np
        interp_P_np = log10_P_np
        interp_MH_np = log10_MH_np
        interp_CO_np = log10_CO_np
        log10_vmr = jnp.asarray(log10_vmr_np, dtype=jnp.float64)
        log10_mmw = jnp.asarray(log10_mmw_np, dtype=jnp.float64)
        use_log_axes = True
    else:
        T_np = np.asarray(data["temperature"], dtype=np.float64)
        P_np = np.asarray(data["pressure"], dtype=np.float64)
        MH_np = np.asarray(data["M_H"], dtype=np.float64)
        CO_np = np.asarray(data["C_O"], dtype=np.float64)
        vmr_np = np.asarray(data["mixing_ratios"], dtype=np.float64)
        mmw_np = np.asarray(data["mean_molecular_weight"], dtype=np.float64)

        vmr_np = np.maximum(vmr_np, 1e-300)
        mmw_np = np.maximum(mmw_np, 1e-300)
        log10_vmr = jnp.log10(jnp.asarray(vmr_np, dtype=jnp.float64))
        log10_mmw = jnp.log10(jnp.asarray(mmw_np, dtype=jnp.float64))
        interp_T_np = T_np
        interp_P_np = P_np
        interp_MH_np = MH_np
        interp_CO_np = CO_np
        use_log_axes = False

    species_names = _decode_species_names(np.asarray(data["species_names"]))

    _assert_strictly_increasing("temperature", T_np)
    _assert_strictly_increasing("pressure", P_np)
    _assert_strictly_increasing("M_H", MH_np)
    _assert_strictly_increasing("C_O", CO_np)
    _assert_strictly_increasing("interp_temperature", interp_T_np)
    _assert_strictly_increasing("interp_pressure", interp_P_np)
    _assert_strictly_increasing("interp_M_H", interp_MH_np)
    _assert_strictly_increasing("interp_C_O", interp_CO_np)

    nT, nP, nM, nC = len(T_np), len(P_np), len(MH_np), len(CO_np)
    expected_vmr_shape = (nT, nP, nM, nC, len(species_names))
    expected_mmw_shape = (nT, nP, nM, nC)
    if log10_vmr.shape != expected_vmr_shape:
        raise ValueError(
            "FastChem mixing_ratios shape mismatch: "
            f"expected {expected_vmr_shape}, got {log10_vmr.shape}"
        )
    if log10_mmw.shape != expected_mmw_shape:
        raise ValueError(
            "FastChem mean_molecular_weight shape mismatch: "
            f"expected {expected_mmw_shape}, got {log10_mmw.shape}"
        )

    T = jnp.asarray(T_np, dtype=jnp.float64)
    P = jnp.asarray(P_np, dtype=jnp.float64)
    MH = jnp.asarray(MH_np, dtype=jnp.float64)
    CO = jnp.asarray(CO_np, dtype=jnp.float64)
    interp_T = jnp.asarray(interp_T_np, dtype=jnp.float64)
    interp_P = jnp.asarray(interp_P_np, dtype=jnp.float64)
    interp_MH = jnp.asarray(interp_MH_np, dtype=jnp.float64)
    interp_CO = jnp.asarray(interp_CO_np, dtype=jnp.float64)

    species_index_raw = {name: i for i, name in enumerate(species_names)}
    species_index_canonical: dict[str, int] = {}
    for i, name in enumerate(species_names):
        species_index_canonical.setdefault(_canonical_formula(name), i)

    axes = (interp_T, interp_P, interp_MH, interp_CO)
    vmr_interp = RegularGridInterpolator(
        axes, log10_vmr, method="linear", bounds_error=False, fill_value=None
    )
    mmw_interp = RegularGridInterpolator(
        axes, log10_mmw, method="linear", bounds_error=False, fill_value=None
    )

    return FastChemGridModel(
        temperature=T,
        pressure=P,
        M_H=MH,
        C_O=CO,
        interp_temperature=interp_T,
        interp_pressure=interp_P,
        interp_M_H=interp_MH,
        interp_C_O=interp_CO,
        mixing_ratios=log10_vmr,
        mean_molecular_weight=log10_mmw,
        use_log_axes=use_log_axes,
        species_names=species_names,
        species_index_raw=species_index_raw,
        species_index_canonical=species_index_canonical,
        vmr_interpolator=vmr_interp,
        mmw_interpolator=mmw_interp,
    )


def resolve_species_indices(
    model: FastChemGridModel,
    species_out: list[str] | tuple[str, ...],
    species_map_override: dict[str, str] | None = None,
) -> tuple[dict[str, int], list[str]]:
    """Resolve output species names to indices in the FastChem grid."""
    overrides = species_map_override or {}
    sym_to_fc, fc_to_sym = _build_symbol_maps()

    resolved: dict[str, int] = {}
    missing: list[str] = []
    for sp in species_out:
        sp_name = str(sp).strip()
        if not sp_name:
            continue

        candidates: list[str] = []
        ov = overrides.get(sp_name)
        if ov:
            candidates.append(str(ov))
        candidates.append(sp_name)
        if sp_name in sym_to_fc:
            candidates.append(sym_to_fc[sp_name])

        found_idx = None
        for cand in candidates:
            idx = model.species_index_raw.get(cand)
            if idx is None:
                idx = model.species_index_canonical.get(_canonical_formula(cand))
            if idx is None:
                idx = model.species_index_raw.get(fc_to_sym.get(cand, ""))
            if idx is not None:
                found_idx = idx
                break

        if found_idx is None:
            missing.append(sp_name)
        else:
            resolved[sp_name] = int(found_idx)

    return resolved, missing


def _build_points(
    model: FastChemGridModel,
    T_lay: jnp.ndarray,
    p_lay_bar: jnp.ndarray,
    metallicity: jnp.ndarray,
    co_ratio: jnp.ndarray,
) -> jnp.ndarray:
    if model.use_log_axes:
        T_work = jnp.log10(jnp.maximum(T_lay, 1e-300))
        P_work = jnp.log10(jnp.maximum(p_lay_bar, 1e-300))
        MH_work = metallicity
        CO_work = jnp.log10(jnp.maximum(co_ratio, 1e-300))
    else:
        T_work = T_lay
        P_work = p_lay_bar
        MH_work = metallicity
        CO_work = co_ratio

    Tq = jnp.clip(T_work, model.interp_temperature[0], model.interp_temperature[-1])
    Pq = jnp.clip(P_work, model.interp_pressure[0], model.interp_pressure[-1])
    MHq = jnp.clip(MH_work, model.interp_M_H[0], model.interp_M_H[-1])
    COq = jnp.clip(CO_work, model.interp_C_O[0], model.interp_C_O[-1])
    MH_vec = jnp.full_like(Tq, MHq)
    CO_vec = jnp.full_like(Tq, COq)
    return jnp.stack((Tq, Pq, MH_vec, CO_vec), axis=-1)


def interpolate_profile_vmap(
    model: FastChemGridModel,
    T_lay: jnp.ndarray,
    p_lay_bar: jnp.ndarray,
    metallicity: jnp.ndarray,
    co_ratio: jnp.ndarray,
    species_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate a profile using one batched interpolator call."""
    points = _build_points(model, T_lay, p_lay_bar, metallicity, co_ratio)
    log10_vmr_all = model.vmr_interpolator(points)  # (nlay, nspecies)
    log10_mmw = model.mmw_interpolator(points)      # (nlay,)
    log10_vmr = jnp.take(log10_vmr_all, species_idx, axis=-1)
    vmr = jnp.power(10.0, log10_vmr)
    mmw = jnp.power(10.0, log10_mmw)
    return vmr, mmw


def interpolate_profile_scan(
    model: FastChemGridModel,
    T_lay: jnp.ndarray,
    p_lay_bar: jnp.ndarray,
    metallicity: jnp.ndarray,
    co_ratio: jnp.ndarray,
    species_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate a profile layer-by-layer with `lax.scan`."""
    points = _build_points(model, T_lay, p_lay_bar, metallicity, co_ratio)

    def _step(_, point):
        point_2d = point[None, :]
        log10_vmr_all = model.vmr_interpolator(point_2d)[0]
        log10_mmw = model.mmw_interpolator(point_2d)[0]
        log10_vmr = jnp.take(log10_vmr_all, species_idx, axis=-1)
        return None, (log10_vmr, log10_mmw)

    _, (log10_vmr_scan, log10_mmw_scan) = jax.lax.scan(_step, None, points)
    vmr = jnp.power(10.0, log10_vmr_scan)
    mmw = jnp.power(10.0, log10_mmw_scan)
    return vmr, mmw
