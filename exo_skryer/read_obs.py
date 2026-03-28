"""
read_obs.py
===========
"""

from pathlib import Path
from typing import Any, Optional
import numpy as np

__all__ = ['resolve_obs_path', 'read_obs_data']


def _is_missing_obs_path(value: Any) -> bool:
    """Return whether an observation-path config entry should be treated as unset."""
    if value is None:
        return True
    return str(value).strip().lower() in {"", "none", "null", "~"}


def resolve_obs_path(cfg: Any) -> Any:
    """Resolve the observational data path from configuration.

    Parameters
    ----------
    cfg : config object
        Parsed YAML configuration object with `cfg.data.obs` attribute.

    Returns
    -------
    Any
        Observational data spec. Either a single path string (``data.obs``) or a
        dict ``{"east": ..., "west": ...}`` for separate limb spectra.

    Raises
    ------
    ValueError
        If no observational data path is present in the configuration.
    """
    data_cfg = getattr(cfg, "data", None)
    rel_obs_path: Optional[str] = None

    if data_cfg is not None:
        rel_obs_path = getattr(data_cfg, "obs", None)

    obs_east = getattr(data_cfg, "obs_east", None) if data_cfg is not None else None
    obs_west = getattr(data_cfg, "obs_west", None) if data_cfg is not None else None
    has_east = not _is_missing_obs_path(obs_east)
    has_west = not _is_missing_obs_path(obs_west)
    if has_east or has_west:
        if not has_east or not has_west:
            raise ValueError(
                "When using separate limb spectra, both cfg.data.obs_east and "
                "cfg.data.obs_west must be set."
            )
        return {"east": obs_east, "west": obs_west}

    if _is_missing_obs_path(rel_obs_path):
        raise ValueError(
            "No observational data path found. Set cfg.data.obs in the YAML config."
        )

    return rel_obs_path


def _load_columns(path: str) -> np.ndarray:
    '''
      Input: path to observational data file
      Output: raw data from the data file
    '''

    raw = np.genfromtxt(path, dtype=str, comments="#", autostrip=True)
    if raw.ndim == 1:
        raw = raw[None, :]

    return raw


def _resolve_path(path, base_dir=None) -> Path:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        if base_dir is not None:
            path_obj = (Path(base_dir) / path_obj).resolve()
        else:
            path_obj = path_obj.resolve()
    return path_obj


def _read_single_obs_data(path, base_dir=None):
    path_obj = _resolve_path(path, base_dir=base_dir)
    print(f"[info] Reading observational data from: {path_obj}")

    raw = _load_columns(path_obj)
    if raw.shape[1] < 4:
        raise ValueError(f"[error] Observational file '{path}' must have at least four columns (wl,dwl,y,dy).")

    max_numeric = min(raw.shape[1], 5)
    floats = None
    for ncols in range(max_numeric, 3, -1):
        try:
            floats = raw[:, :ncols].astype(float)
            numeric_cols = ncols
            break
        except ValueError:
            continue
    if floats is None:
        raise ValueError(f"[error] Could not parse numeric columns from '{path}'.")

    wl = floats[:, 0]
    dwl = floats[:, 1]
    y = floats[:, 2]
    dy_plus = floats[:, 3]
    if numeric_cols >= 5:
        dy_minus = floats[:, 4]
        dy_sym = np.maximum(dy_plus, dy_minus)
    else:
        dy_minus = dy_plus.copy()
        dy_sym = dy_plus.copy()
    if raw.shape[1] > numeric_cols:
        response_mode = raw[:, numeric_cols]
        print('[info] Using custom wavelength convolution functions for each band')
    else:
        response_mode = np.full(wl.shape, "boxcar", dtype="<U16")
        print('[info] All bands have been defaulted to boxcar convolution')

    offset_group_col = numeric_cols + 1
    if raw.shape[1] > offset_group_col:
        offset_group = raw[:, offset_group_col].astype("<U32")
        unique_groups = np.unique(offset_group)
        print(f'[info] Found {len(unique_groups)} offset groups: {unique_groups.tolist()}')
        for grp in unique_groups:
            mask = offset_group == grp
            wl_min, wl_max = wl[mask].min(), wl[mask].max()
            n_pts = mask.sum()
            print(f'[info]   -> {grp}: {wl_min:.4f} - {wl_max:.4f} um ({n_pts} points)')
        print('[info] Define offset_<group> parameters in YAML to fit instrument offsets')
    else:
        offset_group = np.full(wl.shape, "__no_offset__", dtype="<U32")

    unique_groups, offset_group_idx = np.unique(offset_group, return_inverse=True)
    return {
        "wl": wl,
        "dwl": dwl,
        "y": y,
        "dy": dy_sym,
        "response_mode": response_mode,
        "offset_group": offset_group,
        "offset_group_idx": offset_group_idx,
        "offset_group_names": unique_groups,
    }


def _combine_limb_obs(east_obs: dict, west_obs: dict) -> dict:
    y = np.concatenate([east_obs["y"], west_obs["y"]])
    dy = np.concatenate([east_obs["dy"], west_obs["dy"]])
    wl = np.concatenate([east_obs["wl"], west_obs["wl"]])
    dwl = np.concatenate([east_obs["dwl"], west_obs["dwl"]])
    response_mode = np.concatenate([east_obs["response_mode"], west_obs["response_mode"]])
    offset_group = np.concatenate([east_obs["offset_group"], west_obs["offset_group"]]).astype("<U32")
    unique_groups, offset_group_idx = np.unique(offset_group, return_inverse=True)

    n_east = east_obs["y"].shape[0]
    n_west = west_obs["y"].shape[0]

    return {
        "wl": wl,
        "dwl": dwl,
        "y": y,
        "dy": dy,
        "response_mode": response_mode,
        "offset_group": offset_group,
        "offset_group_idx": offset_group_idx,
        "offset_group_names": unique_groups,
        "has_limb_observations": True,
        "wl_east": east_obs["wl"],
        "dwl_east": east_obs["dwl"],
        "y_east": east_obs["y"],
        "dy_east": east_obs["dy"],
        "response_mode_east": east_obs["response_mode"],
        "offset_group_east": east_obs["offset_group"],
        "wl_west": west_obs["wl"],
        "dwl_west": west_obs["dwl"],
        "y_west": west_obs["y"],
        "dy_west": west_obs["dy"],
        "response_mode_west": west_obs["response_mode"],
        "offset_group_west": west_obs["offset_group"],
        "east_slice": slice(0, n_east),
        "west_slice": slice(n_east, n_east + n_west),
    }


def read_obs_data(path, base_dir=None):
    '''
      Input: path to observational data file
      Output: Dictionary containing observational data
    '''

    if isinstance(path, dict):
        if set(path.keys()) >= {"east", "west"}:
            east_obs = _read_single_obs_data(path["east"], base_dir=base_dir)
            west_obs = _read_single_obs_data(path["west"], base_dir=base_dir)
            print(
                f"[info] Using separate limb observations: east={len(east_obs['y'])} bins, "
                f"west={len(west_obs['y'])} bins"
            )
            return _combine_limb_obs(east_obs, west_obs)
        raise ValueError("Observational dict specs must contain 'east' and 'west' keys.")

    obs_dict = _read_single_obs_data(path, base_dir=base_dir)
    obs_dict["has_limb_observations"] = False
    return obs_dict
