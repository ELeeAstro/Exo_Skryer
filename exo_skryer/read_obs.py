"""
read_obs.py
===========
"""

from pathlib import Path
from typing import Any, Optional
import numpy as np

__all__ = ['resolve_obs_path', 'read_obs_data']


def resolve_obs_path(cfg: Any) -> str:
    """Resolve the observational data path from configuration.

    Parameters
    ----------
    cfg : config object
        Parsed YAML configuration object with `cfg.data.obs` attribute.

    Returns
    -------
    str
        Observational data path as specified in ``cfg.data.obs``.

    Raises
    ------
    ValueError
        If no observational data path is present in the configuration.
    """
    data_cfg = getattr(cfg, "data", None)
    rel_obs_path: Optional[str] = None

    if data_cfg is not None:
        rel_obs_path = getattr(data_cfg, "obs", None)

    if rel_obs_path is None:
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


def read_obs_data(path, base_dir=None):
    '''
      Input: path to observational data file
      Output: Dictionary containing observational data
    '''

    path_obj = Path(path)
    if not path_obj.is_absolute():
        if base_dir is not None:
            path_obj = (Path(base_dir) / path_obj).resolve()
        else:
            path_obj = path_obj.resolve()

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

    wl = floats[:, 0] # Central wavelengths
    dwl = floats[:, 1] # Wavelength +/- band widths (half width)
    y = floats[:, 2] # Observation y data (usually R_p^2/R_s^2 or F_p/F_s)
    dy_plus = floats[:, 3] # Observation positive error bars
    if numeric_cols >= 5:
        dy_minus = floats[:, 4]
        dy_sym = np.maximum(dy_plus, dy_minus)
    else:
        dy_minus = dy_plus.copy()
        dy_sym = dy_plus.copy()
    if raw.shape[1] > numeric_cols:
        response_mode = raw[:, numeric_cols] # Response function for the wavelength band
        print('[info] Using custom wavelength convolution functions for each band')
    else:
        response_mode = np.full(wl.shape, "boxcar", dtype="<U16") # use boxcar as default
        print('[info] All bands have been defaulted to boxcar convolution')

    # Output dictionary values
    obs_dict = {
        "wl": wl,
        "dwl": dwl,
        "y": y,
        "dy": dy_sym,
        "response_mode": response_mode,
    }

    return obs_dict
