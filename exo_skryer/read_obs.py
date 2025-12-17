"""
read_obs.py
===========

Overview:
   Reads in the observation data for the retrieval

    - Usage
    - Key Functions
    - Notes
"""

from pathlib import Path
import numpy as np

__all__ = ['read_obs_data']

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
    dy_minus = floats[:, 3] # Observation negative error bars
    if numeric_cols >= 5:
        dy_plus = floats[:, 4]
    else:
        dy_plus = dy_minus.copy()
    if raw.shape[1] > numeric_cols:
        response_mode = raw[:, numeric_cols] # Response function for the wavelength band
        print('[info] Using custom wavelength convolution functions for each band')
    else:
        response_mode = np.full(wl.shape, "boxcar", dtype="<U16") # use boxcar as default
        print('[info] All bands have been defaulted to boxcar convolution')

    # Output dictionary values
    dy_sym = 0.5 * (dy_plus + dy_minus)
    obs_dict = {
        "wl": wl,
        "dwl": dwl,
        "y": y,
        "dy": dy_sym,
        "dy_plus": dy_plus,
        "dy_minus": dy_minus,
        "response_mode": response_mode,
    }

    return obs_dict
