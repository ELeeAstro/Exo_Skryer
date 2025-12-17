"""
help_io.py
==========
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import arviz as az
import json

# --------- internal: normalise shapes to (chains, draws) ----------

__all__ = [
    "to_inferencedata",
    "save_inferencedata",
    "save_observed_data_csv"
]


def _to_chains_draws(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr  # (chains, draws)
    if arr.ndim == 1:
        return arr[None, :]  # (1, draws)
    raise ValueError(f"Expected 1D/2D per-parameter array, got {arr.shape}")


def _is_fixed(p) -> bool:
    # Fixed ≡ dist='delta' (you don't really use 'fixed' in YAML anymore)
    return str(getattr(p, "dist", "")).lower() == "delta" or bool(getattr(p, "fixed", False))


# --------- main: build InferenceData from your samples_dict ---------

def to_inferencedata(
    samples_dict: Dict[str, np.ndarray],
    cfg,
    include_fixed: bool = False,
) -> az.InferenceData:
    """
    Convert your {param: array} mapping into ArviZ InferenceData.

    Parameters
    ----------
    samples_dict : dict
        Mapping name -> samples. Each value can be shaped (draws,) or (chains, draws).
        Typically this is what comes back from your sampler (after JAX → NumPy).
    cfg : SimpleNamespace
        Parsed YAML configuration. Expects `cfg.params` as a list of parameter objects
        with at least attributes: `name`, `dist`, and optionally `init` / `value`.
    include_fixed : bool, default False
        If True, include delta/fixed parameters in the posterior group.
        If False, skip them.

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData with a `posterior` group.
    """
    posterior: Dict[str, np.ndarray] = {}

    params_cfg = getattr(cfg, "params", None)
    if params_cfg is None:
        raise ValueError("cfg.params must be defined in your YAML configuration.")

    for p in params_cfg:
        name = p.name
        if name not in samples_dict:
            # If we want fixed params and they are not in samples_dict, fabricate them
            if include_fixed and _is_fixed(p):
                val = getattr(p, "value", getattr(p, "init", None))
                if val is not None:
                    # minimal (1,1); ArviZ will handle broadcasting as needed
                    posterior[name] = np.asarray([[float(val)]], dtype=float)
            continue

        arr = _to_chains_draws(np.asarray(samples_dict[name], dtype=float))
        if (not include_fixed) and _is_fixed(p):
            # skip fixed/delta if requested
            continue

        posterior[name] = arr

    if not posterior:
        raise ValueError("No variables to export in posterior group.")

    idata = az.from_dict(posterior=posterior)

    # attach minimal attrs so you remember how many chains/draws you had
    c, d = next(iter(posterior.values())).shape
    idata.attrs["chains"] = int(c)
    idata.attrs["draws"] = int(d)

    # optional: store some config meta-info if present
    # (you can tweak these to whatever you like)
    model_name = getattr(cfg, "model_name", None)
    if model_name is None:
        # fall back to something from physics section if useful
        phys = getattr(cfg, "physics", None)
        if phys is not None:
            model_name = getattr(phys, "vert_stuct", "unknown_model")
        else:
            model_name = "unknown_model"

    idata.attrs["model_name"] = model_name

    # you could also stash the sampler engine if present
    sampling = getattr(cfg, "sampling", None)
    if sampling is not None:
        engine = getattr(sampling, "engine", None)
        if engine is not None:
            idata.attrs["sampling_engine"] = engine

    return idata


def save_inferencedata(
    idata: "az.InferenceData",
    outdir: Path,
    stem: str = "posterior",
    make_summary_csv: bool = True,
    evidence: dict | None = None,
) -> Path:
    """
    Save an InferenceData object to NetCDF (and optionally summary CSV + evidence JSON).
    """
    # Attach evidence into attrs (if provided) *before* saving
    if evidence is not None:
        for key, val in evidence.items():
            try:
                idata.attrs[key] = float(val)
            except Exception:
                idata.attrs[key] = val

    out_nc = outdir / f"{stem}.nc"
    az.to_netcdf(idata, out_nc)

    if make_summary_csv:
        summ = az.summary(
            idata,
            var_names=list(idata.posterior.data_vars),
            stat_funcs=None,
            round_to=None,
            kind="all",
        )
        csv_path = outdir / "arviz_summary.csv"
        summ.to_csv(csv_path)

    # Optional: also dump evidence to a small JSON for quick lookup
    if evidence is not None:
        evid_path = outdir / "evidence.json"
        with evid_path.open("w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)

    return out_nc


def save_observed_data_csv(
    outdir: Path,
    lam: np.ndarray,
    dlam: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    response_mode: np.ndarray | None = None,
    stem: str = "observed_data",
) -> Path:
    """
    Save observed data as CSV with columns: lam,dlam,y,dy
    Values are written with full float precision; NaNs are allowed.
    """
    import csv

    lam  = np.asarray(lam)
    dlam = np.asarray(dlam)
    y    = np.asarray(y)
    dy   = np.asarray(dy)

    if not (lam.shape == dlam.shape == y.shape == dy.shape):
        raise ValueError(
            "Observed arrays must have identical shapes; got "
            f"{lam.shape}, {dlam.shape}, {y.shape}, {dy.shape}"
        )

    if response_mode is not None:
        response_mode = np.asarray(response_mode)
        if response_mode.shape != lam.shape:
            raise ValueError(
                "response_mode must match observed shapes; got "
                f"{response_mode.shape} vs {lam.shape}"
            )
    else:
        response_mode = np.full(lam.shape, "", dtype=object)

    out = outdir / f"{stem}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lam_um", "dlam_um", "depth", "depth_sigma", "response_mode"])
        # write with repr precision to avoid losing info
        for a, b, c, d, m in zip(lam, dlam, y, dy, response_mode):
            w.writerow([repr(float(a)), repr(float(b)), repr(float(c)), repr(float(d)), str(m)])
    return out
