#!/usr/bin/env python3
"""
plot_Tp_band.py — plot median temperature–pressure profile and 95% credible band.

Expected layout (experiment directory):
  - config.yaml           (your YAML, including physics + params)
  - posterior.nc          (ArviZ InferenceData with posterior group)

Source layout:
  - ../../src_ori/vert_Tp.py

with functions like:
    isothermal(p_lev, params_dict)
    Milne(p_lev, params_dict)
    Guillot(p_lev, params_dict)

This script:
  - reads the YAML config,
  - reads nlay from cfg.physics.nlay,
  - reads p_top and p_bot from params (delta/fixed),
  - builds log-spaced pressure levels between p_top and p_bot,
  - selects T–p model from cfg.physics.vert_stuct (e.g. "Milne"),
  - draws parameter samples from posterior.nc,
  - evaluates T(p) for those samples,
  - computes median and 95% credible interval,
  - plots T vs p with the band shaded.

Outputs (in experiment directory):
  - Tp_band.png / Tp_band.pdf
  - Tp_band_quantiles.npz (T_p02_5, T_p50, T_p97_5, p_lay, p_lev, draw_idx)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
from types import SimpleNamespace
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import arviz as az
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

# ---------------- YAML / cfg helpers ----------------

def _to_ns(x):
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    return x


def _read_cfg(path: Path):
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return _to_ns(y)


def _is_fixed_param(p) -> bool:
    dist = str(getattr(p, "dist", "")).lower()
    return dist == "delta" or bool(getattr(p, "fixed", False))


def _fixed_value_param(p):
    val = getattr(p, "value", None)
    if val is not None:
        return float(val)
    init = getattr(p, "init", None)
    if init is not None:
        return float(init)
    return None


# ---------------- chain handling (ArviZ posterior.nc) ----------------

def _flatten_param(a: np.ndarray) -> np.ndarray:
    """
    Flatten chains/draws to 1D (N,) vector.

    - 2D: (chain, draw) -> flatten
    - 1D: already flattened
    - >=3D: treat as vector/tensor param; take first component politely.
    """
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # (chain, draw)
        return arr.reshape(-1)
    if arr.ndim >= 3:
        # (chain, draw, *tail)
        tail = arr.shape[2:]
        if np.prod(tail) == 1:
            return arr.reshape(-1)
        # Collapse tail and take first component
        return arr.reshape(arr.shape[0], arr.shape[1], -1)[..., 0].reshape(-1)
    raise ValueError(f"Unsupported param shape {arr.shape} for flattening")


def _build_param_draws_from_idata(
    posterior_ds,
    params_cfg: List[SimpleNamespace],
):
    """
    Build dict name->(N_total,) samples for each param in cfg.params, filling
    fixed/delta params from their value/init if missing in posterior.

    posterior_ds: idata.posterior (xarray.Dataset)
    """
    out: Dict[str, np.ndarray] = {}

    if "chain" not in posterior_ds.dims or "draw" not in posterior_ds.dims:
        raise ValueError("posterior.nc must have dims ('chain', 'draw').")
    n_chain = int(posterior_ds.sizes["chain"])
    n_draw = int(posterior_ds.sizes["draw"])
    N_total = n_chain * n_draw

    for p in params_cfg:
        name = getattr(p, "name", None)
        if not name:
            continue

        if name in posterior_ds.data_vars:
            arr = posterior_ds[name].values  # (chain, draw[, ...])
            out[name] = _flatten_param(arr)
        elif _is_fixed_param(p):
            val = _fixed_value_param(p)
            if val is None:
                raise ValueError(f"Fixed/delta param '{name}' needs value/init in YAML.")
            out[name] = np.full((N_total,), float(val), dtype=float)
        else:
            raise KeyError(f"Free parameter '{name}' not found in posterior.nc.")

    lengths = {k: v.shape[0] for k, v in out.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent parameter lengths: {lengths}")
    return out, N_total


# ---------------- T–p grid helpers ----------------

def _get_fixed_param_value_from_cfg(cfg, name: str) -> float:
    """
    Look up a parameter by name in cfg.params and return its fixed/value/init.

    We assume p_top and p_bot are delta/fixed params (as in the YAML example).
    """
    params_cfg = getattr(cfg, "params", [])
    for p in params_cfg:
        if getattr(p, "name", None) == name:
            if not _is_fixed_param(p):
                raise ValueError(
                    f"Parameter '{name}' is not delta/fixed; "
                    "current script assumes fixed p_top/p_bot."
                )
            val = _fixed_value_param(p)
            if val is None:
                raise ValueError(f"Parameter '{name}' has no value/init in YAML.")
            return float(val)
    raise KeyError(f"Parameter '{name}' not found in cfg.params.")


def _build_pressure_grid(cfg):
    """
    Construct pressure levels p_lev and layer-mean p_lay from cfg.

    Uses:
      - cfg.physics.nlay       (number of layers)
      - p_top, p_bot from cfg.params (delta/fixed), in bar
      - log-spaced in pressure between p_top and p_bot
    """
    phys = getattr(cfg, "physics", None)
    if phys is None:
        raise ValueError("cfg.physics missing; please add it to your YAML.")

    n_lay = int(getattr(phys, "nlay", 50))

    # p_top, p_bot in bar from params
    p_top = _get_fixed_param_value_from_cfg(cfg, "p_top")
    p_bot = _get_fixed_param_value_from_cfg(cfg, "p_bot")

    if p_top <= 0.0 or p_bot <= 0.0:
        raise ValueError(f"p_top and p_bot must be > 0, got p_top={p_top}, p_bot={p_bot}")

    # Ensure ordering (top < bottom)
    p_min = min(p_top, p_bot)
    p_max = max(p_top, p_bot)

    # Interface pressures (levels), log-spaced in bar
    p_lev = np.logspace(np.log10(p_max), np.log10(p_min), n_lay + 1) * 1e6

    # Layer-mean pressures (simple arithmetic mean in p)
    p_lay = 0.5 * (p_lev[:-1] + p_lev[1:])

    return n_lay, p_lev, p_lay


def _choose_vert_struct_model(cfg) -> str:
    """
    Decide which vertical structure function to use.

    Uses cfg.physics.vert_Tp (preferred) falling back to vert_stuct/vert_struct.
    """
    phys = getattr(cfg, "physics", None)
    if phys is None:
        raise ValueError("cfg.physics missing; cannot pick T–p model.")

    name = getattr(phys, "vert_Tp", None)
    if not name:
        name = getattr(phys, "vert_stuct", None)
    if not name:
        name = getattr(phys, "vert_struct", None)
    if not name:
        raise ValueError(
            "cfg.physics.vert_Tp (or vert_stuct/vert_struct) missing; "
            "cannot select T–p model."
        )
    return str(name)


# ---------------- main plotting logic ----------------

def plot_Tp_band(
    config_path: str,
    outname: str = "Tp_band",
    max_samples: int | None = 2000,
    random_seed: int = 123,
    show_plot: bool = True,
):
    cfg_path = Path(config_path).resolve()
    exp_dir = cfg_path.parent

    # Add ../../src_ori to sys.path so we can import vertical structure code
    src_root = (exp_dir / "../../src_ori").resolve()
    if src_root.is_dir() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    # Prefer vert_Tp (current home) but allow legacy module names
    vs_mod = None
    last_err = None
    for mod_name in ("vert_Tp", "vert_struct", "phys_vert_struct"):
        try:
            vs_mod = __import__(mod_name)
            break
        except ImportError as exc:
            last_err = exc
    if vs_mod is None:
        raise ImportError(
            f"Could not import any vertical structure module (tried vert_Tp, "
            f"vert_struct, phys_vert_struct) from {src_root}."
        ) from last_err

    try:
        from data_constants import G, M_jup, R_jup
    except ImportError as e:
        raise ImportError(
            f"Could not import data_constants from {src_root}; ensure src_ori is accessible."
        ) from e

    # Load cfg as SimpleNamespace
    cfg = _read_cfg(cfg_path)

    # Build pressure grid from physics + params
    n_lay, p_lev, p_lay = _build_pressure_grid(cfg)

    # Select vertical structure model
    model_name = _choose_vert_struct_model(cfg)
    try:
        vert_struct_fn = getattr(vs_mod, model_name)
    except AttributeError:
        raise AttributeError(
            f"{vs_mod.__name__} has no function '{model_name}'. "
            "Expected one of: isothermal, Milne, Guillot, ..."
        )

    # Load posterior from ArviZ NetCDF
    posterior_path = exp_dir / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(
            f"Missing {posterior_path}. Run the retrieval and save InferenceData first."
        )
    idata = az.from_netcdf(posterior_path)
    posterior_ds = idata.posterior

    # Build param draws (name -> (N_total,))
    params_cfg = getattr(cfg, "params", [])
    if not params_cfg:
        raise ValueError(
            "cfg.params is empty or missing; cannot reconstruct parameters."
        )
    param_draws, N_total = _build_param_draws_from_idata(posterior_ds, params_cfg)

    # Sub-sample draws for model evaluation
    rng = np.random.default_rng(random_seed)
    if max_samples is not None and max_samples > 0 and max_samples < N_total:
        idx = np.sort(rng.choice(N_total, size=max_samples, replace=False))
    else:
        idx = np.arange(N_total)
    M = idx.size

    # Evaluate T(p) for each selected draw
    T_lay_samples = np.empty((M, n_lay), dtype=float)
    T_lev_samples = np.empty((M, n_lay + 1), dtype=float)

    for k, ii in enumerate(idx):
        pars: Dict[str, float] = {}
        for p_cfg in params_cfg:
            name = getattr(p_cfg, "name", None)
            if not name:
                continue
            pars[name] = float(param_draws[name][ii])
        if "log_g" not in pars:
            if "M_p" in pars and "R_p" in pars:
                Mp = pars["M_p"] * M_jup
                Rp = pars["R_p"] * R_jup
                g_val = G * Mp / max(Rp**2, 1e-30)
                pars["log_g"] = float(np.log10(max(g_val, 1e-30)))
            else:
                raise KeyError(
                    "Parameter samples must include 'log_g' or both 'M_p' and 'R_p' "
                    "to reconstruct log_g for T(p) evaluation."
                )

        T_lev, T_lay = vert_struct_fn(p_lev, pars)
        T_lev = np.asarray(T_lev, dtype=float)
        T_lay = np.asarray(T_lay, dtype=float)
        if T_lev.shape[0] != p_lev.shape[0]:
            raise ValueError(
                f"{model_name} returned {T_lev.shape[0]} level temps but "
                f"{p_lev.shape[0]} pressures were requested."
            )
        if T_lay.shape[0] != n_lay:
            raise ValueError(
                f"{model_name} returned {T_lay.shape[0]} layer temps but "
                f"{n_lay} layers were requested."
            )
        T_lev_samples[k, :] = T_lev
        T_lay_samples[k, :] = T_lay

    # Pointwise quantiles in T at each layer
    T_lay_q02_5 = np.quantile(T_lay_samples, 0.025, axis=0)
    T_lay_q50   = np.quantile(T_lay_samples, 0.50,  axis=0)
    T_lay_q97_5 = np.quantile(T_lay_samples, 0.975, axis=0)

    T_lev_q02_5 = np.quantile(T_lev_samples, 0.025, axis=0)
    T_lev_q50   = np.quantile(T_lev_samples, 0.50,  axis=0)
    T_lev_q97_5 = np.quantile(T_lev_samples, 0.975, axis=0)

    # Save quantiles
    np.savez_compressed(
        exp_dir / f"{outname}_quantiles.npz",
        p_lev=p_lev,
        p_lay=p_lay,
        T_lay_q02_5=T_lay_q02_5,
        T_lay_q50=T_lay_q50,
        T_lay_q97_5=T_lay_q97_5,
        T_lev_q02_5=T_lev_q02_5,
        T_lev_q50=T_lev_q50,
        T_lev_q97_5=T_lev_q97_5,
        draw_idx=idx,
    )

    # Plot T–p profile with credible band
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", 2)
    fig, ax = plt.subplots(figsize=(5, 6))

    # Shaded 95% credible band
    ax.fill_betweenx(
        p_lay/1e6,
        T_lay_q02_5,
        T_lay_q97_5,
        alpha=0.3,
        label="95% credible band",
        color=palette[0],
    )
    # Median profile (acts as "best-fit" summary)
    ax.plot(T_lay_q50, p_lay/1e6, lw=2, label="Median T(p)", color=palette[1])

    ax.set_yscale("log")
    ax.invert_yaxis()  # low pressures at top

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title(f"T–p structure: {model_name}")

    ax.legend()
    fig.tight_layout()

    png = exp_dir / f"{outname}.png"
    pdf = exp_dir / f"{outname}.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"[Tp_band] saved:\n  {png}\n  {pdf}")

    if show_plot:
        plt.show()

    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", required=True,
        help="Path to YAML config used for the run (in the experiment directory).",
    )
    ap.add_argument(
        "--outname", type=str, default="Tp_band",
        help="Output filename stem (default: Tp_band)",
    )
    ap.add_argument(
        "--max-samples", type=int, default=2000,
        help="Evaluate T(p) on at most this many posterior draws "
             "(<=0 means use all draws).",
    )
    ap.add_argument(
        "--seed", type=int, default=123,
        help="Random seed for subsampling posterior draws.",
    )
    ap.add_argument(
        "--no-show", action="store_true",
        help="Do not display the plot window (useful on headless systems).",
    )
    args = ap.parse_args()

    max_samples = None if args.max_samples <= 0 else args.max_samples

    plot_Tp_band(
        args.config,
        outname=args.outname,
        max_samples=max_samples,
        random_seed=args.seed,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
