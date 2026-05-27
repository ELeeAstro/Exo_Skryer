#!/usr/bin/env python3
"""
bestfit_chem.py
===============

Plot chemistry profiles (VMR vs pressure) at the median posterior parameters.

This script is intentionally lightweight: it evaluates only the vertical
structure and chemistry kernels (no radiative transfer / opacity convolution).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml


def _to_ns(x):
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    return x


def _read_cfg(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return _to_ns(yaml.safe_load(f))


def _configure_runtime_from_cfg(cfg) -> None:
    runtime_cfg = getattr(cfg, "runtime", None)
    if runtime_cfg is None:
        return
    platform = str(getattr(runtime_cfg, "platform", "cpu")).lower()
    if platform == "gpu":
        cuda_devices = str(getattr(runtime_cfg, "cuda_visible_devices", ""))
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")


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


def _flatten_param(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        return a.reshape(-1)
    if a.ndim >= 3:
        tail = int(np.prod(a.shape[2:]))
        if tail == 1:
            return a.reshape(-1)
        return a.reshape(a.shape[0], a.shape[1], tail)[..., 0].reshape(-1)
    raise ValueError(f"Unsupported parameter shape: {a.shape}")


def _median_params_from_posterior(cfg, posterior_ds) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for p in getattr(cfg, "params", []):
        name = getattr(p, "name", None)
        if not name:
            continue
        if name in posterior_ds.data_vars:
            vec = _flatten_param(posterior_ds[name].values)
            params[name] = float(np.median(vec))
        elif _is_fixed_param(p):
            val = _fixed_value_param(p)
            if val is None:
                raise ValueError(f"Fixed parameter '{name}' requires value/init in YAML.")
            params[name] = float(val)
        else:
            raise KeyError(f"Free parameter '{name}' not found in posterior.")
    return params


def _build_param_draws_from_idata(cfg, posterior_ds):
    """Return dict name->(N_total,) samples, filling fixed params with their constant value."""
    params_cfg = getattr(cfg, "params", [])
    if "chain" not in posterior_ds.dims or "draw" not in posterior_ds.dims:
        raise ValueError("posterior.nc must have dims ('chain', 'draw').")
    N_total = int(posterior_ds.sizes["chain"]) * int(posterior_ds.sizes["draw"])
    out: Dict[str, np.ndarray] = {}
    for p in params_cfg:
        name = getattr(p, "name", None)
        if not name:
            continue
        if name in posterior_ds.data_vars:
            out[name] = _flatten_param(posterior_ds[name].values)
        elif _is_fixed_param(p):
            val = _fixed_value_param(p)
            if val is None:
                raise ValueError(f"Fixed parameter '{name}' requires value/init in YAML.")
            out[name] = np.full((N_total,), float(val), dtype=float)
        else:
            raise KeyError(f"Free parameter '{name}' not found in posterior.")
    return out, N_total


def _build_pressure_grid(nlay: int, p_bot_bar: float, p_top_bar: float, bar_cgs: float):
    p_lev = jnp.logspace(jnp.log10(p_bot_bar * bar_cgs), jnp.log10(p_top_bar * bar_cgs), nlay + 1)
    p_lay = (p_lev[1:] - p_lev[:-1]) / jnp.log(p_lev[1:] / p_lev[:-1])
    return p_lev, p_lay


def _parse_species_arg(species_arg: str | None) -> List[str]:
    if not species_arg:
        return []
    return [s.strip() for s in species_arg.split(",") if s.strip()]


def _default_plot_species(vmr_lay: Dict[str, np.ndarray]) -> List[str]:
    return [key for key in vmr_lay.keys() if not str(key).startswith("__")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to retrieval_config.yaml")
    ap.add_argument("--species", default=None, help="Comma-separated species list (default: config/all)")
    ap.add_argument("--outname", default="bestfit_chem", help="Output stem")
    ap.add_argument("--no-show", action="store_true", help="Do not display plot window")
    ap.add_argument("--max-samples", type=int, default=500,
                    help="Max posterior draws for 1σ bands (0 = use all, default: 500)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for draw subsampling")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    exp_dir = cfg_path.parent
    cfg = _read_cfg(cfg_path)
    _configure_runtime_from_cfg(cfg)

    # Import exo_skryer from repository root.
    repo_root = (exp_dir / "../..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)

    from exo_skryer.data_constants import bar
    from exo_skryer import kernel_registry as KR
    from exo_skryer.build_chem import (
        load_nasa9_if_needed,
        init_fastchem_grid_if_needed,
        init_element_potentials_if_needed,
        init_atmodeller_if_needed,
        prepare_chemistry_kernel,
    )
    from exo_skryer.vert_mu import compute_mu

    posterior_path = exp_dir / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior file: {posterior_path}")
    posterior_ds = az.from_netcdf(posterior_path).posterior

    full_params = _median_params_from_posterior(cfg, posterior_ds)
    nlay = int(getattr(cfg.physics, "nlay", 99))
    p_bot = float(full_params["p_bot"])
    p_top = float(full_params["p_top"])
    p_lev, p_lay = _build_pressure_grid(nlay, p_bot, p_top, bar)

    tp_name = getattr(cfg.physics, "vert_Tp", None) or getattr(cfg.physics, "vert_struct", None)
    Tp_kernel = KR.resolve(tp_name, KR.VERT_TP, "physics.vert_Tp")
    chemistry_kernel = KR.resolve(getattr(cfg.physics, "vert_chem", None), KR.VERT_CHEM, "physics.vert_chem")

    # Initialize chemistry caches.
    load_nasa9_if_needed(cfg, exp_dir)
    init_fastchem_grid_if_needed(cfg, exp_dir)
    init_element_potentials_if_needed(cfg, exp_dir)
    init_atmodeller_if_needed(cfg, exp_dir)

    opacity_schemes = {
        "line_opac": str(getattr(cfg.physics, "opac_line", "none")).lower(),
        "ray_opac": str(getattr(cfg.physics, "opac_ray", "none")).lower(),
        "cia_opac": str(getattr(cfg.physics, "opac_cia", "none")).lower(),
        "special_opac": str(getattr(cfg.physics, "opac_special", "none")).lower(),
    }
    chemistry_kernel, _trace_species = prepare_chemistry_kernel(cfg, chemistry_kernel, opacity_schemes)

    _T_lev, T_lay = Tp_kernel(p_lev, full_params)
    vmr_lay = chemistry_kernel(p_lay, T_lay, full_params, nlay)

    requested_species = _parse_species_arg(args.species)
    if not requested_species:
        ep_cfg = getattr(cfg, "easychem_jax", None)
        if ep_cfg is not None and getattr(ep_cfg, "species", None):
            requested_species = [str(s) for s in ep_cfg.species]
        else:
            requested_species = _default_plot_species(vmr_lay)

    missing = [s for s in requested_species if s not in vmr_lay]
    plot_species = [s for s in requested_species if s in vmr_lay]
    if not plot_species:
        raise ValueError(
            "None of the requested species are present in chemistry output. "
            f"Requested={requested_species}, available={list(vmr_lay.keys())}"
        )
    if missing:
        print(f"[warn] Skipping unavailable species: {missing}")

    p_bar = np.asarray(p_lay / bar)
    mu_lay = np.asarray(vmr_lay["__mu_lay__"]) if "__mu_lay__" in vmr_lay else np.asarray(compute_mu(vmr_lay))

    # --- 1σ bands: sample posterior draws and evaluate chemistry for each ---
    param_draws, N_total = _build_param_draws_from_idata(cfg, posterior_ds)
    rng = np.random.default_rng(args.seed)
    max_s = args.max_samples
    if max_s > 0 and max_s < N_total:
        idx = np.sort(rng.choice(N_total, size=max_s, replace=False))
    else:
        idx = np.arange(N_total)
    M = idx.size

    vmr_samples: Dict[str, np.ndarray] = {sp: np.empty((M, nlay)) for sp in plot_species}
    mu_samples = np.empty((M, nlay))

    print(f"[bestfit_chem] Evaluating chemistry on {M} posterior draws for 1σ bands…")
    for k, ii in enumerate(idx):
        pars = {name: float(param_draws[name][ii]) for name in param_draws}
        _, T_lay_k = Tp_kernel(p_lev, pars)
        vmr_k = chemistry_kernel(p_lay, T_lay_k, pars, nlay)
        for sp in plot_species:
            vmr_samples[sp][k] = np.clip(np.asarray(vmr_k.get(sp, vmr_lay[sp])), 1e-300, 1.0)
        mu_k = np.asarray(vmr_k["__mu_lay__"]) if "__mu_lay__" in vmr_k else np.asarray(compute_mu(vmr_k))
        mu_samples[k] = mu_k

    q1_lo, q1_hi = 0.15865525393145707, 0.8413447460685429
    vmr_q1_lo = {sp: np.quantile(vmr_samples[sp], q1_lo, axis=0) for sp in plot_species}
    vmr_q50   = {sp: np.quantile(vmr_samples[sp], 0.50,   axis=0) for sp in plot_species}
    vmr_q1_hi = {sp: np.quantile(vmr_samples[sp], q1_hi,  axis=0) for sp in plot_species}
    mu_q1_lo  = np.quantile(mu_samples, q1_lo, axis=0)
    mu_q50    = np.quantile(mu_samples, 0.50,   axis=0)
    mu_q1_hi  = np.quantile(mu_samples, q1_hi,  axis=0)

    # --- VMR plot ---
    palette = sns.color_palette("husl", len(plot_species))
    fig_vmr, ax_vmr = plt.subplots(figsize=(8.0, 6.0))
    for sp, color in zip(plot_species, palette):
        ax_vmr.fill_betweenx(p_bar, vmr_q1_lo[sp], vmr_q1_hi[sp],
                             color=color, alpha=0.20)
        ax_vmr.plot(vmr_q50[sp], p_bar, lw=1.8, label=sp, color=color)

    ax_vmr.set_xscale("log")
    ax_vmr.set_yscale("log")
    ax_vmr.invert_yaxis()
    ax_vmr.set_xlabel("VMR")
    ax_vmr.set_ylabel("Pressure [bar]")
    ax_vmr.set_title("Best-fit Chemistry (Posterior Median ± 1σ)")
    ax_vmr.set_xlim(1e-14, 1.0)
    ax_vmr.grid(True, alpha=0.25)
    ax_vmr.legend(fontsize=8, ncol=2)
    fig_vmr.tight_layout()

    # --- Mean molecular weight plot ---
    fig_mu, ax_mu = plt.subplots(figsize=(7.0, 6.0))
    ax_mu.fill_betweenx(p_bar, mu_q1_lo, mu_q1_hi, color="black", alpha=0.20)
    ax_mu.plot(mu_q50, p_bar, lw=2.0, color="black")
    ax_mu.set_yscale("log")
    ax_mu.invert_yaxis()
    ax_mu.set_xlabel("Mean Molecular Weight")
    ax_mu.set_ylabel("Pressure [bar]")
    ax_mu.set_title("Best-fit Mean Molecular Weight (Posterior Median ± 1σ)")
    ax_mu.grid(True, alpha=0.25)
    fig_mu.tight_layout()

    out_vmr_png = exp_dir / f"{args.outname}_vmr.png"
    out_vmr_pdf = exp_dir / f"{args.outname}_vmr.pdf"
    out_mu_png = exp_dir / f"{args.outname}_mu.png"
    out_mu_pdf = exp_dir / f"{args.outname}_mu.pdf"
    fig_vmr.savefig(out_vmr_png, dpi=180)
    fig_vmr.savefig(out_vmr_pdf, dpi=180)
    fig_mu.savefig(out_mu_png, dpi=180)
    fig_mu.savefig(out_mu_pdf, dpi=180)
    print(f"[bestfit_chem] Saved:\n  {out_vmr_png}\n  {out_vmr_pdf}\n  {out_mu_png}\n  {out_mu_pdf}")

    out_npz = exp_dir / f"{args.outname}.npz"
    np.savez_compressed(
        out_npz,
        pressure_bar=p_bar,
        mu=mu_lay,
        species=np.array(plot_species, dtype=object),
        vmr=np.stack([np.asarray(vmr_lay[s]) for s in plot_species], axis=1),
    )
    print(f"[bestfit_chem] Saved data: {out_npz}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
