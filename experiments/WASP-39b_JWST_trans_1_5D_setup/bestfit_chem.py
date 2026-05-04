#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict


def _configure_runtime() -> None:
    exp_dir = Path(__file__).resolve().parent
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    mpl_cache_dir = exp_dir / ".mplconfig"
    mpl_cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))


_configure_runtime()

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _build_param_draws_from_posterior(posterior_ds, params_cfg):
    if "chain" not in posterior_ds.dims or "draw" not in posterior_ds.dims:
        raise ValueError("posterior.nc must have dims ('chain', 'draw').")

    n_total = int(posterior_ds.sizes["chain"]) * int(posterior_ds.sizes["draw"])
    out: Dict[str, np.ndarray] = {}
    for p in params_cfg:
        name = p.name
        dist = str(getattr(p, "dist", "")).lower()
        if name in posterior_ds.data_vars:
            out[name] = np.asarray(posterior_ds[name].values, dtype=float).reshape(-1)
        elif dist == "delta":
            out[name] = np.full(n_total, float(getattr(p, "value", getattr(p, "init"))), dtype=float)
        else:
            raise KeyError(f"Free parameter '{name}' missing from posterior.")
    return out, n_total


def _quantiles(samples: np.ndarray):
    return (
        np.quantile(samples, 0.16, axis=0),
        np.quantile(samples, 0.50, axis=0),
        np.quantile(samples, 0.84, axis=0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="retrieval_config.yaml")
    ap.add_argument("--posterior", default="posterior.nc")
    ap.add_argument("--species", default="H2O,CO2,CO,CH4")
    ap.add_argument("--outname", default="bestfit_chem")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    exp_dir = config_path.parent
    repo_root = (exp_dir / "../..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)

    from exo_skryer.read_yaml import read_yaml
    from exo_skryer.data_constants import bar
    from exo_skryer import kernel_registry as KR
    from exo_skryer.build_chem import (
        init_atmodeller_if_needed,
        init_element_potentials_if_needed,
        init_fastchem_grid_if_needed,
        prepare_chemistry_kernel,
    )
    from exo_skryer.limb_asymmetry import merge_limb_parameter_dict, split_limb_parameter_dict
    from exo_skryer.vert_mu import compute_mu

    cfg = read_yaml(config_path)
    posterior_ds = az.from_netcdf(exp_dir / args.posterior).posterior
    param_draws, n_total = _build_param_draws_from_posterior(posterior_ds, cfg.params)

    rng = np.random.default_rng(args.seed)
    max_samples = None if args.max_samples <= 0 else args.max_samples
    if max_samples is not None and max_samples < n_total:
        idx = np.sort(rng.choice(n_total, size=max_samples, replace=False))
    else:
        idx = np.arange(n_total)

    joint0, east0, _ = split_limb_parameter_dict({name: float(values[0]) for name, values in param_draws.items()})
    params_east0 = merge_limb_parameter_dict(joint0, east0)
    nlay = int(getattr(cfg.physics, "nlay", 99))
    p_lev = jnp.logspace(
        jnp.log10(params_east0["p_bot"] * bar),
        jnp.log10(params_east0["p_top"] * bar),
        nlay + 1,
    )
    p_lay = (p_lev[1:] - p_lev[:-1]) / jnp.log(p_lev[1:] / p_lev[:-1])

    tp_name = getattr(cfg.physics, "vert_Tp", None) or getattr(cfg.physics, "vert_struct", None)
    tp_kernel = KR.resolve(tp_name, KR.VERT_TP, "physics.vert_Tp")
    chem_kernel = KR.resolve(getattr(cfg.physics, "vert_chem", None), KR.VERT_CHEM, "physics.vert_chem")

    init_fastchem_grid_if_needed(cfg, None)
    init_element_potentials_if_needed(cfg, None)
    init_atmodeller_if_needed(cfg, None)
    chem_kernel, _ = prepare_chemistry_kernel(
        cfg,
        chem_kernel,
        {
            "line_opac": str(getattr(cfg.physics, "opac_line", "none")).lower(),
            "ray_opac": str(getattr(cfg.physics, "opac_ray", "none")).lower(),
            "cia_opac": str(getattr(cfg.physics, "opac_cia", "none")).lower(),
            "special_opac": str(getattr(cfg.physics, "opac_special", "none")).lower(),
        },
    )

    species = [s.strip() for s in args.species.split(",") if s.strip()]
    p_bar = np.asarray(p_lay / bar, dtype=float)

    east_samples = {sp: np.empty((idx.size, nlay), dtype=float) for sp in species}
    west_samples = {sp: np.empty((idx.size, nlay), dtype=float) for sp in species}
    mu_east_samples = np.empty((idx.size, nlay), dtype=float)
    mu_west_samples = np.empty((idx.size, nlay), dtype=float)

    for k, ii in enumerate(idx):
        full_params = {name: float(values[ii]) for name, values in param_draws.items()}
        joint, east, west = split_limb_parameter_dict(full_params)
        params_east = merge_limb_parameter_dict(joint, east)
        params_west = merge_limb_parameter_dict(joint, west)

        _, t_lay_east = tp_kernel(p_lev, params_east)
        _, t_lay_west = tp_kernel(p_lev, params_west)
        vmr_east = chem_kernel(p_lay, t_lay_east, params_east, nlay)
        vmr_west = chem_kernel(p_lay, t_lay_west, params_west, nlay)

        for sp in species:
            east_samples[sp][k] = np.clip(np.asarray(vmr_east.get(sp, jnp.full(nlay, 1e-300))), 1e-300, 1.0)
            west_samples[sp][k] = np.clip(np.asarray(vmr_west.get(sp, jnp.full(nlay, 1e-300))), 1e-300, 1.0)

        mu_east_samples[k] = np.asarray(vmr_east["__mu_lay__"]) if "__mu_lay__" in vmr_east else np.asarray(compute_mu(vmr_east))
        mu_west_samples[k] = np.asarray(vmr_west["__mu_lay__"]) if "__mu_lay__" in vmr_west else np.asarray(compute_mu(vmr_west))

    palette = sns.color_palette("husl", len(species))
    fig_vmr, axes_vmr = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, title, limb_samples in (
        (axes_vmr[0], "East limb", east_samples),
        (axes_vmr[1], "West limb", west_samples),
    ):
        for sp, color in zip(species, palette):
            q16, q50, q84 = _quantiles(limb_samples[sp])
            ax.fill_betweenx(p_bar, q16, q84, color=color, alpha=0.20)
            ax.plot(q50, p_bar, lw=1.8, label=sp, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_xlabel("VMR", fontsize=14)
        ax.set_title(title)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.25)
    axes_vmr[0].set_ylabel("Pressure [bar]", fontsize=14)
    axes_vmr[1].legend(fontsize=8, ncol=2)
    fig_vmr.tight_layout()

    fig_mu, axes_mu = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for ax, title, mu_samples in (
        (axes_mu[0], "East limb", mu_east_samples),
        (axes_mu[1], "West limb", mu_west_samples),
    ):
        q16, q50, q84 = _quantiles(mu_samples)
        ax.fill_betweenx(p_bar, q16, q84, color="black", alpha=0.20)
        ax.plot(q50, p_bar, lw=2.0, color="black")
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_xlabel("Mean Molecular Weight", fontsize=14)
        ax.set_title(title)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.25)
    axes_mu[0].set_ylabel("Pressure [bar]", fontsize=14)
    fig_mu.tight_layout()

    out_vmr_png = exp_dir / f"{args.outname}_vmr.png"
    out_vmr_pdf = exp_dir / f"{args.outname}_vmr.pdf"
    out_mu_png = exp_dir / f"{args.outname}_mu.png"
    out_mu_pdf = exp_dir / f"{args.outname}_mu.pdf"
    fig_vmr.savefig(out_vmr_png, dpi=180)
    fig_vmr.savefig(out_vmr_pdf, dpi=180)
    fig_mu.savefig(out_mu_png, dpi=180)
    fig_mu.savefig(out_mu_pdf, dpi=180)

    np.savez_compressed(
        exp_dir / f"{args.outname}.npz",
        pressure_bar=p_bar,
        draw_idx=idx,
        species=np.array(species, dtype=object),
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
