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
import matplotlib.ticker as mticker
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
        np.quantile(samples, 0.15865525393145707, axis=0),
        np.quantile(samples, 0.50, axis=0),
        np.quantile(samples, 0.8413447460685429, axis=0),
        np.quantile(samples, 0.02275013194817921, axis=0),
        np.quantile(samples, 0.9772498680518208, axis=0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="retrieval_config.yaml")
    ap.add_argument("--posterior", default="posterior.nc")
    ap.add_argument("--outname", default="Tp_east_west")
    ap.add_argument("--max-samples", type=int, default=500)
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
    from exo_skryer.limb_asymmetry import merge_limb_parameter_dict, split_limb_parameter_dict

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

    east_samples = np.empty((idx.size, nlay), dtype=float)
    west_samples = np.empty((idx.size, nlay), dtype=float)
    for k, ii in enumerate(idx):
        full_params = {name: float(values[ii]) for name, values in param_draws.items()}
        joint, east, west = split_limb_parameter_dict(full_params)
        params_east = merge_limb_parameter_dict(joint, east)
        params_west = merge_limb_parameter_dict(joint, west)
        _, t_lay_east = tp_kernel(p_lev, params_east)
        _, t_lay_west = tp_kernel(p_lev, params_west)
        east_samples[k] = np.asarray(t_lay_east, dtype=float)
        west_samples[k] = np.asarray(t_lay_west, dtype=float)

    east_q1_lo, east_q50, east_q1_hi, east_q2_lo, east_q2_hi = _quantiles(east_samples)
    west_q1_lo, west_q50, west_q1_hi, west_q2_lo, west_q2_hi = _quantiles(west_samples)
    p_bar = np.asarray(p_lay / bar, dtype=float)

    np.savez_compressed(
        exp_dir / f"{args.outname}_quantiles.npz",
        pressure_bar=p_bar,
        draw_idx=idx,
        east_q1_lo=east_q1_lo,
        east_p50=east_q50,
        east_q1_hi=east_q1_hi,
        east_q2_lo=east_q2_lo,
        east_q2_hi=east_q2_hi,
        west_q1_lo=west_q1_lo,
        west_p50=west_q50,
        west_q1_hi=west_q1_hi,
        west_q2_lo=west_q2_lo,
        west_q2_hi=west_q2_hi,
    )

    palette = sns.color_palette("colorblind", 8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    panels = [
        (axes[0], "East limb", east_q2_lo, east_q2_hi, east_q1_lo, east_q1_hi, east_q50, palette[0], palette[1], palette[3]),
        (axes[1], "West limb", west_q2_lo, west_q2_hi, west_q1_lo, west_q1_hi, west_q50, palette[4], palette[5], palette[6]),
    ]
    for ax, title, q2_lo, q2_hi, q1_lo, q1_hi, q50, color_2s, color_1s, color_med in panels:
        ax.fill_betweenx(p_bar, q2_lo, q2_hi, alpha=0.25, color=color_2s, label=r"2$\sigma$")
        ax.fill_betweenx(p_bar, q1_lo, q1_hi, alpha=0.35, color=color_1s, label=r"1$\sigma$")
        ax.plot(q50, p_bar, lw=2, color=color_med, label="Median")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.invert_yaxis()  # low pressures at top
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(float(p_bar.max()), float(p_bar.min()))
        ax.set_xlabel("Temperature [K]", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(False)
        ax.legend(
            fontsize=12,
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            borderpad=0.8,
            labelspacing=0.6,
            handlelength=2.2,
            handletextpad=0.8,
        )

    axes[0].set_ylabel("Pressure [bar]", fontsize=14)
    fig.tight_layout()

    out_png = exp_dir / f"{args.outname}.png"
    out_pdf = exp_dir / f"{args.outname}.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
