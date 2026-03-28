#!/usr/bin/env python3
"""
posterior_corner.py
====================

Generate a corner-style plot from an ArviZ posterior NetCDF file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml


def _infer_scalar_params(data_vars: Iterable) -> List[str]:
    scalar_names: List[str] = []
    for name, var in data_vars.items():
        dims = tuple(var.dims)
        if dims == ("chain", "draw"):
            scalar_names.append(name)
        else:
            print(f"[posterior_corner] Skipping non-scalar variable '{name}' with dims {dims}")
    return scalar_names


def _resolve_var_names(posterior_ds, requested: Sequence[str] | None) -> List[str]:
    if requested:
        missing = [v for v in requested if v not in posterior_ds.data_vars]
        if missing:
            raise KeyError(f"Variables not found in posterior: {missing}")
        return list(requested)
    names = _infer_scalar_params(posterior_ds.data_vars)
    if not names:
        raise ValueError(
            "No scalar parameters detected automatically. "
            "Provide --params explicitly."
        )
    return names


def _flatten_param(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.reshape(-1)
    if arr.ndim >= 3:
        tail = int(np.prod(arr.shape[2:]))
        if tail == 1:
            return arr.reshape(-1)
        reshaped = arr.reshape(arr.shape[0], arr.shape[1], tail)
        return reshaped[..., 0].reshape(-1)
    raise ValueError(f"Unsupported parameter shape {arr.shape}")


def _build_sample_matrix(
    posterior_ds,
    var_names: List[str],
    log_params: Set[str],
) -> np.ndarray:
    samples = []
    for name in var_names:
        arr = posterior_ds[name].values
        vec = _flatten_param(np.asarray(arr, dtype=float))
        if name in log_params:
            if np.any(vec <= 0):
                raise ValueError(f"Parameter '{name}' has non-positive samples; cannot take log10.")
            vec = np.log10(vec)
        samples.append(vec)
    stacked = np.vstack(samples).T
    if stacked.shape[0] == 0:
        raise ValueError("No posterior samples available to plot.")
    return stacked


def _load_config_data(config_path: Path | None) -> Dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _infer_log_params_from_config(
    cfg: Dict[str, Any],
    candidate_names: Sequence[str],
) -> Set[str]:
    params_cfg = cfg.get("params", [])
    candidate_set = set(candidate_names)
    log_params: Set[str] = set()

    for entry in params_cfg:
        name = entry.get("name")
        if name not in candidate_set:
            continue
        transform = str(entry.get("transform", "")).lower()
        dist = str(entry.get("dist", "")).lower()
        is_log = transform == "log" or dist.startswith("log_") or dist == "lognormal"
        if is_log:
            log_params.add(name)

    return log_params


def _extract_plot_order(
    cfg: Dict[str, Any],
    candidate_names: Sequence[str],
) -> Dict[str, float]:
    params_cfg = cfg.get("params", [])
    candidate_set = set(candidate_names)
    orders: Dict[str, float] = {}
    for entry in params_cfg:
        name = entry.get("name")
        if name not in candidate_set:
            continue
        if "plot_order" not in entry:
            continue
        try:
            orders[name] = float(entry["plot_order"])
        except (TypeError, ValueError):
            continue
    return orders


def _apply_plot_order(var_names: List[str], order_map: Dict[str, float]) -> List[str]:
    ordered = []
    for idx, name in enumerate(var_names):
        order_value = order_map.get(name)
        if order_value is not None and order_value <= 0:
            continue
        sort_bucket = 0 if order_value is not None else 1
        sort_key = (sort_bucket, order_value if order_value is not None else idx, idx)
        ordered.append((sort_key, name))
    ordered.sort(key=lambda item: item[0])
    return [name for _, name in ordered]


def _load_corner_config(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("corner_config must be a mapping of parameter names.")
    out: Dict[str, Dict[str, Any]] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            out[key] = val
        elif isinstance(val, str):
            out[key] = {"label": val}
        else:
            raise ValueError("corner_config entries must be dicts or strings.")
    return out


def _replace_diag_with_kde(
    fig: plt.Figure,
    samples: np.ndarray,
    quantiles: Sequence[float],
    color: str,
) -> None:
    n_params = samples.shape[1]
    try:
        axes = np.array(fig.axes).reshape(n_params, n_params)
    except ValueError:
        print("[posterior_corner] Could not reshape axes grid for KDE replacement.")
        return

    for idx in range(n_params):
        ax = axes[idx, idx]
        xlabel = ax.get_xlabel()
        ax.clear()
        sns.kdeplot(
            samples[:, idx],
            ax=ax,
            color=color,
            linewidth=1.5,
            fill=True,
            alpha=0.25,
            bw_adjust=1.0,
        )
        for q in quantiles:
            val = np.quantile(samples[:, idx], q)
            linestyle = "--" if np.isclose(q, 0.5) else ":"
            ax.axvline(val, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9)

        if len(quantiles) >= 3:
            q_low, q_mid, q_high = np.quantile(samples[:, idx], [quantiles[0], quantiles[1], quantiles[2]])
            title = f"${q_mid:.3f}^{{+{q_high - q_mid:.3f}}}_{{-{q_mid - q_low:.3f}}}$"
            ax.set_title(title, fontsize=12)

        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel(xlabel)
        if idx != n_params - 1:
            ax.set_xticklabels([])


def plot_corner(
    posterior_path: Path,
    params: Sequence[str] | None = None,
    outname: str = "posterior_corner",
    quantiles: Sequence[float] = (0.1585, 0.5, 0.8415),
    config_path: Path | None = None,
    extra_log_params: Sequence[str] | None = None,
    label_map_path: Path | None = None,
    kde_diag: bool = False,
    enforce_label_map: bool = False,
    plot_points: bool = True,
) -> Path:
    if not posterior_path.exists():
        raise FileNotFoundError(f"Could not find posterior file: {posterior_path}")

    idata = az.from_netcdf(posterior_path)
    posterior_ds = idata.posterior
    var_names = _resolve_var_names(posterior_ds, params)

    if config_path is None:
        default_cfg = posterior_path.parent / "retrieval_config.yaml"
        config_path = default_cfg if default_cfg.exists() else None
    config_data = _load_config_data(config_path)

    if label_map_path is None:
        default_label_yaml = posterior_path.parent / "corner_config.yaml"
        label_map_path = default_label_yaml if default_label_yaml.exists() else None
    corner_cfg = _load_corner_config(label_map_path)

    if enforce_label_map and corner_cfg:
        filtered = [name for name in var_names if name in corner_cfg]
        skipped = [name for name in var_names if name not in corner_cfg]
        if skipped:
            print(f"[posterior_corner] Skipping variables not in label map: {', '.join(skipped)}")
        if not filtered:
            raise ValueError("No variables remain after applying label-map filtering.")
        var_names = filtered

    order_map = _extract_plot_order(config_data, var_names)
    label_order_overrides = {
        name: cfg.get("order")
        for name, cfg in corner_cfg.items()
        if isinstance(cfg, dict) and "order" in cfg
    }
    order_map.update({k: v for k, v in label_order_overrides.items() if v is not None})
    var_names = _apply_plot_order(var_names, order_map)
    if not var_names:
        raise ValueError("No parameters remain after applying plot_order filtering.")

    log_params = _infer_log_params_from_config(config_data, var_names)
    if extra_log_params:
        log_params.update(extra_log_params)

    samples = _build_sample_matrix(posterior_ds, var_names, log_params)
    max_points = 5000
    if samples.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(samples.shape[0], max_points, replace=False)
        samples = samples[idx]

    labels = []
    for name in var_names:
        default_label = f"log10 {name}" if name in log_params else name
        labels.append(corner_cfg.get(name, {}).get("label", default_label))
    print(f"[posterior_corner] Plotting variables: {', '.join(labels)}")

    sns.set_theme(style="ticks", palette="colorblind")
    contour_color = sns.color_palette("colorblind", 1)[0]
    scatter_color = "#808080"

    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=quantiles,
        show_titles=True,
        hist_bin_factor=1.2,
        label_kwargs={"fontsize": 14, "labelpad": 8},
        title_kwargs={"fontsize": 14},
        plot_contours=True,
        plot_density=True,
        fill_contours=False,
        contour_kwargs={"colors": [contour_color], "linewidths": 1.6, "linestyles": "solid"},
        levels=[0.393, 0.864, 0.989],
        plot_datapoints=plot_points,
        data_kwargs={"alpha": 0.25, "ms": 2.0, "mew": 0.0, "color": scatter_color} if plot_points else None,
        max_n_ticks=4,
        smooth=0.75,
        color=contour_color,
        labelpad=0.04,
    )
    if fig is None:
        raise RuntimeError("corner.corner returned None; no plot generated.")

    try:
        axes_grid = np.array(fig.axes).reshape(len(var_names), len(var_names))
        for ax in axes_grid.flat:
            ax.tick_params(axis="both", labelsize=12)
        for col, lab in enumerate(labels):
            ax = axes_grid[-1, col]
            ax.set_xlabel(lab, fontsize=14, labelpad=8)
            ax.tick_params(axis="x", labelbottom=True)
    except ValueError:
        pass

    if kde_diag:
        _replace_diag_with_kde(fig, samples, quantiles, contour_color)

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.0525, top=0.985, wspace=0.04, hspace=0.04)

    out_png = posterior_path.parent / f"{outname}.png"
    out_pdf = posterior_path.parent / f"{outname}.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"[posterior_corner] saved:\n  {out_png}\n  {out_pdf}")
    plt.show()
    return out_png


def main():
    ap = argparse.ArgumentParser(description="ArviZ corner plot helper for posterior.nc.")
    ap.add_argument("--posterior", type=str, default="posterior.nc")
    ap.add_argument("--params", nargs="+")
    ap.add_argument("--config", type=str)
    ap.add_argument("--log-params", nargs="+")
    ap.add_argument("--outname", type=str, default="posterior_corner")
    ap.add_argument("--label-map", type=str)
    ap.add_argument("--kde-diag", action="store_true")
    ap.add_argument("--no-points", action="store_true")
    args = ap.parse_args()

    posterior_path = Path(args.posterior).resolve()
    config_path = Path(args.config).resolve() if args.config else None
    label_map_path = Path(args.label_map).resolve() if args.label_map else None
    plot_corner(
        posterior_path,
        params=args.params,
        outname=args.outname,
        config_path=config_path,
        extra_log_params=args.log_params,
        label_map_path=label_map_path,
        kde_diag=args.kde_diag,
        enforce_label_map=args.label_map is not None,
        plot_points=not args.no_points,
    )


if __name__ == "__main__":
    main()
