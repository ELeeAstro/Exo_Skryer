#!/usr/bin/env python3
"""
posterior_nautilus.py
=====================

Overview:
    Generate a corner-style plot (pairwise marginalized posterior) from a
    Nautilus nested sampling HDF5 checkpoint file produced by the retrieval
    pipeline.

    Mirrors posterior_corner.py in style and options, but reads from
    nautilus_checkpoint.hdf5 instead of posterior.nc.  The prior is
    reconstructed from the retrieval YAML config so the Nautilus Sampler
    can be re-hydrated from the checkpoint without re-running.

    By default importance weights are passed directly to corner (the native
    Nautilus approach, more accurate than equal-weight resampling).  Use
    --equal-weight to request the resampled posterior instead.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import corner
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Prior / checkpoint helpers
# ---------------------------------------------------------------------------

def _build_prior_from_config(cfg: Dict[str, Any]):
    """
    Reconstruct a Nautilus Prior from the retrieval YAML config dict.
    Returns (prior, param_names) where param_names lists every free parameter
    (delta parameters are skipped, matching build_prior_nautilus in the sampler).
    """
    try:
        from nautilus import Prior
    except ImportError:
        raise ImportError(
            "Nautilus is not installed. Install with: pip install nautilus-sampler"
        )

    prior = Prior()
    param_names: List[str] = []

    for p in cfg.get("params", []):
        name = p.get("name")
        dist = str(p.get("dist", "")).lower()
        if dist == "delta":
            continue

        param_names.append(name)

        if dist == "uniform":
            prior.add_parameter(name, dist=(float(p["low"]), float(p["high"])))
        elif dist in ("gaussian", "normal"):
            prior.add_parameter(
                name, dist=sps.norm(loc=float(p["mu"]), scale=float(p["sigma"]))
            )
        elif dist == "lognormal":
            prior.add_parameter(
                name,
                dist=sps.lognorm(
                    s=float(p["sigma"]), scale=float(np.exp(float(p["mu"])))
                ),
            )
        elif dist == "log_uniform":
            prior.add_parameter(
                name, dist=sps.reciprocal(float(p["low"]), float(p["high"]))
            )
        else:
            raise ValueError(
                f"Unsupported distribution '{dist}' for parameter '{name}'"
            )

    return prior, param_names


def _load_nautilus_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    equal_weight: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[str]]:
    """
    Re-hydrate a Nautilus Sampler from an HDF5 checkpoint file and extract
    the posterior.

    A dummy likelihood is passed to the Sampler constructor; posterior()
    does not call the likelihood so the checkpoint data is returned as-is.

    Returns
    -------
    points     : dict of {param_name: 1D array}
    log_w      : 1D array of log importance weights (uniform if equal_weight)
    log_l      : 1D array of log-likelihood values
    param_names: ordered list of free parameter names
    """
    try:
        from nautilus import Sampler
    except ImportError:
        raise ImportError(
            "Nautilus is not installed. Install with: pip install nautilus-sampler"
        )

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    prior, param_names = _build_prior_from_config(cfg)

    # Dummy likelihood — only needed for the Sampler constructor; never called here.
    def _dummy_likelihood(_theta):
        return 0.0

    sampler = Sampler(
        prior=prior,
        likelihood=_dummy_likelihood,
        filepath=str(checkpoint_path),
        resume=True,
    )

    try:
        points, log_w, log_l = sampler.posterior(
            return_as_dict=True, equal_weight=equal_weight
        )
    except TypeError:
        # Older Nautilus versions without return_as_dict
        points, log_w, log_l = sampler.posterior(equal_weight=equal_weight)
        if not isinstance(points, dict):
            arr = np.asarray(points, dtype=np.float64)
            points = {name: arr[:, i] for i, name in enumerate(param_names)}

    points = {k: np.asarray(v, dtype=np.float64) for k, v in points.items()}
    log_w = np.asarray(log_w, dtype=np.float64)
    log_l = np.asarray(log_l, dtype=np.float64)

    return points, log_w, log_l, param_names


# ---------------------------------------------------------------------------
# Variable / label helpers  (mirrors posterior_corner.py)
# ---------------------------------------------------------------------------

def _resolve_var_names(
    param_names: List[str], requested: Sequence[str] | None
) -> List[str]:
    if requested:
        missing = [v for v in requested if v not in param_names]
        if missing:
            raise KeyError(f"Variables not found in posterior: {missing}")
        return list(requested)
    if not param_names:
        raise ValueError(
            "No free parameters found in checkpoint. "
            "Provide --params explicitly (e.g., --params R_p log_10_g)."
        )
    return list(param_names)


def _convert_clr_to_vmr(
    points: Dict[str, np.ndarray],
    var_names: List[str],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Detect CLR parameters, convert to log10(VMR), and update the points dict.
    Mirrors the xarray version in posterior_corner.py but works on plain dicts.
    """
    clr_names = [v for v in var_names if v.startswith("clr_")]
    if not clr_names:
        return points, var_names

    from exo_skryer.build_chem import clr_samples_to_vmr

    species = [name[4:] for name in clr_names]  # strip "clr_" prefix
    samples_dict = {name: points[name] for name in clr_names}
    derived = clr_samples_to_vmr(samples_dict, species)

    pts = dict(points)
    for clr_name in clr_names:
        log_key = f"log_10_f_{clr_name[4:]}"
        pts[log_key] = derived[log_key]

    new_var_names = []
    for v in var_names:
        if v.startswith("clr_"):
            new_var_names.append(f"log_10_f_{v[4:]}")
        else:
            new_var_names.append(v)

    print(f"[posterior_nautilus] Converted CLR → log10(VMR) for: {', '.join(species)}")
    return pts, new_var_names


def _build_sample_matrix(
    points: Dict[str, np.ndarray],
    var_names: List[str],
    log_params: Set[str],
) -> np.ndarray:
    samples = []
    for name in var_names:
        vec = np.asarray(points[name], dtype=float).ravel()
        if name in log_params:
            if np.any(vec <= 0):
                raise ValueError(
                    f"Parameter '{name}' has non-positive samples; cannot take log10."
                )
            vec = np.log10(vec)
        samples.append(vec)
    stacked = np.vstack(samples).T
    if stacked.shape[0] == 0:
        raise ValueError("No posterior samples available to plot.")
    return stacked


def _load_config_data(config_path: Path | None) -> Dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


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


def _apply_plot_order(
    var_names: List[str], order_map: Dict[str, float]
) -> List[str]:
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
    if path is None:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
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


# ---------------------------------------------------------------------------
# Axis / style helpers  (mirrors posterior_corner.py)
# ---------------------------------------------------------------------------

def _weighted_quantile(
    values: np.ndarray, quantile: float, weights: np.ndarray
) -> float:
    """Compute a weighted quantile via sorted cumulative weights."""
    sorter = np.argsort(values)
    sorted_values = values[sorter]
    cumulative = np.cumsum(weights[sorter])
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, sorted_values))


def _update_histogram_quantile_styles(
    fig: plt.Figure,
    quantiles: Sequence[float],
) -> None:
    """
    Update the linestyles of quantile markers in diagonal histograms.
    Dashed for median, dotted for credible intervals.
    """
    n_params = int(np.sqrt(len(fig.axes)))
    try:
        axes = np.array(fig.axes).reshape(n_params, n_params)
    except ValueError:
        return
    for idx in range(n_params):
        ax = axes[idx, idx]
        for line in ax.get_lines():
            x_data = line.get_xdata()
            if len(x_data) == 2 and x_data[0] == x_data[1]:  # vertical line
                for q in quantiles:
                    if np.isclose(q, 0.5, atol=0.01):
                        line.set_linestyle("--")
                    else:
                        line.set_linestyle(":")


def _replace_diag_with_kde(
    fig: plt.Figure,
    samples: np.ndarray,
    quantiles: Sequence[float],
    color: str,
    weights: np.ndarray | None = None,
) -> None:
    """
    Swap the diagonal histograms for KDE curves while retaining quantile markers.
    Supports importance weights via seaborn's weights parameter.
    """
    n_params = samples.shape[1]
    try:
        axes = np.array(fig.axes).reshape(n_params, n_params)
    except ValueError:
        print("[posterior_nautilus] Could not reshape axes grid for KDE replacement.")
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
            weights=weights,
        )
        for q in quantiles:
            if weights is not None:
                val = _weighted_quantile(samples[:, idx], q, weights)
            else:
                val = float(np.quantile(samples[:, idx], q))
            linestyle = "--" if np.isclose(q, 0.5) else ":"
            ax.axvline(val, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9)

        if len(quantiles) >= 3:
            def _q(frac):
                return (
                    _weighted_quantile(samples[:, idx], frac, weights)
                    if weights is not None
                    else float(np.quantile(samples[:, idx], frac))
                )
            q_low, q_mid, q_high = _q(quantiles[0]), _q(quantiles[1]), _q(quantiles[2])
            title = f"${q_mid:.3f}^{{+{q_high - q_mid:.3f}}}_{{-{q_mid - q_low:.3f}}}$"
            ax.set_title(title, fontsize=12)

        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel(xlabel)
        if idx != n_params - 1:
            ax.set_xticklabels([])


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_corner_nautilus(
    checkpoint_path: Path,
    config_path: Path | None = None,
    params: Sequence[str] | None = None,
    outname: str = "posterior_corner_nautilus",
    quantiles: Sequence[float] = (0.1585, 0.5, 0.8415),  # 1σ for 1D Gaussian
    extra_log_params: Sequence[str] | None = None,
    label_map_path: Path | None = None,
    kde_diag: bool = False,
    enforce_label_map: bool = False,
    equal_weight: bool = False,
) -> Path:
    """
    Load nautilus_checkpoint.hdf5, select variables, and save a classic corner
    plot with histograms, scatter points, and density contours.

    When equal_weight=False (default) importance weights are forwarded to
    corner.corner() directly — the recommended Nautilus approach.
    When equal_weight=True the sampler performs weighted resampling before
    returning, and corner receives unweighted samples.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint file: {checkpoint_path}")

    if config_path is None:
        default_cfg = checkpoint_path.parent / "retrieval_config.yaml"
        config_path = default_cfg if default_cfg.exists() else None
    if config_path is None:
        raise FileNotFoundError(
            "Could not find retrieval_config.yaml. Provide --config explicitly."
        )

    print(f"[posterior_nautilus] Loading checkpoint: {checkpoint_path}")
    points, log_w, log_l, param_names = _load_nautilus_checkpoint(
        checkpoint_path, config_path, equal_weight=equal_weight
    )

    n_total = len(next(iter(points.values()))) if points else 0
    print(f"[posterior_nautilus] Posterior samples: {n_total}")

    # Normalised linear weights for corner (None when equal-weighted)
    if equal_weight:
        weights_plot = None
    else:
        lw = log_w - np.max(log_w)
        weights_plot = np.exp(lw)
        weights_plot /= weights_plot.sum()

    # --- choose variables ---
    var_names = _resolve_var_names(param_names, params)

    # --- CLR → log10(VMR) conversion (if CLR parameters are present) ---
    points, var_names = _convert_clr_to_vmr(points, var_names)

    config_data = _load_config_data(config_path)

    if label_map_path is None:
        default_label_yaml = checkpoint_path.parent / "corner_config.yaml"
        label_map_path = default_label_yaml if default_label_yaml.exists() else None
    corner_cfg = _load_corner_config(label_map_path)

    if enforce_label_map and corner_cfg:
        filtered = [name for name in var_names if name in corner_cfg]
        skipped = [name for name in var_names if name not in corner_cfg]
        if skipped:
            print(
                f"[posterior_nautilus] Skipping variables not in label map: "
                f"{', '.join(skipped)}"
            )
        if not filtered:
            raise ValueError("No variables remain after applying label-map filtering.")
        var_names = filtered

    # plot_order from retrieval config + corner_config.yaml
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

    # --- log10 handling ---
    log_params = _infer_log_params_from_config(config_data, var_names)
    if extra_log_params:
        log_params.update(extra_log_params)
    if log_params:
        print(
            f"[posterior_nautilus] Log-scaling parameters: "
            f"{', '.join(sorted(log_params))}"
        )

    # --- sample matrix & labels ---
    samples = _build_sample_matrix(points, var_names, log_params)

    labels = []
    for name in var_names:
        default_label = f"log10 {name}" if name in log_params else name
        labels.append(corner_cfg.get(name, {}).get("label", default_label))
    print(f"[posterior_nautilus] Plotting variables: {', '.join(labels)}")

    # --- style & corner call ---
    sns.set_theme(style="ticks", palette="colorblind")
    contour_color = sns.color_palette("colorblind", 1)[0]
    scatter_color = "#808080"

    contour_kwargs = {
        "colors": [contour_color],
        "linewidths": 1.6,
        "linestyles": "solid",
    }
    data_kwargs = {
        "alpha": 0.25,
        "ms": 2.0,
        "mew": 0.0,
        "color": scatter_color,
    }

    fig = corner.corner(
        samples,
        weights=weights_plot,
        labels=labels,
        quantiles=quantiles,
        show_titles=True,
        hist_bin_factor=1.2,
        label_kwargs={"fontsize": 14, "labelpad": 8},
        title_kwargs={"fontsize": 14},
        plot_contours=True,
        plot_density=True,
        fill_contours=False,
        contour_kwargs=contour_kwargs,
        levels=[0.393, 0.864],  # 1-sigma and 2-sigma for 2D Gaussian
        plot_datapoints=True,
        data_kwargs=data_kwargs,
        max_n_ticks=4,
        smooth=0.75,
        color=contour_color,
        labelpad=0.04,
    )

    if fig is None:
        raise RuntimeError("corner.corner returned None; no plot generated.")

    # Ensure bottom-row x-labels are set (sometimes corner hides one)
    try:
        axes_grid = np.array(fig.axes).reshape(len(var_names), len(var_names))
        for ax in axes_grid.flat:
            ax.tick_params(axis="both", labelsize=12)
        for col, lab in enumerate(labels):
            ax = axes_grid[-1, col]
            ax.set_xlabel(lab, fontsize=14, labelpad=8)
            ax.tick_params(axis="x", labelbottom=True)
        br_ax = axes_grid[-1, -1]
        br_ax.set_xlabel(labels[-1], fontsize=14, labelpad=8)
        br_ax.tick_params(axis="x", labelbottom=True)
        br_ax.xaxis.label.set_visible(True)
        br_ax.xaxis.set_label_coords(0.5, -0.10)
        for lbl in br_ax.get_xticklabels():
            lbl.set_visible(True)
        if not br_ax.get_xlabel():
            br_ax.set_xlabel(labels[-1], fontsize=14, labelpad=10)
            br_ax.text(
                0.5, -0.15, labels[-1],
                transform=br_ax.transAxes,
                ha="center", va="top", fontsize=14,
            )
    except ValueError:
        pass

    # Update quantile line styles / replace with KDE
    if kde_diag:
        _replace_diag_with_kde(fig, samples, quantiles, contour_color, weights=weights_plot)
    else:
        _update_histogram_quantile_styles(fig, quantiles)

    fig.subplots_adjust(
        left=0.06, right=0.995, bottom=0.085, top=0.97, wspace=0.04, hspace=0.04
    )

    # Per-parameter axis tweaks (optional, mirrors posterior_corner.py)
    try:
        axes = np.array(fig.axes).reshape(len(var_names), len(var_names))
        if "R_p" in var_names:
            idx = var_names.index("R_p")
            diag_ax = axes[idx, idx]
            diag_ax.set_yticklabels([])
            if idx == len(var_names) - 1:
                formatter = plt.FuncFormatter(lambda x, _: f"{x:.2f}")
                diag_ax.xaxis.set_major_formatter(formatter)
            else:
                diag_ax.set_xticklabels([])
    except ValueError:
        pass

    out_png = checkpoint_path.parent / f"{outname}.png"
    out_pdf = checkpoint_path.parent / f"{outname}.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"[posterior_nautilus] saved:\n  {out_png}\n  {out_pdf}")
    plt.show()
    return out_png


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Nautilus corner plot helper for nautilus_checkpoint.hdf5."
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="nautilus_checkpoint.hdf5",
        help=(
            "Path to the Nautilus HDF5 checkpoint file "
            "(default: nautilus_checkpoint.hdf5 in current directory)."
        ),
    )
    ap.add_argument(
        "--config",
        type=str,
        help=(
            "Retrieval YAML config path used to reconstruct the prior and infer "
            "log-scaled parameters (defaults to retrieval_config.yaml next to checkpoint)."
        ),
    )
    ap.add_argument(
        "--params",
        nargs="+",
        help="Specific parameter names to include. Defaults to all free parameters.",
    )
    ap.add_argument(
        "--log-params",
        nargs="+",
        help=(
            "Additional parameter names to plot in log10 space "
            "(applied after YAML inference)."
        ),
    )
    ap.add_argument(
        "--outname",
        type=str,
        default="posterior_corner_nautilus",
        help="Filename stem for outputs (default: posterior_corner_nautilus).",
    )
    ap.add_argument(
        "--label-map",
        type=str,
        help="Path to YAML file mapping param names to custom axis labels.",
    )
    ap.add_argument(
        "--kde-diag",
        action="store_true",
        help=(
            "Replace diagonal histograms with KDE curves "
            "while keeping quantile markers."
        ),
    )
    ap.add_argument(
        "--equal-weight",
        action="store_true",
        help=(
            "Use equal-weight resampling instead of passing importance weights "
            "to corner (less accurate but identical interface to posterior_corner.py)."
        ),
    )
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve() if args.config else None
    label_map_path = Path(args.label_map).resolve() if args.label_map else None
    plot_corner_nautilus(
        checkpoint_path,
        config_path=config_path,
        params=args.params,
        outname=args.outname,
        extra_log_params=args.log_params,
        label_map_path=label_map_path,
        kde_diag=args.kde_diag,
        enforce_label_map=args.label_map is not None,
        equal_weight=args.equal_weight,
    )


if __name__ == "__main__":
    main()
