#!/usr/bin/env python3
"""
plot_pymultinest.py
===================

Corner plot from PyMultiNest output files, using the same interface as posterior_corner.py.

Usage:
    python plot_pymultinest.py [options]

Options match posterior_corner.py for consistency.
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Any, Dict, Sequence
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
import yaml


# ============================================================================
# Helper functions (copied from posterior_corner.py)
# ============================================================================

def _load_config_data(config_path: Path | None) -> Dict[str, Any]:
    """Load retrieval config YAML."""
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_corner_config(path: Path | None) -> Dict[str, Dict[str, Any]]:
    """
    Load corner config (label map) YAML.
    Format can be:
        param_name: "LaTeX label"
    OR
        param_name:
            label: "LaTeX label"
            order: 1
    """
    if path is None:
        return {}
    if not path.exists():
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
    """
    Swap the diagonal histograms for KDE curves while retaining quantile markers.
    Matches posterior_corner.py implementation.
    """
    n_params = samples.shape[1]
    try:
        axes = np.array(fig.axes).reshape(n_params, n_params)
    except ValueError:
        print("[plot_pymultinest] Could not reshape axes grid for KDE replacement.")
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
            # Use dashed line for median (0.5), dotted for credible intervals
            linestyle = "--" if np.isclose(q, 0.5) else ":"
            ax.axvline(val, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9)

        # Compute and add title with median and credible intervals
        if len(quantiles) >= 3:
            q_low, q_mid, q_high = np.quantile(samples[:, idx], [quantiles[0], quantiles[1], quantiles[2]])
            title = f"${q_mid:.3f}^{{+{q_high - q_mid:.3f}}}_{{-{q_mid - q_low:.3f}}}$"
            ax.set_title(title, fontsize=12)

        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel(xlabel)
        if idx != n_params - 1:
            ax.set_xticklabels([])


# ============================================================================
# Main plotting function (matches posterior_corner.py interface)
# ============================================================================

def plot_corner(
    pymultinest_path: Path,
    params: Sequence[str] | None = None,
    outname: str = "corner_pymultinest",
    quantiles: Sequence[float] = (0.1585, 0.5, 0.8415),
    config_path: Path | None = None,
    label_map_path: Path | None = None,
    kde_diag: bool = False,
) -> Path:
    """
    Load PyMultiNest equal-weighted samples, select variables, and save a corner plot.

    Args:
        pymultinest_path: Path to pymultinest_post_equal_weights.dat
        params: List of parameter names to plot (None = all)
        outname: Output filename without extension
        quantiles: Quantiles for credible intervals (default: 1σ)
        config_path: Path to retrieval_config.yaml (None = auto-detect)
        label_map_path: Path to corner_config.yaml (None = auto-detect)
        kde_diag: Use KDE on diagonal instead of histograms

    Returns:
        Path to saved figure
    """
    if not pymultinest_path.exists():
        raise FileNotFoundError(f"Could not find PyMultiNest output: {pymultinest_path}")

    print(f"[plot_pymultinest] Loading PyMultiNest samples from {pymultinest_path}...")

    # Read data: columns are [weight, -2*logL, param1, param2, ...]
    data = np.loadtxt(pymultinest_path)
    samples = data[:, 2:]  # Skip weight and -2*logL

    print(f"[plot_pymultinest] Loaded {len(samples)} samples with {samples.shape[1]} parameters")

    # Auto-detect config if not provided
    if config_path is None:
        default_cfg = pymultinest_path.parent / "retrieval_config.yaml"
        config_path = default_cfg if default_cfg.exists() else None
    config_data = _load_config_data(config_path)

    # Auto-detect label map if not provided
    if label_map_path is None:
        default_label_yaml = pymultinest_path.parent / "corner_config.yaml"
        label_map_path = default_label_yaml if default_label_yaml.exists() else None
    corner_cfg = _load_corner_config(label_map_path)

    # Get parameter names from config
    try:
        internal_names = [
            p["name"] for p in config_data.get("params", [])
            if p.get("dist", "").lower() != "delta"
        ]
        if not internal_names:
            raise ValueError("No parameters found in config")
    except Exception as e:
        print(f"[plot_pymultinest] Could not load parameter names from config: {e}")
        internal_names = [f"param_{i}" for i in range(samples.shape[1])]

    # Build labels from corner config
    param_labels = []
    for name in internal_names:
        if name in corner_cfg:
            label = corner_cfg[name].get("label", name)
        else:
            label = name
        param_labels.append(label)

    # Select subset if requested
    if params:
        indices_to_plot = [i for i, name in enumerate(internal_names) if name in params]
        if not indices_to_plot:
            raise ValueError(f"None of the requested params found. Available: {internal_names}")
        samples = samples[:, indices_to_plot]
        param_labels = [param_labels[i] for i in indices_to_plot]
        internal_names = [internal_names[i] for i in indices_to_plot]

    print(f"[plot_pymultinest] Plotting variables: {', '.join(param_labels)}")

    # Load evidence from stats file if available
    logZ = None
    logZ_err = None
    try:
        stats_file = pymultinest_path.parent / "pymultinest_stats.dat"
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            parts = lines[0].split()
            logZ = float(parts[0])
            logZ_err = float(parts[1])
    except Exception as e:
        print(f"[plot_pymultinest] Could not read evidence from stats file: {e}")

    # Print summary statistics
    print("\n[plot_pymultinest] Posterior Summary:")
    if logZ is not None:
        print(f"  logZ = {logZ:.2f} ± {logZ_err:.2f}")
    print(f"  Samples: {len(samples)}")
    print("\nMedian values:")
    for i, name in enumerate(internal_names):
        q_low, q_mid, q_high = np.quantile(samples[:, i], quantiles)
        print(f"  {name:25s}: {q_mid:.3f} +{q_high - q_mid:.3f} -{q_mid - q_low:.3f}")

    # Create corner plot with seaborn theme (matching posterior_corner.py)
    sns.set_theme(style="ticks", palette="colorblind")
    contour_color = sns.color_palette("colorblind", 1)[0]

    fig = corner.corner(
        samples,
        labels=param_labels,
        quantiles=list(quantiles),
        show_titles=True,
        title_fmt=".3f",
        smooth=1.0,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        levels=(0.68, 0.95),
        color=contour_color,
        hist_kwargs={"density": True},
    )

    if fig is None:
        raise RuntimeError("corner.corner returned None; no plot generated.")

    # Replace diagonal with KDE if requested
    if kde_diag:
        _replace_diag_with_kde(fig, samples, quantiles, contour_color)

    # Add title
    title = "PyMultiNest Posterior"
    if logZ is not None:
        title += f" | logZ = {logZ:.2f} ± {logZ_err:.2f}"
    fig.suptitle(title, fontsize=14, y=1.0)

    # Save
    output_path = pymultinest_path.parent / f"{outname}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n[plot_pymultinest] Saved corner plot to {output_path}")
    plt.close()

    return output_path


# ============================================================================
# CLI wrapper
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot PyMultiNest corner (interface matches posterior_corner.py)"
    )
    parser.add_argument(
        "pymultinest_path",
        nargs="?",
        type=Path,
        default=Path("pymultinest_post_equal_weights.dat"),
        help="Path to pymultinest_post_equal_weights.dat (default: pymultinest_post_equal_weights.dat)"
    )
    parser.add_argument(
        "--params",
        nargs="*",
        help="Parameters to plot (default: all)"
    )
    parser.add_argument(
        "--outname",
        type=str,
        default="corner_pymultinest",
        help="Output filename without extension (default: corner_pymultinest)"
    )
    parser.add_argument(
        "--kde-diag",
        action="store_true",
        help="Use KDE on diagonal instead of histograms"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to retrieval_config.yaml (default: auto-detect)"
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Path to corner_config.yaml (default: auto-detect)"
    )
    args = parser.parse_args()

    plot_corner(
        pymultinest_path=args.pymultinest_path,
        params=args.params,
        outname=args.outname,
        quantiles=(0.1585, 0.5, 0.8415),
        config_path=args.config,
        label_map_path=args.label_map,
        kde_diag=args.kde_diag,
    )
