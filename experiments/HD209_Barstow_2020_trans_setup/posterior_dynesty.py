#!/usr/bin/env python3
"""
plot_dynesty.py
===============

Corner plot from Dynesty pickle output, using the same interface as posterior_corner.py.

Usage:
    python plot_dynesty.py [options]

Options match posterior_corner.py for consistency.
"""

from __future__ import annotations

from pathlib import Path
import pickle
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
        print("[plot_dynesty] Could not reshape axes grid for KDE replacement.")
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
    dynesty_path: Path,
    params: Sequence[str] | None = None,
    outname: str = "corner_dynesty",
    quantiles: Sequence[float] = (0.1585, 0.5, 0.8415),
    config_path: Path | None = None,
    label_map_path: Path | None = None,
    kde_diag: bool = False,
) -> Path:
    """
    Load Dynesty pickle, select variables, and save a corner plot.

    Args:
        dynesty_path: Path to dynesty_results.pkl
        params: List of parameter names to plot (None = all)
        outname: Output filename without extension
        quantiles: Quantiles for credible intervals (default: 1σ)
        config_path: Path to retrieval_config.yaml (None = auto-detect)
        label_map_path: Path to corner_config.yaml (None = auto-detect)
        kde_diag: Use KDE on diagonal instead of histograms

    Returns:
        Path to saved figure
    """
    if not dynesty_path.exists():
        raise FileNotFoundError(f"Could not find Dynesty results: {dynesty_path}")

    print(f"[plot_dynesty] Loading Dynesty results from {dynesty_path}...")
    with open(dynesty_path, "rb") as f:
        results = pickle.load(f)

    # Extract equal-weighted posterior samples
    samples = results['samples']
    weights = np.exp(results['logwt'] - results['logz'][-1])

    # Resample to equal weights
    n_samples_out = 10000
    indices = np.random.choice(
        len(samples),
        size=n_samples_out,
        replace=True,
        p=weights / weights.sum()
    )
    samples_equal = samples[indices]

    # Auto-detect config if not provided
    if config_path is None:
        default_cfg = dynesty_path.parent / "retrieval_config.yaml"
        config_path = default_cfg if default_cfg.exists() else None
    config_data = _load_config_data(config_path)

    # Auto-detect label map if not provided
    if label_map_path is None:
        default_label_yaml = dynesty_path.parent / "corner_config.yaml"
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
        print(f"[plot_dynesty] Could not load parameter names from config: {e}")
        internal_names = [f"param_{i}" for i in range(samples_equal.shape[1])]

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
        samples_equal = samples_equal[:, indices_to_plot]
        param_labels = [param_labels[i] for i in indices_to_plot]
        internal_names = [internal_names[i] for i in indices_to_plot]

    print(f"[plot_dynesty] Plotting variables: {', '.join(param_labels)}")

    # Print summary statistics
    print("\n[plot_dynesty] Posterior Summary:")
    print(f"  logZ = {results['logz'][-1]:.2f} ± {results['logzerr'][-1]:.2f}")
    print(f"  Samples: {len(samples_equal)}")
    print("\nMedian values:")
    for i, name in enumerate(internal_names):
        q_low, q_mid, q_high = np.quantile(samples_equal[:, i], quantiles)
        print(f"  {name:25s}: {q_mid:.3f} +{q_high - q_mid:.3f} -{q_mid - q_low:.3f}")

    # Create corner plot with seaborn theme (matching posterior_corner.py)
    sns.set_theme(style="ticks", palette="colorblind")
    contour_color = sns.color_palette("colorblind", 1)[0]

    fig = corner.corner(
        samples_equal,
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
        _replace_diag_with_kde(fig, samples_equal, quantiles, contour_color)

    # Add title
    fig.suptitle(
        f"Dynesty Posterior | logZ = {results['logz'][-1]:.2f} ± {results['logzerr'][-1]:.2f}",
        fontsize=14,
        y=1.0
    )

    # Save
    output_path = dynesty_path.parent / f"{outname}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n[plot_dynesty] Saved corner plot to {output_path}")
    plt.close()

    return output_path


# ============================================================================
# CLI wrapper
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Dynesty corner (interface matches posterior_corner.py)"
    )
    parser.add_argument(
        "dynesty_path",
        nargs="?",
        type=Path,
        default=Path("dynesty_results.pkl"),
        help="Path to dynesty_results.pkl (default: dynesty_results.pkl)"
    )
    parser.add_argument(
        "--params",
        nargs="*",
        help="Parameters to plot (default: all)"
    )
    parser.add_argument(
        "--outname",
        type=str,
        default="corner_dynesty",
        help="Output filename without extension (default: corner_dynesty)"
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
        dynesty_path=args.dynesty_path,
        params=args.params,
        outname=args.outname,
        quantiles=(0.1585, 0.5, 0.8415),
        config_path=args.config,
        label_map_path=args.label_map,
        kde_diag=args.kde_diag,
    )
