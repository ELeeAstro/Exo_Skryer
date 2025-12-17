#!/usr/bin/env python3
"""
plot_nk_profile.py
==================

Overview:
    Visualize the posterior distributions of the complex refractive indices
    (real part `n` and imaginary part `k`) inferred for the WASP-17b cloud
    model. The script reads an ArviZ NetCDF (posterior_corner.nc by default)
    plus the experiment retrieval_config.yaml to obtain:
      * The wl-node wavelengths defined in the config
      * Posterior samples for log_10_k_i parameters and either `n`
        (constant) or node-based `n_i`
      * The master wavelength grid specified via opac.wl_master
    It then:
      1. Converts log_10(k) samples to k
      2. Uses exo_skryer.aux_funtions.pchip_1d to interpolate each sampled k
         profile (and n profile when node-based `n_i` are present) onto the
         master wavelength grid
      3. Builds median/±1σ (16th/84th percentile) summaries at both the node
         positions and across the interpolated grid
      4. Plots separate panels for n (linear y) and k (log y), with shaded
         1σ regions and optional overlays from nk_data.

Usage:
    python plot_nk_profile.py \
        --config retrieval_config.yaml \
        --posterior posterior_corner.nc \
        --output nk_profile.png \
        --read nk_data/SiO2.txt

Notes:
    * The script falls back to posterior.nc if posterior_corner.nc is missing.
    * Set --max-draws to control how many posterior draws are interpolated.
      (Default: 2000 evenly spaced draws from the posterior.)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml

# Ensure repository root is importable for exo_skryer modules.
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exo_skryer.aux_funtions import pchip_1d

jax.config.update("jax_enable_x64", True)

COLORBLIND_PALETTE = sns.color_palette("colorblind")
SEABORN_BLUE = COLORBLIND_PALETTE[0]
SEABORN_ORANGE = COLORBLIND_PALETTE[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot posterior n/k profiles.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("retrieval_config.yaml"),
        help="Path to retrieval_config.yaml (default: experiment folder)",
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path("posterior_corner.nc"),
        help="Path to posterior_corner.nc (relative to config folder by default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("nk_profile.png"),
        help="Output figure path (default: nk_profile.png next to config)",
    )
    parser.add_argument(
        "--max-draws",
        type=int,
        default=200,
        help="Maximum number of posterior draws to interpolate (default: 2000).",
    )
    parser.add_argument(
        "--read",
        type=Path,
        default=None,
        help="Optional nk_data file to overlay (e.g., nk_data/SiO2.txt).",
    )
    return parser.parse_args()


def _load_yaml_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_relative_path(base_dir: Path, target: Path) -> Path:
    if target.is_absolute():
        return target
    return (base_dir / target).resolve()


def _read_wavelength_grid(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Wavelength grid not found: {path}")
    wavelengths: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"No data found in wavelength file: {path}")
    for entry in lines[1:]:
        parts = entry.replace(",", " ").split()
        value = float(parts[-1])
        wavelengths.append(value)
    arr = np.asarray(wavelengths, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("Wavelength grid must contain positive values.")
    return arr


def _extract_node_wavelengths(
    cfg: Dict,
    prefix: str,
    *,
    required: bool = False,
) -> Tuple[np.ndarray, Sequence[int]]:
    params = cfg.get("params", [])
    nodes: Dict[int, float] = {}
    for entry in params:
        name = str(entry.get("name", ""))
        if not name.startswith(prefix):
            continue
        try:
            idx = int(name.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid node name '{name}' for prefix '{prefix}'") from exc
        nodes[idx] = float(entry.get("value"))
    if not nodes:
        if required:
            raise ValueError(f"No node entries found for prefix '{prefix}'.")
        return np.asarray([], dtype=float), []
    sorted_indices = sorted(nodes)
    wavelengths = np.asarray([nodes[i] for i in sorted_indices], dtype=float)
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError(f"{prefix} wavelengths must be strictly increasing.")
    return wavelengths, sorted_indices


def _flatten_var(samples) -> np.ndarray:
    arr = np.asarray(samples, dtype=float)
    return arr.reshape(-1)


def _stack_ordered_samples(
    posterior_ds,
    template: str,
    indices: Sequence[int],
) -> Tuple[np.ndarray, List[str]]:
    values: List[np.ndarray] = []
    var_names: List[str] = []
    missing: List[str] = []
    for idx in indices:
        var = template.format(idx=idx)
        if var not in posterior_ds:
            missing.append(var)
            continue
        values.append(_flatten_var(posterior_ds[var].values))
        var_names.append(var)
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Posterior variable(s) not found: {missing_str}")
    stacked = np.stack(values, axis=1)  # (n_samples, n_nodes)
    return stacked, var_names


def _stack_logk_samples(posterior_ds, indices: Sequence[int]) -> Tuple[np.ndarray, List[str]]:
    return _stack_ordered_samples(posterior_ds, "log_10_k_{idx}", indices)


def _stack_n_samples(
    posterior_ds,
    indices: Sequence[int],
) -> Tuple[np.ndarray, List[str]]:
    return _stack_ordered_samples(posterior_ds, "n_{idx}", indices)


def _choose_draw_indices(n_total: int, max_draws: int) -> np.ndarray:
    if n_total <= max_draws or max_draws <= 0:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, max_draws, dtype=int)


def _interpolate_pchip_samples(
    samples: np.ndarray,
    wl_nodes: np.ndarray,
    wl_master: np.ndarray,
    draw_indices: np.ndarray,
) -> np.ndarray:
    # Pre-jit interpolation for efficiency.
    x_nodes = jnp.asarray(wl_nodes)
    x_eval = jnp.asarray(wl_master)

    @jax.jit
    def _interp_single(y_nodes: jnp.ndarray) -> jnp.ndarray:
        return pchip_1d(x_eval, x_nodes, y_nodes)

    n_draws = draw_indices.shape[0]
    n_eval = wl_master.shape[0]
    curves = np.empty((n_draws, n_eval), dtype=float)
    for i, idx in enumerate(draw_indices):
        y_nodes = jnp.asarray(samples[idx], dtype=jnp.float64)
        curves[i] = np.asarray(_interp_single(y_nodes))
    return curves


def _compute_quantiles(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q16, q50, q84 = np.quantile(data, [0.16, 0.5, 0.84], axis=0)
    return q16, q50, q84


def _load_nk_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"nk data file not found: {path}")
    wl: List[float] = []
    n_vals: List[float] = []
    k_vals: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                wl_value = float(parts[0])
                n_value = float(parts[1])
                k_value = float(parts[2])
            except ValueError:
                continue
            wl.append(wl_value)
            n_vals.append(n_value)
            k_vals.append(k_value)
    if not wl:
        raise ValueError(f"No numeric entries found in nk data file: {path}")
    wl_arr = np.asarray(wl, dtype=float)
    n_arr = np.asarray(n_vals, dtype=float)
    k_arr = np.asarray(k_vals, dtype=float)
    order = np.argsort(wl_arr)
    return wl_arr[order], n_arr[order], k_arr[order]


def _plot_profiles(
    wl_nodes: np.ndarray,
    k_node_stats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    wl_master: np.ndarray,
    k_interp_stats: Tuple[np.ndarray, np.ndarray, np.ndarray],
    n_profile: Dict[str, object],
    output_path: Path,
    overlay: Tuple[np.ndarray, np.ndarray, np.ndarray, str] | None = None,
) -> None:
    k_node_lo, k_node_mid, k_node_hi = k_node_stats
    k_lo, k_mid, k_hi = k_interp_stats

    fig, (ax_n, ax_k) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
        constrained_layout=True,
    )
    #ax_n.set_xscale("log")
    #ax_k.set_xscale("log")
    ax_k.set_yscale("log")
    ax_k.set_ylim(1e-6,10.0)
    ax_n.set_ylim(0.0,4.5)
    ax_k.set_xlim(7,12)
    ax_n.set_xlim(7,12)   

    ax_k.plot(
        wl_master,
        k_mid,
        color=SEABORN_BLUE,
        label="$k$ (median)",
        linewidth=2.0,
    )
    ax_k.fill_between(
        wl_master,
        k_lo,
        k_hi,
        color=SEABORN_BLUE,
        alpha=0.25,
        label="$k$ ±1σ",
    )
    ax_k.scatter(
        wl_nodes,
        k_node_mid,
        color=SEABORN_BLUE,
        marker="o",
        s=40,
        label="$k$ nodes",
        zorder=3,
    )
    ax_k.vlines(
        wl_nodes,
        k_node_lo,
        k_node_hi,
        color=SEABORN_BLUE,
        linewidth=1.0,
        alpha=0.7,
    )

    n_mode = n_profile.get("mode", "constant")
    if n_mode == "pchip":
        n_lo, n_mid, n_hi = n_profile["interp_stats"]  # type: ignore[index]
        ax_n.plot(
            wl_master,
            n_mid,
            color=SEABORN_ORANGE,
            linewidth=2.0,
            label="$n$ (median)",
        )
        ax_n.fill_between(
            wl_master,
            n_lo,
            n_hi,
            color=SEABORN_ORANGE,
            alpha=0.25,
            label="$n$ ±1σ",
        )
        n_node_lo, n_node_mid, n_node_hi = n_profile["node_stats"]  # type: ignore[index]
        n_wl_nodes = n_profile["wl_nodes"]  # type: ignore[assignment]
        ax_n.scatter(
            n_wl_nodes,
            n_node_mid,
            color=SEABORN_ORANGE,
            marker="s",
            s=40,
            label="$n$ nodes",
            zorder=3,
        )
        ax_n.vlines(
            n_wl_nodes,
            n_node_lo,
            n_node_hi,
            color=SEABORN_ORANGE,
            linewidth=1.0,
            alpha=0.7,
        )
    else:
        const_lo, const_mid, const_hi = n_profile["const_stats"]  # type: ignore[index]
        n_line = np.full_like(wl_master, const_mid)
        n_low = np.full_like(wl_master, const_lo)
        n_high = np.full_like(wl_master, const_hi)
        ax_n.plot(
            wl_master,
            n_line,
            color=SEABORN_ORANGE,
            linestyle="--",
            linewidth=2.0,
            label="$n$ median",
        )
        ax_n.fill_between(
            wl_master,
            n_low,
            n_high,
            color=SEABORN_ORANGE,
            alpha=0.25,
            label="$n$ ±1σ",
        )

    if overlay is not None:
        wl_ref, n_ref, k_ref, label = overlay
        ax_n.plot(
            wl_ref,
            n_ref,
            color="tab:green",
            linestyle="-.",
            linewidth=1.5,
            label=f"{label} $n$",
        )
        pos_mask = k_ref > 0.0
        if np.any(pos_mask):
            ax_k.plot(
                wl_ref[pos_mask],
                k_ref[pos_mask],
                color="tab:green",
                linestyle="-.",
                linewidth=1.5,
                label=f"{label} $k$",
            )
        else:
            print(f"[plot_nk_profile] Overlay k values for {label} are non-positive; skipping.")

    ax_k.set_xlabel("Wavelength [$\\mu$m]", fontsize=16)
    ax_n.set_ylabel("$n$", fontsize=16)
    ax_k.set_ylabel("$k$", fontsize=16)
    ax_n.tick_params(axis="both", labelsize=14)
    ax_k.tick_params(axis="both", labelsize=14)
    #ax_n.set_title("WASP-17b Cloud Optical Constants (Posterior)")
    ax_n.legend(loc="best")
    ax_k.legend(loc="best")
    #ax_n.grid(True, which="both", alpha=0.3)
    #ax_k.grid(True, which="both", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config_dir = config_path.parent
    posterior_path = args.posterior
    if not posterior_path.is_absolute():
        posterior_path = config_dir / posterior_path
    posterior_path = posterior_path.resolve()
    if not posterior_path.exists() and args.posterior.name == "posterior_corner.nc":
        fallback = config_dir / "posterior.nc"
        if fallback.exists():
            print(f"[plot_nk_profile] Using fallback posterior at {fallback}")
            posterior_path = fallback
    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior NetCDF not found: {posterior_path}")

    cfg = _load_yaml_config(config_path)
    wl_nodes, node_indices = _extract_node_wavelengths(cfg, "wl_node_", required=True)
    n_wl_nodes, n_node_indices = _extract_node_wavelengths(cfg, "n_node_")
    if not n_node_indices:
        n_wl_nodes = wl_nodes
        n_node_indices = node_indices

    posterior = az.from_netcdf(posterior_path).posterior
    logk_samples, logk_names = _stack_logk_samples(posterior, node_indices)
    if logk_samples.shape[1] != wl_nodes.shape[0]:
        raise ValueError(
            f"Mismatch between wl_node count ({wl_nodes.shape[0]}) "
            f"and posterior vars ({logk_samples.shape[1]}: {logk_names})"
        )
    k_samples = np.power(10.0, logk_samples)
    k_node_stats = _compute_quantiles(k_samples)

    opac_cfg = cfg.get("opac", {})
    wl_master_name = opac_cfg.get("wl_master")
    if not wl_master_name:
        raise ValueError("opac.wl_master must be defined in the config.")
    wl_master_path = _resolve_relative_path(config_dir, Path(wl_master_name))
    wl_master = _read_wavelength_grid(wl_master_path)

    draw_indices = _choose_draw_indices(k_samples.shape[0], args.max_draws)
    k_interp_samples = _interpolate_pchip_samples(k_samples, wl_nodes, wl_master, draw_indices)
    k_interp_stats = _compute_quantiles(k_interp_samples)

    n_profile: Dict[str, object]
    n_var_candidates = [f"n_{idx}" for idx in n_node_indices]
    if all(var in posterior for var in n_var_candidates):
        n_samples, n_var_names = _stack_n_samples(posterior, n_node_indices)
        if n_samples.shape[1] != n_wl_nodes.shape[0]:
            raise ValueError(
                f"Mismatch between n node count ({n_wl_nodes.shape[0]}) "
                f"and posterior vars ({n_samples.shape[1]}: {n_var_names})"
            )
        n_node_stats = _compute_quantiles(n_samples)
        # Mirror draw subsampling, but allow for differing sample counts.
        if n_samples.shape[0] != k_samples.shape[0]:
            n_draw_indices = _choose_draw_indices(n_samples.shape[0], args.max_draws)
        else:
            n_draw_indices = draw_indices
        n_interp_samples = _interpolate_pchip_samples(
            n_samples, n_wl_nodes, wl_master, n_draw_indices
        )
        n_interp_stats = _compute_quantiles(n_interp_samples)
        n_profile = {
            "mode": "pchip",
            "wl_nodes": n_wl_nodes,
            "node_stats": n_node_stats,
            "interp_stats": n_interp_stats,
        }
    elif "n" in posterior:
        n_samples = _flatten_var(posterior["n"].values)
        n_stats = np.quantile(n_samples, [0.16, 0.5, 0.84])
        n_profile = {
            "mode": "constant",
            "const_stats": (n_stats[0], n_stats[1], n_stats[2]),
        }
    else:
        raise KeyError("Posterior variables for n not found (expected 'n' or 'n_{i}').")

    overlay_data: Tuple[np.ndarray, np.ndarray, np.ndarray, str] | None = None
    if args.read:
        overlay_path = args.read
        if not overlay_path.is_absolute():
            repo_candidate = (REPO_ROOT / overlay_path).resolve()
            config_candidate = (config_dir / overlay_path).resolve()
            if repo_candidate.exists():
                overlay_path = repo_candidate
            elif config_candidate.exists():
                overlay_path = config_candidate
            else:
                overlay_path = repo_candidate  # allow error below
        wl_ref, n_ref, k_ref = _load_nk_file(overlay_path)
        overlay_data = (wl_ref, n_ref, k_ref, overlay_path.stem)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = config_dir / output_path
    output_path = output_path.resolve()

    _plot_profiles(
        wl_nodes=wl_nodes,
        k_node_stats=k_node_stats,
        wl_master=wl_master,
        k_interp_stats=k_interp_stats,
        n_profile=n_profile,
        output_path=output_path,
        overlay=overlay_data,
    )
    print(f"[plot_nk_profile] Saved plot to {output_path}")


if __name__ == "__main__":
    main()
