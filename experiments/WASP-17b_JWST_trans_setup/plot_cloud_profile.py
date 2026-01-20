#!/usr/bin/env python3
"""
plot_cloud_profile.py
=====================

Overview:
    Plot the retrieved vertical cloud mass mixing ratio profile (q_c) for the
    WASP-17b JWST transmission setup. The script reads posterior samples from
    posterior.nc (ArviZ format), reconstructs the cloud profile for each draw
    using the same equations as exo_skryer.vert_cloud, and summarizes the median
    plus ±1σ (16th/84th) and ±2σ (2.5th/97.5th) envelopes versus pressure.

    Supports multiple cloud profile types from vert_cloud.py:
    - slab_profile: Uniform slab between P_top and P_bot
    - exponential_decay_profile: Exponential decay with hard base cutoff
    - const_profile: Constant q_c throughout atmosphere

Usage:
    python plot_cloud_profile.py \
        --config retrieval_config.yaml \
        --posterior posterior.nc \
        --output cloud_profile.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Local imports
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exo_skryer.data_constants import bar  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot vertical cloud profile quantiles.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("retrieval_config.yaml"),
        help="Path to retrieval_config.yaml (default: experiment folder).",
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path("posterior.nc"),
        help="Path to posterior NetCDF (default: posterior.nc next to config).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("cloud_profile.png"),
        help="Output PNG path for the plot.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _collect_param_defaults(cfg: Dict) -> Dict[str, float]:
    lookup: Dict[str, float] = {}
    for entry in cfg.get("params", []):
        name = entry.get("name")
        if not name:
            continue
        if "value" in entry:
            lookup[name] = float(entry["value"])
        elif entry.get("dist") == "delta" and "init" in entry:
            lookup[name] = float(entry["init"])
    return lookup


def _build_layer_pressures(p_bot_bar: float, p_top_bar: float, nlay: int) -> np.ndarray:
    p_bot = p_bot_bar * bar
    p_top = p_top_bar * bar
    nlev = nlay + 1
    p_lev = np.logspace(np.log10(p_bot), np.log10(p_top), nlev)
    p_lay = (p_lev[1:] - p_lev[:-1]) / np.log(p_lev[1:] / p_lev[:-1])
    return p_lay


def _flatten_samples(data_array) -> np.ndarray:
    arr = np.asarray(data_array, dtype=np.float64)
    return arr.reshape(-1)


def _compute_slab_profiles(
    log10_q_c: np.ndarray,
    log10_p_top_slab: np.ndarray,
    log10_dp_slab: np.ndarray,
    p_lay: np.ndarray,
) -> np.ndarray:
    """Compute slab cloud profiles matching vert_cloud.slab_profile.

    Parameters
    ----------
    log10_q_c : (n_draw,)
        Log10 cloud mass mixing ratio inside the slab.
    log10_p_top_slab : (n_draw,)
        Log10 pressure at top of slab in bar.
    log10_dp_slab : (n_draw,)
        Log10 pressure extent of slab (P_bot = P_top * 10^dp).
    p_lay : (nlay,)
        Layer pressures in dyne cm^-2.

    Returns
    -------
    q_profiles : (n_draw, nlay)
        Cloud mass mixing ratio profiles.
    """
    q_c_slab = np.power(10.0, log10_q_c)  # (n_draw,)

    # Slab boundaries in pressure (bars -> dyne cm^-2)
    P_top = np.power(10.0, log10_p_top_slab) * bar  # (n_draw,)
    P_bot = np.power(10.0, log10_p_top_slab + log10_dp_slab) * bar  # (n_draw,)

    # Broadcast: p_lay (nlay,) vs P_top/P_bot (n_draw,)
    # Result: (n_draw, nlay)
    p_lay_2d = p_lay[None, :]  # (1, nlay)
    P_top_2d = P_top[:, None]  # (n_draw, 1)
    P_bot_2d = P_bot[:, None]  # (n_draw, 1)

    # Slab mask: 1 inside [P_top, P_bot], 0 outside
    slab_mask = (p_lay_2d >= P_top_2d) & (p_lay_2d <= P_bot_2d)

    q_profiles = q_c_slab[:, None] * slab_mask
    return q_profiles


def _compute_exponential_profiles(
    log10_q_c: np.ndarray,
    log10_alpha_cld: np.ndarray,
    log10_p_base: np.ndarray,
    p_lay: np.ndarray,
) -> np.ndarray:
    """Compute exponential decay profiles matching vert_cloud.exponential_decay_profile.

    Parameters
    ----------
    log10_q_c : (n_draw,)
        Log10 cloud mass mixing ratio at base pressure.
    log10_alpha_cld : (n_draw,)
        Log10 pressure power-law exponent.
    log10_p_base : (n_draw,)
        Log10 base pressure in bar.
    p_lay : (nlay,)
        Layer pressures in dyne cm^-2.

    Returns
    -------
    q_profiles : (n_draw, nlay)
        Cloud mass mixing ratio profiles.
    """
    q_c_0 = np.power(10.0, log10_q_c)  # (n_draw,)
    alpha = np.power(10.0, log10_alpha_cld)  # (n_draw,)
    p_base = np.power(10.0, log10_p_base) * bar  # (n_draw,)

    p_lay_2d = p_lay[None, :]  # (1, nlay)
    p_base_2d = np.maximum(p_base[:, None], 1e-30)  # (n_draw, 1)
    alpha_2d = alpha[:, None]  # (n_draw, 1)
    q_c_0_2d = q_c_0[:, None]  # (n_draw, 1)

    # Hard cutoff: clouds only for P < P_base
    cloud_mask = p_lay_2d < p_base_2d

    # Exponential profile
    q_c_profile = q_c_0_2d * (p_lay_2d / p_base_2d) ** alpha_2d

    q_profiles = np.where(cloud_mask, q_c_profile, 0.0)
    return q_profiles


def _compute_const_profiles(
    log10_q_c: np.ndarray,
    p_lay: np.ndarray,
) -> np.ndarray:
    """Compute constant profiles matching vert_cloud.const_profile.

    Parameters
    ----------
    log10_q_c : (n_draw,)
        Log10 cloud mass mixing ratio.
    p_lay : (nlay,)
        Layer pressures in dyne cm^-2.

    Returns
    -------
    q_profiles : (n_draw, nlay)
        Cloud mass mixing ratio profiles (constant at all layers).
    """
    q_c = np.power(10.0, log10_q_c)  # (n_draw,)
    nlay = len(p_lay)
    q_profiles = np.broadcast_to(q_c[:, None], (len(q_c), nlay)).copy()
    return q_profiles


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config_dir = config_path.parent
    posterior_path = args.posterior
    if not posterior_path.is_absolute():
        posterior_path = config_dir / posterior_path
    posterior_path = posterior_path.resolve()
    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior NetCDF not found: {posterior_path}")

    cfg = _load_yaml(config_path)
    params_lookup = _collect_param_defaults(cfg)
    physics_cfg = cfg.get("physics", {})
    nlay = int(physics_cfg.get("nlay", 99))

    # Get cloud profile type from config
    vert_cloud = physics_cfg.get("vert_cloud", "slab_profile")
    print(f"[plot_cloud_profile] Using vert_cloud model: {vert_cloud}")

    p_bot_bar = params_lookup.get("p_bot")
    p_top_bar = params_lookup.get("p_top")
    if p_bot_bar is None or p_top_bar is None:
        raise KeyError("Both p_bot and p_top delta parameters must be defined in the config.")
    p_lay = _build_layer_pressures(p_bot_bar, p_top_bar, nlay)

    posterior = az.from_netcdf(posterior_path).posterior

    # Compute profiles based on cloud model type
    if vert_cloud == "slab_profile":
        required = ["log_10_q_c", "log_10_p_top_slab", "log_10_dp_slab"]
        missing = [name for name in required if name not in posterior]
        if missing:
            raise KeyError(f"Posterior variables missing for slab_profile: {', '.join(missing)}")

        log10_q_c = _flatten_samples(posterior["log_10_q_c"].values)
        log10_p_top_slab = _flatten_samples(posterior["log_10_p_top_slab"].values)
        log10_dp_slab = _flatten_samples(posterior["log_10_dp_slab"].values)

        q_profiles = _compute_slab_profiles(log10_q_c, log10_p_top_slab, log10_dp_slab, p_lay)

    elif vert_cloud == "exponential_decay_profile":
        required = ["log_10_q_c", "log_10_alpha_cld", "log_10_p_base"]
        missing = [name for name in required if name not in posterior]
        if missing:
            raise KeyError(f"Posterior variables missing for exponential_decay_profile: {', '.join(missing)}")

        log10_q_c = _flatten_samples(posterior["log_10_q_c"].values)
        log10_alpha_cld = _flatten_samples(posterior["log_10_alpha_cld"].values)
        log10_p_base = _flatten_samples(posterior["log_10_p_base"].values)

        q_profiles = _compute_exponential_profiles(log10_q_c, log10_alpha_cld, log10_p_base, p_lay)

    elif vert_cloud == "const_profile":
        required = ["log_10_q_c"]
        missing = [name for name in required if name not in posterior]
        if missing:
            raise KeyError(f"Posterior variables missing for const_profile: {', '.join(missing)}")

        log10_q_c = _flatten_samples(posterior["log_10_q_c"].values)
        q_profiles = _compute_const_profiles(log10_q_c, p_lay)

    elif vert_cloud in ("no_cloud", "None", None):
        print("[plot_cloud_profile] No cloud model configured, nothing to plot.")
        return

    else:
        raise ValueError(f"Unknown vert_cloud model: {vert_cloud}")

    # Compute quantiles
    q16, q50, q84 = np.quantile(q_profiles, [0.16, 0.5, 0.84], axis=0)
    q025, q975 = np.quantile(q_profiles, [0.025, 0.975], axis=0)

    p_lay_bar = p_lay / bar

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()

    ax.fill_betweenx(
        p_lay_bar,
        q025,
        q975,
        color="tab:blue",
        alpha=0.15,
        label=r"$q_c$ ±2σ",
    )
    ax.fill_betweenx(
        p_lay_bar,
        q16,
        q84,
        color="tab:blue",
        alpha=0.35,
        label=r"$q_c$ ±1σ",
    )
    ax.plot(q50, p_lay_bar, color="tab:blue", linewidth=2.0, label=r"Median $q_c$")

    ax.set_xlabel(r"Cloud mass mixing ratio $q_c$")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title(f"WASP-17b Cloud Vertical Profile ({vert_cloud})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = config_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"[plot_cloud_profile] Saved plot to {output_path}")


if __name__ == "__main__":
    main()
