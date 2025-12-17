#!/usr/bin/env python3
"""
plot_cloud_profile.py
=====================

Overview:
    Plot the retrieved vertical cloud mass mixing ratio profile (q_c) for the
    WASP-17b JWST transmission setup. The script reads posterior samples from
    posterior.nc (ArviZ format), reconstructs the analytic cloud profile for
    each draw using the same equations as exo_skryer.opacity_cloud.direct_nk,
    and summarizes the median plus ±1σ (16th/84th) and ±2σ (2.5th/97.5th)
    envelopes versus pressure.

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


def _compute_q_profiles(
    log10_q0: np.ndarray,
    log10_H: np.ndarray,
    log10_p_base: np.ndarray,
    p_lay: np.ndarray,
    width_base_dex: float,
) -> np.ndarray:
    q0 = np.power(10.0, log10_q0)
    H_cld = np.power(10.0, log10_H)
    alpha = 1.0 / np.maximum(H_cld, 1e-12)
    p_base = np.power(10.0, log10_p_base) * bar

    d_base = max(width_base_dex * np.log(10.0), 1e-12)

    logP = np.log(np.maximum(p_lay, 1e-30))[None, :]  # (1, nlay)
    logPb = np.log(np.maximum(p_base, 1e-30))[:, None]  # (n_draw, 1)

    S_base = 0.5 * (1.0 - np.tanh((logP - logPb) / d_base))
    scale = (p_lay[None, :] / np.maximum(p_base[:, None], 1e-30)) ** alpha[:, None]
    q_profiles = q0[:, None] * scale * S_base
    return np.clip(q_profiles, 0.0, None)


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

    p_bot_bar = params_lookup.get("p_bot")
    p_top_bar = params_lookup.get("p_top")
    if p_bot_bar is None or p_top_bar is None:
        raise KeyError("Both p_bot and p_top delta parameters must be defined in the config.")
    p_lay = _build_layer_pressures(p_bot_bar, p_top_bar, nlay)

    width_base_dex = params_lookup.get("width_base_dex", 0.25)

    posterior = az.from_netcdf(posterior_path).posterior
    required = ["log_10_q_c_0", "log_10_H_cld", "log_10_p_base"]
    missing = [name for name in required if name not in posterior]
    if missing:
        raise KeyError(f"Posterior variables missing: {', '.join(missing)}")

    log10_q0 = _flatten_samples(posterior["log_10_q_c_0"].values)
    log10_H = _flatten_samples(posterior["log_10_H_cld"].values)
    log10_p_base = _flatten_samples(posterior["log_10_p_base"].values)
    if not (log10_q0.size == log10_H.size == log10_p_base.size):
        raise ValueError("Posterior arrays for cloud parameters must have matching sizes.")

    q_profiles = _compute_q_profiles(log10_q0, log10_H, log10_p_base, p_lay, width_base_dex)
    q16, q50, q84 = np.quantile(q_profiles, [0.16, 0.5, 0.84], axis=0)
    q025, q975 = np.quantile(q_profiles, [0.025, 0.975], axis=0)

    p_lay_bar = p_lay / bar

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
        label="q_c ±2σ",
    )
    ax.fill_betweenx(
        p_lay_bar,
        q16,
        q84,
        color="tab:blue",
        alpha=0.35,
        label="q_c ±1σ",
    )
    ax.plot(q50, p_lay_bar, color="tab:blue", linewidth=2.0, label="Median q_c")

    ax.set_xlabel("Cloud mass mixing ratio q_c")
    ax.set_ylabel("Pressure [bar]")
    ax.set_title("WASP-17b Cloud Vertical Profile")
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
