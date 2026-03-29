#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az

from posterior_corner import _infer_scalar_params, _select_params_by_suffix, plot_corner


def _resolve_joint_and_limb_params(posterior_path: Path, limb_suffix: str) -> list[str]:
    posterior_ds = az.from_netcdf(posterior_path).posterior
    scalar_names = _infer_scalar_params(posterior_ds.data_vars)
    selected = _select_params_by_suffix(scalar_names, ("_joint", limb_suffix))
    if not selected:
        raise ValueError(
            f"No scalar posterior parameters found with suffix '_joint' or '{limb_suffix}'."
        )
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description="Corner plot for joint + west posterior parameters.")
    ap.add_argument("--posterior", type=str, default="posterior.nc")
    ap.add_argument("--config", type=str)
    ap.add_argument("--log-params", nargs="+")
    ap.add_argument("--outname", type=str, default="posterior_corner_west")
    ap.add_argument("--label-map", type=str)
    ap.add_argument("--kde-diag", action="store_true")
    ap.add_argument("--no-points", action="store_true")
    args = ap.parse_args()

    posterior_path = Path(args.posterior).resolve()
    config_path = Path(args.config).resolve() if args.config else None
    label_map_path = Path(args.label_map).resolve() if args.label_map else None

    params = _resolve_joint_and_limb_params(posterior_path, "_west")
    plot_corner(
        posterior_path,
        params=params,
        outname=args.outname,
        config_path=config_path,
        extra_log_params=args.log_params,
        label_map_path=label_map_path,
        kde_diag=args.kde_diag,
        enforce_label_map=False,
        plot_points=not args.no_points,
    )


if __name__ == "__main__":
    main()
