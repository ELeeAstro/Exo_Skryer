"""
build_model.py
==============

Public entry point for forward models. ``build_forward_model`` picks the builder from
``physics.rt_scheme``:

- ``transit_1d``, ``transit_1_5d`` (``transit_2d``/``transit_3d`` planned):
  :func:`exo_skryer.build_model_trans.build_forward_model_trans`
- ``emission_1d``: :func:`exo_skryer.build_model_em.build_forward_model_em`
- ``albedo_1d`` (planned; stub): :func:`exo_skryer.build_model_alb.build_forward_model_alb`
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import jax.numpy as jnp
import numpy as np

__all__ = ["build_forward_model"]


def build_forward_model(
    cfg,
    obs: Dict,
    stellar_flux: Optional[np.ndarray] = None,
    return_highres: bool = False,
    exp_dir: Optional[Path] = None,
) -> Callable[[Dict[str, jnp.ndarray]], Union[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    """Build a JIT-compiled forward model ``forward_model(params) -> spectrum``.

    Parameters
    ----------
    cfg : config object
        Configuration (``cfg.physics``, ``cfg.opac``, ``cfg.params``).
    obs : dict
        Observational data (limb slices for 1.5D with limb-resolved spectra).
    stellar_flux : `~numpy.ndarray`, optional
        Stellar flux on the high-resolution grid (planet emission).
    return_highres : bool, optional
        Return a dict of high-resolution and binned spectra plus diagnostics instead of
        the binned spectrum only.
    exp_dir : `~pathlib.Path`, optional
        Experiment directory for relative chemistry data paths.
    """
    rt_scheme = str(getattr(cfg.physics, "rt_scheme", "")).strip().lower()
    if rt_scheme.startswith("transit"):
        from .build_model_trans import build_forward_model_trans as build
    elif rt_scheme.startswith("emission"):
        from .build_model_em import build_forward_model_em as build
    elif rt_scheme.startswith("albedo"):
        from .build_model_alb import build_forward_model_alb as build
    else:
        raise ValueError(f"Unknown physics.rt_scheme '{rt_scheme}'. "
                         "Options: transit_1d | transit_1_5d | emission_1d | albedo_1d (planned)")
    return build(cfg, obs, stellar_flux=stellar_flux, return_highres=return_highres, exp_dir=exp_dir)
