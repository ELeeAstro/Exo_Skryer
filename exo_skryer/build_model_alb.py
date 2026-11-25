"""
build_model_alb.py
==================

Forward model for reflected-light (albedo) spectra (planned ``physics.rt_scheme:
albedo_1d``). Mirrors ``build_model_em``: one column, ``atmosphere -> opacities ->
albedo RT (RT_alb_core_os / RT_alb_core_ck with a solver from RT_alb_solvers) ->
instrument binning``.

Stub: raises until the shortwave solvers and the albedo convention (geometric albedo
vs. reflectance at a phase angle) are defined.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import jax.numpy as jnp
import numpy as np

__all__ = ["build_forward_model_alb"]


def build_forward_model_alb(
    cfg,
    obs: Dict,
    stellar_flux: Optional[np.ndarray] = None,
    return_highres: bool = False,
    exp_dir: Optional[Path] = None,
) -> Callable[[Dict[str, jnp.ndarray]], Union[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    """Build the JIT-compiled albedo forward model (stub)."""
    raise NotImplementedError(
        "physics.rt_scheme: albedo_1d is planned but not implemented yet (build_model_alb is a stub; "
        "the solvers in RT_alb_solvers are stubs too)."
    )
