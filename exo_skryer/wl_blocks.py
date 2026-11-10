"""
wl_blocks.py
============

Run the opacities and radiative transfer one block of wavelengths at a time.

Nothing between the atmosphere and the high-resolution spectrum couples
wavelengths, so the spectrum can be computed block by block. Only one block's
(nlay, block) arrays then exist at a time, forward or backward, so the
per-particle working memory depends on the block size rather than on the
number of wavelengths.

Each block reads its slice of every opacity table with ``dynamic_slice``: no
reshaped copy of the tables, no padding. When ``block`` doesn't divide nwl,
the last block is shifted back to end at nwl, and the wavelengths it shares
with the previous block are dropped when the spectrum is assembled.

Every runtime table must be registered below with its wavelength axis (or as
having none), so a new table can't silently be passed whole into each block.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

__all__ = [
    "WAVELENGTH_AXIS",
    "NO_WAVELENGTH_AXIS",
    "check_wavelength_axes",
    "block_starts",
    "slice_block",
    "slice_opac",
    "assemble_blocks",
    "map_wavelength_blocks",
]

# Runtime opacity-cache keys with a wavelength axis, and which axis it is.
WAVELENGTH_AXIS: Dict[str, int] = {
    "line_sigma_cube": -1,                 # (nspec, nT, nP, nwl)
    "ck_sigma_cube": -2,                   # (nspec, nT, nP, nwl, ng)
    "cia_master_wavelength": 0,            # (nwl,)
    "cia_retained_sigma_cube": -1,         # (npairs, nT, nwl)
    "ray_master_wavelength": 0,            # (nwl,)
    "ray_sigma_linear_table": -1,          # (nspec, nwl)
    "ray_refractivity_coeff_table": -1,    # (nspec, nwl)
    "cloud_nk_n": 0,                       # (nwl,)
    "cloud_nk_k": 0,                       # (nwl,)
    "hminus_master_wavelength": 0,         # (nwl,)
    "hminus_bf_log10_sigma": -1,           # (nT, nwl)
    "hminus_ff_log10_sigma": -1,           # (nT, nwl)
}

# Runtime opacity-cache keys with no wavelength axis (passed to every block whole).
NO_WAVELENGTH_AXIS = frozenset({
    "line_log10_pressure_grid",
    "line_log10_temperature_grids",
    "ck_log10_pressure_grid",
    "ck_log10_temperature_grids",
    "g_points",
    "g_weights",
    "cia_pair_species_i",
    "cia_pair_species_j",
    "cia_retained_log10_temperature_grids",
    "cia_retained_temperature_grids",
    "hminus_temperature_grid",
    "hminus_log10_temperature_grid",
})


def check_wavelength_axes(opac: Mapping[str, jnp.ndarray], nwl: int) -> None:
    """Raise if a runtime table is unregistered or its wavelength axis isn't nwl long."""
    for key, arr in opac.items():
        if key in WAVELENGTH_AXIS:
            axis = WAVELENGTH_AXIS[key]
            if arr.shape[axis] != nwl:
                raise ValueError(
                    f"opac['{key}'] has shape {tuple(arr.shape)}; expected wavelength axis {axis} "
                    f"of length {nwl} for wavelength blocks."
                )
        elif key not in NO_WAVELENGTH_AXIS:
            raise KeyError(
                f"opac['{key}'] (shape {tuple(arr.shape)}) is not registered in wl_blocks.py. Add it to "
                "WAVELENGTH_AXIS (with its wavelength axis) or NO_WAVELENGTH_AXIS before using opac.wl_block."
            )


def block_starts(nwl: int, block: int) -> np.ndarray:
    """Start index of each block; the last block is shifted back to end at nwl."""
    if not 0 < block < nwl:
        raise ValueError(f"Block size must be in (0, nwl={nwl}), got {block}.")
    nb = -(-nwl // block)
    return np.minimum(np.arange(nb) * block, nwl - block).astype(np.int32)


def slice_block(x: jnp.ndarray, start, block: int, axis: int) -> jnp.ndarray:
    return lax.dynamic_slice_in_dim(x, start, block, axis=axis % x.ndim)


def slice_opac(opac: Mapping[str, jnp.ndarray], start, block: int) -> Dict[str, jnp.ndarray]:
    """Each table's slice for the block starting at ``start`` (same keys)."""
    return {
        k: slice_block(v, start, block, WAVELENGTH_AXIS[k]) if k in WAVELENGTH_AXIS else v
        for k, v in opac.items()
    }


def assemble_blocks(y: jnp.ndarray, nwl: int) -> jnp.ndarray:
    """Join block outputs ``(nb, ..., block)`` into ``(..., nwl)``.

    The last block covers [nwl - block, nwl); only its part past the previous
    block is kept.
    """
    nb, block = y.shape[0], y.shape[-1]
    head = jnp.moveaxis(y[:-1], 0, -2).reshape(y.shape[1:-1] + ((nb - 1) * block,))
    tail = y[-1][..., nb * block - nwl:]
    return jnp.concatenate([head, tail], axis=-1)


def map_wavelength_blocks(
    column: Callable[[jnp.ndarray], Tuple[jnp.ndarray, ...]],
    nwl: int,
    block: int,
) -> Tuple[jnp.ndarray, ...]:
    """Run ``column(start)`` over every block and assemble its outputs to full length.

    ``column`` returns a tuple of arrays whose last axis is the block. It runs
    under ``jax.checkpoint``, so under reverse-mode AD only the block start is
    stored per block and the block is recomputed on the way back.
    """
    starts = jnp.asarray(block_starts(nwl, block))
    outs = lax.map(jax.checkpoint(column), starts)
    return tuple(assemble_blocks(y, nwl) for y in outs)
