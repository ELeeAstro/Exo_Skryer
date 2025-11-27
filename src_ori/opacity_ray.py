"""
opacity_ray.py
==============

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations
from typing import Dict

import jax.numpy as jnp

import registry_ray as XR


def zero_ray_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    layer_count = int(state["nlay"])
    wavelength_count = int(state["nwl"])
    return jnp.zeros((layer_count, wavelength_count))


def compute_ray_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Compute Rayleigh scattering opacity.

    Args:
        state: State dictionary containing:
            - wl: Wavelengths
            - nd_lay: Number density per layer
            - rho_lay: Mass density per layer
            - vmr_lay: VMR dictionary indexed by species name
        params: Parameter dictionary (kept for API compatibility)

    Returns:
        Opacity array of shape (n_layers, n_wavelength) in cm^2/g
    """
    if not XR.has_ray_data():
        return zero_ray_opacity(state, params)
    wavelengths = state["wl"]
    number_density = state["nd_lay"]
    density = state["rho_lay"]
    layer_vmr = state["vmr_lay"]
    layer_count = number_density.shape[0]

    master_wavelength = XR.ray_master_wavelength()
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("Rayleigh wavelength grid must match forward-model grid.")

    sigma_values = XR.ray_sigma_table()
    species_names = XR.ray_species_names()

    mixing_ratio_list = []
    for name in species_names:
        # Try direct lookup first, then fall back to f_ prefix
        if name in layer_vmr:
            value = layer_vmr[name]
        elif f"f_{name}" in layer_vmr:
            value = layer_vmr[f"f_{name}"]
        else:
            raise KeyError(f"Missing Rayleigh mixing ratio for species '{name}'")

        value = jnp.asarray(value)
        if value.ndim == 0:
            mixing_ratio_list.append(jnp.full((layer_count,), value))
        elif value.ndim == 1:
            if value.shape[0] != layer_count:
                raise ValueError(f"Rayleigh mixing ratio '{name}' length {value.shape[0]} != {layer_count}.")
            mixing_ratio_list.append(value)
        else:
            raise ValueError(f"Rayleigh mixing ratio '{name}' has unsupported shape {value.shape}.")

    mixing_ratios = jnp.stack(mixing_ratio_list, axis=0)
    absorption = mixing_ratios[:, :, None] * number_density[None, :, None] * sigma_values[:, None, :]
    return jnp.sum(absorption, axis=0) / density[:, None]
