"""kernel_registry.py
==================
Central registry mapping YAML physics-scheme names to their Python kernel functions.

**Adding a new module is a two-step process:**

1. Implement the function in the appropriate ``vert_*.py`` or ``opacity_*.py`` file
   and make sure it is exported in that module's ``__all__``.
2. Import it below and add one entry to the relevant registry dict.
3. Use the new key string in your YAML config — done!

Aliases (multiple keys that map to the same function) are marked with ``# alias``.

Registry layout
---------------
VERT_TP    : temperature-pressure profile kernels
VERT_ALT   : altitude / hydrostatic-structure kernels
VERT_CHEM  : chemistry / VMR-profile kernels
VERT_MU    : mean-molecular-weight kernels
VERT_CLOUD : cloud vertical-profile kernels
OPAC_CLOUD : cloud opacity kernels  (``None`` entry = disabled)
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

# --- vertical structure ---
from .vert_Tp import (
    isothermal,
    Barstow,
    Milne,
    Modified_Milne,
    Guillot,
    Modified_Guillot,
    Line,
    MandS,
    picket_fence,
)
from .vert_alt import (
    hypsometric,
    hypsometric_variable_g,
    hypsometric_variable_g_pref,
)
from .vert_chem import (
    constant_vmr,
    constant_vmr_clr,
    CE_fastchem_jax,
    CE_rate_jax,
    quench_approx,
)
from .vert_mu import constant_mu, compute_mu
from .vert_cloud import (
    no_cloud,
    exponential_decay_profile,
    slab_profile,
    const_profile,
)

# --- cloud opacity ---
from .opacity_cloud import (
    compute_cloud_opacity,
    grey_cloud,
    deck_and_powerlaw,
    direct_nk,
)

__all__ = [
    "VERT_TP",
    "VERT_ALT",
    "VERT_CHEM",
    "VERT_MU",
    "VERT_CLOUD",
    "OPAC_CLOUD",
    "resolve",
]

# ---------------------------------------------------------------------------
# Temperature-pressure profile kernels
# Signature: (p_lev, params) -> (T_lev, T_lay)
# ---------------------------------------------------------------------------

VERT_TP: dict[str, Callable] = {
    # canonical names
    "isothermal":        isothermal,
    "barstow":           Barstow,
    "milne":             Milne,
    "guillot":           Guillot,
    "modified_guillot":  Modified_Guillot,
    "modified_milne":    Modified_Milne,
    "line":              Line,
    "picket_fence":      picket_fence,
    "mands":             MandS,
    # aliases
    "constant":          isothermal,         # alias
    "guillot_modified":  Modified_Guillot,   # alias
    "guillot_2":         Modified_Guillot,   # alias
    "milne_2":           Modified_Milne,     # alias
    "milne_modified":    Modified_Milne,     # alias
}

# ---------------------------------------------------------------------------
# Altitude / hydrostatic-structure kernels
# Signature: (p_lev, T_lay, mu_lay, params) -> (z_lev, z_lay, dz)
# ---------------------------------------------------------------------------

VERT_ALT: dict[str, Callable] = {
    # canonical names
    "hypsometric":                   hypsometric,
    "variable_g":                    hypsometric_variable_g,
    "p_ref":                         hypsometric_variable_g_pref,
    # aliases
    "constant":                      hypsometric,                    # alias
    "constant_g":                    hypsometric,                    # alias
    "fixed":                         hypsometric,                    # alias
    "hypsometric_variable_g":        hypsometric_variable_g,         # alias
    "variable":                      hypsometric_variable_g,         # alias
    "hypsometric_variable_g_pref":   hypsometric_variable_g_pref,    # alias
}

# ---------------------------------------------------------------------------
# Chemistry / VMR-profile kernels
# Signature: (p_lay, T_lay, params, nlay) -> vmr_lay dict
# ---------------------------------------------------------------------------

VERT_CHEM: dict[str, Callable] = {
    # canonical names
    "constant_vmr":         constant_vmr,
    "constant_vmr_clr":     constant_vmr_clr,
    "ce":                   CE_fastchem_jax,
    "rate_ce":              CE_rate_jax,
    "quench_approx":        quench_approx,
    # aliases
    "constant":             constant_vmr,       # alias
    "constant_clr":         constant_vmr_clr,   # alias
    "clr":                  constant_vmr_clr,   # alias
    "chemical_equilibrium": CE_fastchem_jax,    # alias
    "ce_fastchem_jax":      CE_fastchem_jax,    # alias
    "fastchem_jax":         CE_fastchem_jax,    # alias
    "rate_jax":             CE_rate_jax,        # alias
    "ce_rate_jax":          CE_rate_jax,        # alias
    "quench":               quench_approx,      # alias
}

# ---------------------------------------------------------------------------
# Mean-molecular-weight kernels
# Signature: (params, vmr_lay, nlay) -> mu_lay
#
# These are thin wrappers so that constant_mu / compute_mu present a uniform
# interface regardless of whether the user passes `mu` as a parameter.
# ---------------------------------------------------------------------------

def _mu_auto(params, vmr_lay, nlay):
    """Use constant_mu if 'mu' is a free parameter, otherwise compute from VMR."""
    if "mu" in params:
        return constant_mu(params, nlay)
    return compute_mu(vmr_lay)


def _mu_constant(params, vmr_lay, nlay):
    """Fixed mean molecular weight from the 'mu' retrieval parameter."""
    return constant_mu(params, nlay)


def _mu_dynamic(params, vmr_lay, nlay):
    """Mean molecular weight computed self-consistently from VMR profiles."""
    return compute_mu(vmr_lay)


VERT_MU: dict[str, Callable] = {
    # canonical names
    "auto":     _mu_auto,
    "constant": _mu_constant,
    "dynamic":  _mu_dynamic,
    # aliases
    "fixed":    _mu_constant,   # alias
    "variable": _mu_dynamic,    # alias
    "vmr":      _mu_dynamic,    # alias
}

# ---------------------------------------------------------------------------
# Cloud vertical-profile kernels
# Signature: (p_lay, T_lay, mu_lay, rho_lay, nd_lay, params) -> q_c_lay
# ---------------------------------------------------------------------------

VERT_CLOUD: dict[str, Callable] = {
    # canonical names
    "none":        no_cloud,
    "exponential": exponential_decay_profile,
    "slab":        slab_profile,
    "constant":    const_profile,
    # aliases
    "off":                       no_cloud,                    # alias
    "no_cloud":                  no_cloud,                    # alias
    "exp_decay":                 exponential_decay_profile,   # alias
    "exponential_decay":         exponential_decay_profile,   # alias
    "exponential_decay_profile": exponential_decay_profile,   # alias
    "slab_profile":              slab_profile,                # alias
    "const":                     const_profile,              # alias
    "const_profile":             const_profile,              # alias
}

# ---------------------------------------------------------------------------
# Cloud opacity kernels
# Signature: (state, params) -> (k_ext, ssa, g)
# A value of None means cloud opacity is disabled for that entry.
# ---------------------------------------------------------------------------

OPAC_CLOUD: dict[str, Optional[Callable]] = {
    # canonical names
    "none":          None,
    "grey":          grey_cloud,
    "powerlaw":      deck_and_powerlaw,
    "f18":           partial(compute_cloud_opacity, opacity_scheme="f18"),
    "madt_rayleigh": partial(compute_cloud_opacity, opacity_scheme="madt_rayleigh"),
    "lxmie":         partial(compute_cloud_opacity, opacity_scheme="lxmie"),
    "direct_nk":     direct_nk,
    # aliases
    "off":              None,                                                          # alias
    "deck_and_powerlaw": deck_and_powerlaw,                                            # alias
    "madt-rayleigh":    partial(compute_cloud_opacity, opacity_scheme="madt_rayleigh"), # alias
    "mie_madt":         partial(compute_cloud_opacity, opacity_scheme="madt_rayleigh"), # alias
    "mie_full":         partial(compute_cloud_opacity, opacity_scheme="lxmie"),         # alias
    "full_mie":         partial(compute_cloud_opacity, opacity_scheme="lxmie"),         # alias
    "nk":               direct_nk,                                                     # alias
}

# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------

def resolve(name, registry: dict, cfg_key: str):
    """Return the kernel for *name* from *registry*, raising a clear error on failure.

    Parameters
    ----------
    name : str or None
        The scheme name from the YAML config (case-insensitive).  A Python
        ``None`` (i.e. the YAML key was absent entirely) always raises.
    registry : dict
        One of the ``VERT_TP`` / ``VERT_ALT`` / ``VERT_CHEM`` / … dicts above.
    cfg_key : str
        Config key shown in the error message, e.g. ``"physics.vert_Tp"``.

    Returns
    -------
    Callable or None
        The kernel function.  ``None`` is a valid return value for registries
        that explicitly map ``"none"`` to ``None`` (e.g. ``OPAC_CLOUD``).

    Raises
    ------
    ValueError
        If *name* is Python ``None`` (key missing from YAML) or not found in
        the registry, with a list of valid options included in the message.
    """
    if name is None:
        raise ValueError(
            f"{cfg_key} must be set explicitly. "
            f"Valid options: {_option_list(registry)}"
        )
    key = str(name).strip().lower()
    if key in registry:
        return registry[key]
    raise ValueError(
        f"Unknown {cfg_key}='{name}'. "
        f"Valid options: {_option_list(registry)}"
    )


def _option_list(registry: dict) -> list[str]:
    """Return sorted unique canonical option names for error messages."""
    return sorted(registry.keys())
