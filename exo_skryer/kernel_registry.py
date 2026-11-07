"""kernel_registry.py
==================
Central registry mapping YAML scheme names to kernel functions.

Each entry is a :class:`Kernel`: the function plus what it needs at build time
(opacity-cache keys, setup loaders). Builders read only this registry, so a new
scheme needs no builder edits:

1. Implement the kernel with the signature of its kind (listed below).
2. Register it, either here or from your own script::

       from exo_skryer import kernel_registry as KR
       KR.register("vert_Tp", "my_profile", my_profile)

3. Use ``vert_Tp: my_profile`` in the YAML. A ``module:function`` path
   (``vert_Tp: mypkg.profiles:my_profile``) also works without registering.

Kinds and signatures
--------------------
- ``vert_Tp``: ``(p_lev, params) -> (T_lev, T_lay)``
- ``vert_alt``: ``(p_lev, T_lay, mu_lay, params) -> (z_lev, z_lay, dz)``
- ``vert_chem``: ``(p_lay, T_lay, params, nlay) -> vmr_lay``
- ``vert_mu``: ``(params, vmr_lay, nlay) -> mu_lay``
- ``vert_cloud``: ``(p_lay, T_lay, mu_lay, rho_lay, nd_lay, params) -> q_c_lay``
  (kernels may also accept ``p_lev=`` and ``return_cover=`` keywords; see
  ``build_common._cloud_profile_caller``)
- ``opac_line``, ``opac_ray``, ``opac_cia``, ``opac_special``: ``(state, opac, params) -> kappa``
- ``opac_cloud``: ``(state, params) -> (k_ext, ssa, g)``
- ``rt``: keyed ``"<rt_scheme>/<os|ck|ck_trans>"``; 1D kernels are
  ``(state, params, components, opac) -> (spectrum, contrib)``
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Optional

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
    CE_fastchem_grid_jax,
    CE_rate_jax,
    CE_easychem_jax,
    quench_approx,
    CE_atmodeller,
)
from .vert_mu import constant_mu, compute_mu
from .vert_cloud import (
    no_cloud,
    exponential_decay_profile,
    slab_profile,
    const_profile,
)
from .opacity_line import compute_line_opacity
from .opacity_ck import compute_ck_opacity
from .opacity_ray import compute_ray_opacity
from .opacity_cia import compute_cia_opacity
from .opacity_special import compute_special_opacity
from .opacity_cloud import (
    compute_cloud_opacity,
    grey_const_cloud,
    grey_profile_cloud,
    deck_and_powerlaw,
    direct_nk,
    nk_f18_blend,
    f18_skew_cloud,
)
from .RT_trans_1D_os import compute_transit_depth_1d_os
from .RT_trans_1D_ck import compute_transit_depth_1d_ck
from .RT_trans_1D_ck_trans import compute_transit_depth_1d_ck_trans
from .RT_em_1D_os import compute_emission_spectrum_1d_os
from .RT_em_1D_ck import compute_emission_spectrum_1d_ck
from .RT_trans_1_5D_os import compute_transit_depth_1_5d_os
from .RT_trans_1_5D_ck import compute_transit_depth_1_5d_ck
from .RT_trans_1_5D_ck_trans import compute_transit_depth_1_5d_ck_trans
from . import build_chem as _chem

__all__ = [
    "Kernel",
    "KINDS",
    "register",
    "get",
    "names",
    "resolve",
    "VERT_TP",
    "VERT_ALT",
    "VERT_CHEM",
    "VERT_MU",
    "VERT_CLOUD",
    "OPAC_CLOUD",
]


@dataclass(frozen=True)
class Kernel:
    """A registered scheme and its build-time requirements."""
    fn: Optional[Callable]                # None = disabled
    cache: tuple[str, ...] = ()           # opacity-cache keys the kernel reads
    setup: tuple[Callable, ...] = ()      # setup(cfg, exp_dir) loaders, idempotent


KINDS = (
    "vert_Tp", "vert_alt", "vert_chem", "vert_mu", "vert_cloud",
    "opac_line", "opac_ray", "opac_cia", "opac_special", "opac_cloud", "rt",
)
_REGISTRY: dict[str, dict[str, Kernel]] = {kind: {} for kind in KINDS}
_ALIASES: dict[str, dict[str, str]] = {kind: {} for kind in KINDS}


def register(
    kind: str,
    name: str,
    fn: Optional[Callable],
    *,
    cache: Iterable[str] = (),
    setup: Iterable[Callable] = (),
    aliases: Iterable[str] = (),
) -> Kernel:
    """Register ``fn`` under ``name`` (case-insensitive) for ``kind``."""
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown kernel kind {kind!r}. Kinds: {', '.join(KINDS)}")
    entry = Kernel(fn, tuple(cache), tuple(setup))
    key = name.strip().lower()
    _REGISTRY[kind][key] = entry
    for alias in aliases:
        _ALIASES[kind][alias.strip().lower()] = key
    return entry


def names(kind: str) -> list[str]:
    """Canonical names registered for ``kind``."""
    return sorted(_REGISTRY[kind])


def get(kind: str, name, cfg_key: Optional[str] = None) -> Kernel:
    """Look up ``name`` for ``kind``; ``module:function`` paths are imported."""
    cfg_key = cfg_key or f"physics.{kind}"
    if name is None:
        raise ValueError(f"{cfg_key} must be set explicitly. Valid options: {names(kind)}")
    raw = str(name).strip()
    key = raw.lower()
    key = _ALIASES[kind].get(key, key)
    if key in _REGISTRY[kind]:
        return _REGISTRY[kind][key]
    if ":" in raw:
        module_name, _, attr = raw.partition(":")
        try:
            return Kernel(getattr(importlib.import_module(module_name), attr))
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"{cfg_key}='{raw}' could not be imported: {exc}") from exc
    raise ValueError(f"Unknown {cfg_key}='{raw}'. Valid options: {names(kind)}")


class _LegacyView(Mapping):
    """Read-only ``{name_or_alias: fn}`` view of one kind (pre-registry API)."""

    def __init__(self, kind: str):
        self._kind = kind

    def _items(self) -> dict:
        reg = _REGISTRY[self._kind]
        out = {k: v.fn for k, v in reg.items()}
        out.update({a: reg[c].fn for a, c in _ALIASES[self._kind].items()})
        return out

    def __getitem__(self, key):
        return self._items()[key]

    def __iter__(self):
        return iter(self._items())

    def __len__(self):
        return len(self._items())


def resolve(name, registry, cfg_key: str):
    """Pre-registry API: return the kernel function for ``name``."""
    kind = registry._kind if isinstance(registry, _LegacyView) else registry
    return get(kind, name, cfg_key).fn


VERT_TP = _LegacyView("vert_Tp")
VERT_ALT = _LegacyView("vert_alt")
VERT_CHEM = _LegacyView("vert_chem")
VERT_MU = _LegacyView("vert_mu")
VERT_CLOUD = _LegacyView("vert_cloud")
OPAC_CLOUD = _LegacyView("opac_cloud")


# ---------------------------------------------------------------------------
# Opacity-cache keys
# ---------------------------------------------------------------------------

LINE_CACHE = ("line_sigma_cube", "line_log10_pressure_grid", "line_log10_temperature_grids")
CK_CACHE = ("ck_sigma_cube", "ck_log10_pressure_grid", "ck_log10_temperature_grids",
            "g_points", "g_weights")
CIA_CACHE = ("cia_master_wavelength", "cia_pair_species_i", "cia_pair_species_j",
             "cia_retained_sigma_cube", "cia_retained_log10_temperature_grids",
             "cia_retained_temperature_grids")
RAY_CACHE = ("ray_master_wavelength", "ray_sigma_linear_table")
CLOUD_NK_CACHE = ("cloud_nk_n", "cloud_nk_k")


# ---------------------------------------------------------------------------
# Vertical structure
# ---------------------------------------------------------------------------

register("vert_Tp", "isothermal", isothermal, aliases=("constant",))
register("vert_Tp", "barstow", Barstow)
register("vert_Tp", "milne", Milne)
register("vert_Tp", "modified_milne", Modified_Milne, aliases=("milne_2", "milne_modified"))
register("vert_Tp", "guillot", Guillot)
register("vert_Tp", "modified_guillot", Modified_Guillot, aliases=("guillot_modified", "guillot_2"))
register("vert_Tp", "line", Line)
register("vert_Tp", "picket_fence", picket_fence)
register("vert_Tp", "mands", MandS)

register("vert_alt", "hypsometric", hypsometric, aliases=("constant", "constant_g", "fixed"))
register("vert_alt", "variable_g", hypsometric_variable_g,
         aliases=("hypsometric_variable_g", "variable"))
register("vert_alt", "p_ref", hypsometric_variable_g_pref, aliases=("hypsometric_variable_g_pref",))

register("vert_chem", "constant_vmr", constant_vmr, aliases=("constant",))
register("vert_chem", "constant_vmr_clr", constant_vmr_clr, aliases=("constant_clr", "clr"))
register("vert_chem", "ce", CE_fastchem_jax, setup=(_chem.init_fastchem_grid,),
         aliases=("chemical_equilibrium", "ce_fastchem_jax", "fastchem_jax"))
register("vert_chem", "fastchem_grid_jax", CE_fastchem_grid_jax, setup=(_chem.init_fastchem_grid,),
         aliases=("ce_fastchem_grid", "fastchem_ce_grid"))
register("vert_chem", "rate_ce", CE_rate_jax, setup=(_chem.load_nasa9,),
         aliases=("rate_jax", "ce_rate_jax"))
register("vert_chem", "easychem_jax", CE_easychem_jax,
         setup=(_chem.load_nasa9, _chem.init_element_potentials), aliases=("easychem",))
register("vert_chem", "quench_approx", quench_approx, setup=(_chem.init_quench_approx,),
         aliases=("quench",))
register("vert_chem", "atmodeller", CE_atmodeller, setup=(_chem.init_atmodeller,))


def _mu_auto(params, vmr_lay, nlay):
    """Use constant_mu if 'mu' is a free parameter, otherwise compute from VMR."""
    if "mu" in params:
        return constant_mu(params, nlay)
    if "__mu_lay__" in vmr_lay:
        return vmr_lay["__mu_lay__"]
    return compute_mu(vmr_lay)


def _mu_constant(params, vmr_lay, nlay):
    """Fixed mean molecular weight from the 'mu' retrieval parameter."""
    return constant_mu(params, nlay)


def _mu_dynamic(params, vmr_lay, nlay):
    """Mean molecular weight computed self-consistently from VMR profiles."""
    if "__mu_lay__" in vmr_lay:
        return vmr_lay["__mu_lay__"]
    return compute_mu(vmr_lay)


register("vert_mu", "auto", _mu_auto)
register("vert_mu", "constant", _mu_constant, aliases=("fixed",))
register("vert_mu", "dynamic", _mu_dynamic, aliases=("variable", "vmr"))

register("vert_cloud", "none", no_cloud, aliases=("off", "no_cloud"))
register("vert_cloud", "exponential", exponential_decay_profile,
         aliases=("exp_decay", "exponential_decay", "exponential_decay_profile"))
register("vert_cloud", "slab", slab_profile, aliases=("slab_profile",))
register("vert_cloud", "constant", const_profile, aliases=("const", "const_profile"))


# ---------------------------------------------------------------------------
# Opacities
# ---------------------------------------------------------------------------

register("opac_line", "none", None)
register("opac_line", "os", compute_line_opacity, cache=LINE_CACHE)
register("opac_line", "ck", compute_ck_opacity, cache=CK_CACHE)

# Rayleigh and CIA kernels are the same for os and ck.
for _kind, _fn, _cache in (("opac_ray", compute_ray_opacity, RAY_CACHE),
                           ("opac_cia", compute_cia_opacity, CIA_CACHE)):
    register(_kind, "none", None)
    register(_kind, "os", _fn, cache=_cache, aliases=("ck",))

# H- tables are optional; the kernel returns zeros without them.
register("opac_special", "none", None, aliases=("off", "false", "0"))
register("opac_special", "on", compute_special_opacity, aliases=("os", "ck", "true", "hminus", "h-"))

_f18 = partial(compute_cloud_opacity, opacity_scheme="f18")
_madt = partial(compute_cloud_opacity, opacity_scheme="madt_rayleigh")
_lxmie = partial(compute_cloud_opacity, opacity_scheme="lxmie")
register("opac_cloud", "none", None, aliases=("off",))
register("opac_cloud", "grey_const", grey_const_cloud)
register("opac_cloud", "grey_profile", grey_profile_cloud, aliases=("grey_slab",))
register("opac_cloud", "powerlaw", deck_and_powerlaw, aliases=("deck_and_powerlaw",))
register("opac_cloud", "f18", _f18)
register("opac_cloud", "madt_rayleigh", _madt, cache=CLOUD_NK_CACHE, aliases=("madt-rayleigh", "mie_madt"))
register("opac_cloud", "lxmie", _lxmie, cache=CLOUD_NK_CACHE, aliases=("mie_full", "full_mie"))
register("opac_cloud", "direct_nk", direct_nk, aliases=("nk",))
register("opac_cloud", "nk_f18_blend", nk_f18_blend, aliases=("nk_f18",))
register("opac_cloud", "f18_skew", f18_skew_cloud, aliases=("f18_skewnormal",))


# ---------------------------------------------------------------------------
# Radiative transfer
# ---------------------------------------------------------------------------

register("rt", "transit_1d/os", compute_transit_depth_1d_os)
register("rt", "transit_1d/ck", compute_transit_depth_1d_ck)
register("rt", "transit_1d/ck_trans", compute_transit_depth_1d_ck_trans)
register("rt", "emission_1d/os", compute_emission_spectrum_1d_os)
register("rt", "emission_1d/ck", compute_emission_spectrum_1d_ck)
register("rt", "transit_1_5d/os", compute_transit_depth_1_5d_os)
register("rt", "transit_1_5d/ck", compute_transit_depth_1_5d_ck)
register("rt", "transit_1_5d/ck_trans", compute_transit_depth_1_5d_ck_trans)
