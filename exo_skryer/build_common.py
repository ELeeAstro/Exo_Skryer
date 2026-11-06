"""
build_common.py
===============
Pieces shared by the 1D and 1.5D forward-model builders: scheme selection,
build-time validation, and the atmosphere / opacity steps of the model.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial
from typing import Callable, Dict, Mapping, Optional, TypedDict

import jax
import jax.numpy as jnp

from .data_constants import kb, amu, R_jup, R_sun, bar, G, M_jup
from .opacity_line import zero_line_opacity
from .opacity_ck import zero_ck_opacity, compute_ck_opacity_perspecies
from .opacity_ray import zero_ray_opacity
from .opacity_cia import zero_cia_opacity
from .opacity_special import zero_special_opacity
from .opacity_cloud import zero_cloud_opacity
from .vert_chem import constant_vmr, constant_vmr_clr
from .vert_mu import build_compute_mu, constant_mu
from .RT_em_schemes import get_emission_solver
from .limb_asymmetry import split_limb_tag
from . import kernel_registry as KR

__all__ = [
    "CkMix",
    "Refraction",
    "ModelSpec",
    "ModelState",
    "select_model",
    "validate_model",
    "extract_fixed_params",
    "make_mu_kernel",
    "make_atmosphere_fn",
    "make_opacity_fn",
    "check_model_traces",
    "attach_runtime",
    "runtime_args_of",
    "bind_runtime",
]


class CkMix(IntEnum):
    """Correlated-k mixing scheme (``opac.ck_mix``)."""
    RORR = 1
    PRAS = 2
    TRANS = 3  # per-species tables, mixed inside the RT


class Refraction(IntEnum):
    """Refraction treatment (``physics.refraction``)."""
    OFF = 0
    CUTOFF = 1


_OFF = ("none", "off", "false", "0")
_BROWN_DWARF = ("brown_dwarf", "browndwarf", "bd")


class ModelState(TypedDict, total=False):
    """Per-call atmospheric state passed to opacity and RT kernels (cgs)."""
    nwl: int
    nlay: int
    wl: jnp.ndarray            # (nwl,) micron
    R0: jnp.ndarray            # planet radius at p_bot
    R_s: jnp.ndarray           # stellar radius
    p_lev: jnp.ndarray         # (nlev,)
    p_lay: jnp.ndarray         # (nlay,)
    T_lev: jnp.ndarray
    T_lay: jnp.ndarray
    z_lev: jnp.ndarray
    z_lay: jnp.ndarray
    dz: jnp.ndarray
    mu_lay: jnp.ndarray
    rho_lay: jnp.ndarray
    nd_lay: jnp.ndarray
    q_c_lay: jnp.ndarray
    vmr_lay: Dict[str, jnp.ndarray]
    contri_func: bool
    refraction_mode: int       # Refraction value
    ck_mix: int                # CkMix value (ck only)
    cloud_nk_n: jnp.ndarray    # only with cloud n,k data
    cloud_nk_k: jnp.ndarray
    is_brown_dwarf: bool       # 1D only
    stellar_flux: jnp.ndarray  # 1D only
    has_stellar_flux: jnp.ndarray


@dataclass(frozen=True)
class ModelSpec:
    """Schemes selected from ``cfg.physics`` / ``cfg.opac``."""
    geometry: str                      # "1d" | "1_5d"
    rt_scheme: str
    tp: Callable
    alt: Callable
    chem: Callable                     # raw kernel; see prepare_chemistry_kernel
    mu: Callable
    cloud_profile: Callable
    line: Optional[Callable]
    ray: Optional[Callable]
    cia: Optional[Callable]
    special: Optional[Callable]
    cloud: Optional[Callable]
    rt: Callable
    ck: bool
    ck_mix: Optional[CkMix]
    refraction: Refraction
    contri_func: bool
    is_brown_dwarf: bool
    schemes: Dict[str, str] = field(default_factory=dict)  # kind -> configured name
    cache: Dict[str, tuple] = field(default_factory=dict)  # kind -> required cache keys

    def opacity_schemes(self) -> Dict[str, str]:
        """Scheme strings in the form prepare_chemistry_kernel expects."""
        return {
            "line_opac": self.schemes["opac_line"],
            "ray_opac": self.schemes["opac_ray"],
            "cia_opac": self.schemes["opac_cia"],
            "special_opac": self.schemes["opac_special"],
        }

    def describe(self, fmt: Callable) -> Dict[str, object]:
        """Selected kernels keyed by the pre-refactor field names (for checks)."""
        return {
            "Tp_kernel": fmt(self.tp),
            "altitude_kernel": fmt(self.alt),
            "chemistry_kernel": fmt(self.chem),
            "mu_kernel": fmt(self.mu),
            "vert_cloud_kernel": fmt(self.cloud_profile),
            "line_opac_kernel": fmt(self.line),
            "ray_opac_kernel": fmt(self.ray),
            "cia_opac_kernel": fmt(self.cia),
            "cld_opac_kernel": fmt(self.cloud),
            "special_opac_kernel": fmt(self.special),
            "rt_kernel": fmt(self.rt),
            "ck": self.ck,
            "ck_mix_code_static": None if self.ck_mix is None else int(self.ck_mix),
            "contri_func_enabled": self.contri_func,
            "refraction_mode": int(self.refraction),
        }


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _scheme_name(phys, key: str, default=None, required: bool = False) -> str:
    raw = getattr(phys, key, default)
    if raw is None:
        if required:
            raise ValueError(f"physics.{key} must be specified explicitly (use 'None' to disable).")
        raw = "none"
    return str(raw).strip().lower()


def _parse_ck_mix(cfg) -> CkMix:
    raw = str(getattr(cfg.opac, "ck_mix", "RORR")).strip().upper()
    try:
        return CkMix[raw]
    except KeyError:
        raise ValueError(f"Unknown opac.ck_mix='{raw}'. Options: RORR | PRAS | TRANS") from None


def _parse_refraction(phys) -> Refraction:
    raw = getattr(phys, "refraction", None)
    s = "none" if raw is None else str(raw).strip().lower()
    if s in _OFF:
        return Refraction.OFF
    if s in ("cutoff", "refractive_cutoff", "refraction_cutoff"):
        return Refraction.CUTOFF
    raise NotImplementedError(f"Unknown physics.refraction='{raw}'. Options: none | cutoff")


def select_model(cfg, geometry: str = "1d") -> ModelSpec:
    """Select every kernel named in the config from the kernel registry."""
    phys = cfg.physics

    rt_raw = getattr(phys, "rt_scheme", None)
    if rt_raw is None or str(rt_raw).strip().lower() == "none":
        raise ValueError("physics.rt_scheme must be specified explicitly.")
    rt_scheme = str(rt_raw).strip().lower()
    emission_mode = str(getattr(phys, "emission_mode", "planet")).lower().replace(" ", "_")
    if geometry == "1_5d":
        if rt_scheme != "transit_1_5d":
            raise ValueError("build_forward_model_1_5d requires physics.rt_scheme='transit_1_5d'.")
        if emission_mode not in ("planet", "none"):
            raise NotImplementedError("transit_1_5d is only implemented for transit radiative transfer.")

    schemes = {
        "vert_Tp": getattr(phys, "vert_Tp", None) or getattr(phys, "vert_struct", None),
        "vert_alt": getattr(phys, "vert_alt", None),
        "vert_chem": getattr(phys, "vert_chem", None),
        "vert_mu": getattr(phys, "vert_mu", None),
        "vert_cloud": getattr(phys, "vert_cloud", "none") or "none",
    }
    kernels = {kind: KR.get(kind, name) for kind, name in schemes.items()}

    for kind in ("opac_line", "opac_ray", "opac_cia", "opac_cloud"):
        schemes[kind] = _scheme_name(phys, kind, required=True)
    schemes["opac_special"] = _scheme_name(phys, "opac_special", default="on")
    for kind in ("opac_line", "opac_ray", "opac_cia", "opac_cloud", "opac_special"):
        kernels[kind] = KR.get(kind, schemes[kind])
        if kernels[kind].fn is None and kind != "opac_special":
            print(f"[info] physics.{kind} is None")

    ck = schemes["opac_line"] == "ck"
    ck_mix = _parse_ck_mix(cfg) if ck else None
    mode = "os" if not ck else ("ck_trans" if ck_mix is CkMix.TRANS else "ck")
    rt_entry = KR.get("rt", f"{rt_scheme}/{mode}", "physics.rt_scheme")
    rt = rt_entry.fn
    if rt_scheme == "emission_1d":
        rt = partial(rt, emission_solver=get_emission_solver(getattr(phys, "em_scheme", "eaa")))

    return ModelSpec(
        geometry=geometry,
        rt_scheme=rt_scheme,
        tp=kernels["vert_Tp"].fn,
        alt=kernels["vert_alt"].fn,
        chem=kernels["vert_chem"].fn,
        mu=kernels["vert_mu"].fn,
        cloud_profile=kernels["vert_cloud"].fn,
        line=kernels["opac_line"].fn,
        ray=kernels["opac_ray"].fn,
        cia=kernels["opac_cia"].fn,
        special=kernels["opac_special"].fn,
        cloud=kernels["opac_cloud"].fn,
        rt=rt,
        ck=ck,
        ck_mix=ck_mix,
        refraction=_parse_refraction(phys),
        contri_func=bool(getattr(phys, "contri_func", False)),
        is_brown_dwarf=emission_mode in _BROWN_DWARF,
        schemes={k: str(v).strip().lower() for k, v in schemes.items()},
        cache={kind: k.cache for kind, k in kernels.items() if k.cache},
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _param_names(cfg) -> set:
    return {str(p.name) for p in getattr(cfg, "params", []) if getattr(p, "name", None) is not None}


def validate_model(cfg, spec: ModelSpec, opac_cache: Mapping[str, jnp.ndarray]) -> None:
    """Raise at build time on any config / kernel / data inconsistency."""
    for kind, keys in spec.cache.items():
        missing = [k for k in keys if k not in opac_cache]
        if missing:
            raise RuntimeError(
                f"physics.{kind}='{spec.schemes[kind]}' needs opacity tables that were not "
                f"loaded: {missing}. Check cfg.opac."
            )

    names = _param_names(cfg)
    transit = spec.rt_scheme.startswith("transit")
    if spec.refraction is Refraction.CUTOFF:
        if not transit:
            raise NotImplementedError("physics.refraction is only supported for transit rt_schemes.")
        a_sm = "a_sm_joint" if spec.geometry == "1_5d" else "a_sm"
        if a_sm not in names:
            raise ValueError(f"physics.refraction: cutoff requires a delta parameter '{a_sm}' "
                             "(semi-major axis in AU).")
        if "ray_master_wavelength" not in opac_cache:
            raise RuntimeError("physics.refraction: cutoff requires Rayleigh registry data (cfg.opac.ray).")

    if spec.ck_mix is CkMix.TRANS:
        if not transit:
            raise NotImplementedError("ck_mix: TRANS is only supported for transit rt_schemes.")
        if spec.contri_func:
            raise ValueError("physics.contri_func=True is not supported with opac.ck_mix=TRANS. "
                             "Use ck_mix=RORR/PRAS or disable contribution functions.")

    if spec.geometry == "1_5d":
        from .limb_asymmetry import validate_limb_parameter_names
        validate_limb_parameter_names(names)
        if "M_p" in {split_limb_tag(n)[0] for n in names}:
            raise NotImplementedError("transit_1_5d does not support M_p-based gravity inference. "
                                      "Provide explicit log_10_g_east/log_10_g_west instead.")
        if spec.contri_func:
            raise NotImplementedError("physics.contri_func=True is not supported for transit_1_5d.")
        if spec.ck_mix is CkMix.PRAS:
            raise NotImplementedError("transit_1_5d does not support ck_mix: PRAS yet.")
        if spec.ck and str(getattr(cfg.opac, "ck", False)).strip().lower() not in ("true", "1", "yes", "on"):
            raise ValueError("transit_1_5d with physics.opac_line: ck requires opac.ck: True.")


def check_model_traces(cfg, fn: Callable, *args, param_names=None) -> None:
    """Trace ``fn(params, *args)`` once so missing inputs fail at build time.

    Parameters are taken from the free entries of ``cfg.params``; a missing
    key is reported with the kernel that asked for it.
    """
    if param_names is None:
        param_names = [p.name for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]
    dummy = {name: jax.ShapeDtypeStruct((), jnp.float64) for name in param_names}
    try:
        jax.eval_shape(fn, dummy, *args)
    except KeyError as exc:
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        raise ValueError(
            f"Forward model needs {exc} (looked up in {frame.name}, "
            f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}). If this is a parameter, "
            "add it to cfg.params."
        ) from None


# ---------------------------------------------------------------------------
# Runtime arguments
#
# A jitted function bakes any array it captures into the compiled program as a
# constant, so the opacity tables are handed to jit as arguments instead.
# ---------------------------------------------------------------------------

def attach_runtime(forward_model: Callable, impl: Callable, runtime_args: tuple) -> Callable:
    """Record the jitted ``impl(params, *runtime_args)`` behind ``forward_model``."""
    forward_model.impl = impl
    forward_model.runtime_args = runtime_args
    return forward_model


def runtime_args_of(fm: Callable) -> Optional[tuple]:
    """Arrays to pass through jit for ``fm``; None if ``fm`` has none attached."""
    return getattr(fm, "runtime_args", None)


def bind_runtime(fm: Callable, runtime: Optional[tuple]) -> Callable:
    """``fm`` reading its arrays from ``runtime``, usually a jit argument."""
    if runtime is None:
        return fm
    return lambda params: fm.impl(params, *runtime)


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

def extract_fixed_params(cfg, geometry: str = "1d") -> Dict[str, jnp.ndarray]:
    """Delta-distribution parameters and physics.cloud_dist as a fixed-params dict."""
    fixed: Dict[str, jnp.ndarray] = {}
    for param in cfg.params:
        if param.dist == "delta":
            raw = getattr(param, "value", None)
            if isinstance(raw, str):
                low = raw.strip().lower()
                raw = (low == "true") if low in ("true", "false") else float(raw)
            fixed[param.name] = jnp.asarray(raw)

    cloud_dist_raw = getattr(cfg.physics, "cloud_dist", None)
    if cloud_dist_raw is not None:
        s = str(cloud_dist_raw).lower().strip()
        if s in ("1", "mono", "monodisperse"):
            code = 1
        elif s in ("2", "log_normal", "lognormal", "log-normal", "ln"):
            code = 2
        else:
            raise ValueError("physics.cloud_dist must be 'mono' or 'log_normal' (or 1/2).")
        # 1.5D shares it across limbs via the _joint tag.
        key = "cloud_dist_joint" if geometry == "1_5d" else "cloud_dist"
        fixed[key] = jnp.asarray(code, dtype=jnp.int32)
    return fixed


def make_mu_kernel(cfg, spec: ModelSpec, trace_species: tuple) -> Callable:
    """Mean-molecular-weight kernel; constant-VMR chemistry uses a fixed species list."""
    if spec.chem not in (constant_vmr, constant_vmr_clr):
        return spec.mu
    mu_mode = str(getattr(cfg.physics, "vert_mu", "auto")).lower()
    if mu_mode not in ("auto", "dynamic", "variable", "vmr"):
        return spec.mu

    base_names = {split_limb_tag(n)[0] for n in _param_names(cfg)}
    include_h = "log_10_H_over_H2" in base_names
    species = tuple(dict.fromkeys((*trace_species, "H2", "He", *(("H",) if include_h else ()))))
    compute_mu_fast = build_compute_mu(species)
    allow_const = mu_mode == "auto"

    def mu_kernel(params, vmr_lay, nlay):
        if allow_const and "mu" in params:
            return constant_mu(params, nlay)
        if "__mu_lay__" in vmr_lay:
            return vmr_lay["__mu_lay__"]
        return compute_mu_fast(vmr_lay)

    return mu_kernel


def make_atmosphere_fn(spec: ModelSpec, chem: Callable, mu: Callable, nlay: int) -> Callable:
    """Return ``atmosphere(params, wl, opac) -> (state, params)``.

    Builds the pressure grid, T-p, chemistry, mu, altitude, densities and cloud
    profile. With ``M_p`` the radius follows from mass and gravity and is
    written back as ``R_p`` so altitude kernels use the same radius as the RT.
    """
    nlev = nlay + 1

    def atmosphere(params, wl, opac):
        if "M_p" in params:
            M_p = params["M_p"] * M_jup
            g = 10.0 ** params["log_10_g"]
            R0 = jnp.sqrt((G * M_p) / g)
            params = {**params, "R_p": R0 / R_jup}
        else:
            R0 = params["R_p"] * R_jup

        p_lev = jnp.logspace(jnp.log10(params["p_bot"] * bar), jnp.log10(params["p_top"] * bar), nlev)
        p_lay = (p_lev[1:] - p_lev[:-1]) / jnp.log(p_lev[1:] / p_lev[:-1])
        T_lev, T_lay = spec.tp(p_lev, params)
        vmr_lay = chem(p_lay, T_lay, params, nlay)
        mu_lay = mu(params, vmr_lay, nlay)
        z_lev, z_lay, dz = spec.alt(p_lev, T_lay, mu_lay, params)
        rho_lay = (mu_lay * amu * p_lay) / (kb * T_lay)
        nd_lay = p_lay / (kb * T_lay)
        q_c_lay = spec.cloud_profile(p_lay, T_lay, mu_lay, rho_lay, nd_lay, params)

        state: ModelState = {
            "nwl": wl.shape[0],
            "nlay": nlay,
            "wl": wl,
            "R0": R0,
            "R_s": params["R_s"] * R_sun,
            "p_lev": p_lev,
            "p_lay": p_lay,
            "T_lev": T_lev,
            "T_lay": T_lay,
            "z_lev": z_lev,
            "z_lay": z_lay,
            "dz": dz,
            "mu_lay": mu_lay,
            "rho_lay": rho_lay,
            "nd_lay": nd_lay,
            "q_c_lay": q_c_lay,
            "vmr_lay": vmr_lay,
            "contri_func": spec.contri_func,
            "refraction_mode": int(spec.refraction),
        }
        if "cloud_nk_n" in opac:
            state["cloud_nk_n"] = opac["cloud_nk_n"]
            state["cloud_nk_k"] = opac["cloud_nk_k"]
        if spec.ck_mix is not None:
            state["ck_mix"] = int(spec.ck_mix)
        return state, params

    return atmosphere


def make_opacity_fn(spec: ModelSpec) -> Callable:
    """Return ``opacities(state, params, opac) -> components`` (disabled sources are zero)."""

    def opacities(state, params, opac):
        k_cld, ssa, g = zero_cloud_opacity(state, params)
        comps = {
            "line": zero_ck_opacity(state, opac, params) if spec.ck else zero_line_opacity(state, params),
            "rayleigh": zero_ray_opacity(state, params),
            "cia": zero_cia_opacity(state, params),
            "special": zero_special_opacity(state, params),
            "cloud": k_cld,
            "cloud_ssa": ssa,
            "cloud_g": g,
        }
        if spec.line is not None:
            if spec.ck_mix is CkMix.TRANS:
                # Per-species tables; mixing happens in the RT.
                comps["line_perspecies"], comps["vmr_perspecies"] = compute_ck_opacity_perspecies(
                    state, opac, params)
            else:
                comps["line"] = spec.line(state, opac, params)
        for key, kernel in (("rayleigh", spec.ray), ("cia", spec.cia), ("special", spec.special)):
            if kernel is not None:
                comps[key] = kernel(state, opac, params)
        if spec.cloud is not None:
            comps["cloud"], comps["cloud_ssa"], comps["cloud_g"] = spec.cloud(state, params)
        return comps

    return opacities
