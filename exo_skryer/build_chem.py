"""
build_chem.py
=============
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable, Any

__all__ = [
    'infer_trace_species',
    'infer_log10_vmr_keys',
    'infer_clr_keys',
    'validate_log10_vmr_params',
    'validate_clr_params',
    'clr_samples_to_vmr',
    'prepare_chemistry_kernel',
    'load_nasa9_if_needed',
    'init_fastchem_grid_if_needed',
    'init_element_potentials_if_needed',
    'init_atmodeller_if_needed',
]


def _extract_species_list(block) -> list[str]:
    if not block:
        return []
    if isinstance(block, bool):
        return []
    names: list[str] = []
    try:
        iterator = iter(block)
    except TypeError:
        iterator = iter((block,))
    for item in iterator:
        name = getattr(item, "species", item)
        names.append(str(name).strip())
    return names


def _append_unique(seq: list[str], name: str) -> None:
    name = str(name).strip()
    if not name:
        return
    if name not in seq:
        seq.append(name)


def infer_trace_species(
    cfg,
    line_opac_scheme_str: str,
    ray_opac_scheme_str: str,
    cia_opac_scheme_str: str,
    special_opac_scheme_str: str,
) -> tuple[str, ...]:
    required: list[str] = []

    def add_many(names: Iterable[str]) -> None:
        for n in names:
            _append_unique(required, n)

    if line_opac_scheme_str.lower() == "os":
        add_many(_extract_species_list(getattr(cfg.opac, "line", None)))
    elif line_opac_scheme_str.lower() == "ck":
        ck_mode = getattr(cfg.opac, "ck", None)
        if isinstance(ck_mode, bool):
            ck_block = getattr(cfg.opac, "line", None)
        else:
            ck_block = ck_mode
        add_many(_extract_species_list(ck_block))

    if ray_opac_scheme_str.lower() in ("os", "ck"):
        add_many(_extract_species_list(getattr(cfg.opac, "ray", None)))

    if cia_opac_scheme_str.lower() in ("os", "ck"):
        for cia_name in _extract_species_list(getattr(cfg.opac, "cia", None)):
            if cia_name == "H-":
                # H- is treated as special opacity (bound-free/free-free), not a CIA pair.
                continue
            parts = cia_name.split("-")
            if len(parts) == 2:
                # CIA pairs that include atomic hydrogen should not force H to be a
                # retrieved trace VMR; in constant-VMR modes H is derived from the
                # filler via log_10_H_over_H2.
                for p in (parts[0], parts[1]):
                    if p in ("H2", "He", "H"):
                        continue
                    _append_unique(required, p)

    if special_opac_scheme_str.lower() not in ("none", "off", "false", "0"):
        opac_cfg = getattr(cfg, "opac", None)
        special_cfg = getattr(opac_cfg, "special", None) if opac_cfg is not None else None

        hm_enabled = False
        hm_bf = True
        hm_ff = False

        # New config: cfg.opac.special includes species='H-' with optional bf/ff toggles
        if special_cfg not in (None, "None", "none", False):
            hm_enabled = True
            if not isinstance(special_cfg, bool):
                try:
                    iterator = iter(special_cfg)
                except TypeError:
                    iterator = iter((special_cfg,))
                for item in iterator:
                    name = getattr(item, "species", item)
                    if str(name).strip() != "H-":
                        continue
                    hm_bf = bool(getattr(item, "bf", hm_bf))
                    hm_ff = bool(getattr(item, "ff", hm_ff))
                    hm_enabled = True
                    break

        # Back-compat: cfg.opac.cia includes H- (treated as bf only unless ff flag set)
        if not hm_enabled:
            cia_cfg = getattr(opac_cfg, "cia", None) if opac_cfg is not None else None
            if cia_cfg not in (None, "None", "none", False):
                try:
                    iterator = iter(cia_cfg)
                except TypeError:
                    iterator = iter((cia_cfg,))
                for item in iterator:
                    name = getattr(item, "species", item)
                    if str(name).strip() != "H-":
                        continue
                    hm_enabled = True
                    hm_bf = True
                    hm_ff = bool(getattr(item, "ff", hm_ff))
                    break

        if hm_enabled and hm_bf:
            _append_unique(required, "H-")
        # For H- free-free: electrons and atomic-H are handled via separate
        # retrieved proxies (ne/n_tot and H/H2), not via the VMR machinery.

    # H2/He are always filler species, and atomic H is derived from the filler
    # when needed via the dedicated log_10_H_over_H2 parameter.
    trace_species = tuple(s for s in required if s not in ("H2", "He", "H"))
    return trace_species


def infer_active_opacity_species(cfg) -> tuple[str, ...]:
    """Infer active chemistry species directly required by configured opacities."""
    required: list[str] = []

    def add(name: str) -> None:
        _append_unique(required, name)

    def add_many(names: Iterable[str]) -> None:
        for n in names:
            add(n)

    line_mode = str(getattr(getattr(cfg, "physics", None), "opac_line", "none")).lower()
    if line_mode in ("os", "ck"):
        if line_mode == "ck":
            ck_cfg = getattr(getattr(cfg, "opac", None), "ck", None)
            block = getattr(cfg.opac, "line", None) if isinstance(ck_cfg, bool) else ck_cfg
            add_many(_extract_species_list(block))
        else:
            add_many(_extract_species_list(getattr(cfg.opac, "line", None)))

    ray_mode = str(getattr(getattr(cfg, "physics", None), "opac_ray", "none")).lower()
    if ray_mode in ("os", "ck"):
        add_many(_extract_species_list(getattr(cfg.opac, "ray", None)))

    cia_mode = str(getattr(getattr(cfg, "physics", None), "opac_cia", "none")).lower()
    if cia_mode in ("os", "ck"):
        for cia_name in _extract_species_list(getattr(cfg.opac, "cia", None)):
            if cia_name == "H-":
                continue
            parts = cia_name.split("-")
            if len(parts) == 2:
                add(parts[0])
                add(parts[1])

    special_mode = str(getattr(getattr(cfg, "physics", None), "opac_special", "none")).lower()
    if special_mode not in ("none", "off", "false", "0"):
        special_cfg = getattr(getattr(cfg, "opac", None), "special", None)
        hm_enabled = False
        hm_bf = True
        hm_ff = False
        if special_cfg not in (None, "None", "none", False):
            try:
                iterator = iter(special_cfg) if not isinstance(special_cfg, bool) else iter(())
            except TypeError:
                iterator = iter((special_cfg,))
            for item in iterator:
                spec = str(getattr(item, "species", item))
                if spec != "H-":
                    continue
                hm_enabled = True
                hm_bf = bool(getattr(item, "bf", hm_bf))
                hm_ff = bool(getattr(item, "ff", hm_ff))
                break
        if hm_enabled and hm_bf:
            add("H-")
        if hm_enabled and hm_ff:
            add("H")
            add("e-")

    return tuple(required)


def _special_hminus_requirements(cfg) -> tuple[bool, bool]:
    """Return (need_bf, need_ff) from physics/opac special configuration."""
    special_mode = str(getattr(getattr(cfg, "physics", None), "opac_special", "none")).lower()
    if special_mode in ("none", "off", "false", "0"):
        return False, False

    need_bf = False
    need_ff = False
    special_cfg = getattr(getattr(cfg, "opac", None), "special", None)
    if special_cfg not in (None, "None", "none", False):
        try:
            iterator = iter(special_cfg) if not isinstance(special_cfg, bool) else iter(())
        except TypeError:
            iterator = iter((special_cfg,))
        for item in iterator:
            spec = str(getattr(item, "species", item))
            if spec != "H-":
                continue
            need_bf = bool(getattr(item, "bf", True))
            need_ff = bool(getattr(item, "ff", False))
            return need_bf, need_ff

    # Backward-compatible behavior: if special is enabled but no explicit H- row,
    # treat bf as enabled by default.
    return True, False


def infer_log10_vmr_keys(trace_species: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"log_10_f_{s}" for s in trace_species)


def infer_clr_keys(trace_species: tuple[str, ...]) -> tuple[str, ...]:
    """Return CLR parameter keys for each trace species.

    Parameters
    ----------
    trace_species : tuple of str
        Ordered tuple of trace species names.

    Returns
    -------
    tuple of str
        Parameter keys with ``clr_`` prefix, e.g. ``('clr_H2O', 'clr_CO')``.
    """
    return tuple(f"clr_{s}" for s in trace_species)


def validate_log10_vmr_params(cfg, trace_species: tuple[str, ...]) -> None:
    cfg_param_names = {p.name for p in cfg.params}
    missing = [s for s in trace_species if f"log_10_f_{s}" not in cfg_param_names]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing required VMR parameters for: {joined}. "
            f"Add `log_10_f_<species>` entries to cfg.params."
        )


def validate_clr_params(cfg, trace_species: tuple[str, ...]) -> bool:
    """Validate that abundance parameters are present for CLR mode.

    This function accepts either CLR parameters (``clr_<species>``) or traditional
    log10 VMR parameters (``log_10_f_<species>``). When log10 VMR parameters are
    used, they will be converted to CLR coordinates internally via softmax, which
    acts as a soft constraint ensuring valid atmospheric composition.

    Parameters
    ----------
    cfg : config object
        Configuration object containing params list.
    trace_species : tuple of str
        Ordered tuple of trace species names.

    Returns
    -------
    bool
        True if using log_10_f_* parameters, False if using clr_* parameters.

    Raises
    ------
    ValueError
        If neither parameter style is found, or if mixing both styles.
    """
    cfg_param_names = {p.name for p in cfg.params}

    # Check which parameter style is being used
    has_clr = any(f"clr_{s}" in cfg_param_names for s in trace_species)
    has_log10 = any(f"log_10_f_{s}" in cfg_param_names for s in trace_species)

    if has_clr and has_log10:
        raise ValueError(
            "Cannot mix clr_* and log_10_f_* parameters. "
            "Use either CLR parameters (clr_H2O, clr_CO, ...) "
            "or log10 VMR parameters (log_10_f_H2O, log_10_f_CO, ...)"
        )

    if has_clr:
        # Validate all CLR parameters are present
        missing = [s for s in trace_species if f"clr_{s}" not in cfg_param_names]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Missing required CLR parameters for: {joined}. "
                f"Add `clr_<species>` entries to cfg.params."
            )
        return False  # Using native CLR parameters

    if has_log10:
        # Validate all log10 VMR parameters are present
        missing = [s for s in trace_species if f"log_10_f_{s}" not in cfg_param_names]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Missing required VMR parameters for: {joined}. "
                f"Add `log_10_f_<species>` entries to cfg.params."
            )
        return True  # Using log10 VMR parameters, will convert internally

    # Neither style found
    raise ValueError(
        f"No abundance parameters found for species: {', '.join(trace_species)}. "
        f"Add either `clr_<species>` or `log_10_f_<species>` entries to cfg.params."
    )


def clr_samples_to_vmr(
    samples_dict: dict[str, "np.ndarray"],
    species: tuple[str, ...] | list[str],
) -> dict[str, "np.ndarray"]:
    """Convert CLR posterior samples to physical VMR (and log10 VMR) columns.

    Applies the same softmax inverse as ``constant_vmr_clr`` but operates on
    NumPy arrays of posterior samples rather than JAX scalars.

    Parameters
    ----------
    samples_dict : dict[str, np.ndarray]
        Posterior samples keyed by parameter name.  Must contain
        ``clr_<species>`` for every species in *species*.
    species : tuple or list of str
        Ordered trace species names (e.g. ``('H2O', 'CO', 'CO2')``).

    Returns
    -------
    derived : dict[str, np.ndarray]
        New columns ready to merge into the samples dictionary:

        - ``log_10_f_<species>`` : log10(VMR) for each trace species
        - ``f_<species>``        : linear VMR for each trace species
        - ``f_H2_He``            : combined H2+He filler fraction
    """
    import numpy as np
    from scipy.special import logsumexp as _logsumexp

    species = tuple(species)
    n_samples = len(samples_dict[f"clr_{species[0]}"])

    # (N_samples, N_species) matrix of CLR values
    z = np.column_stack([samples_dict[f"clr_{s}"] for s in species])

    # Prepend z_filler = 0 column → (N_samples, N_species + 1)
    z_all = np.concatenate([np.zeros((n_samples, 1)), z], axis=1)

    # Softmax: VMR_i = exp(z_i) / sum(exp(z_j))
    log_denom = _logsumexp(z_all, axis=1, keepdims=True)
    vmr_all = np.exp(z_all - log_denom)

    # First column is filler, rest are trace species (same order as input)
    derived: dict[str, np.ndarray] = {}
    derived["f_H2_He"] = vmr_all[:, 0]

    for i, s in enumerate(species):
        vmr_i = vmr_all[:, i + 1]
        derived[f"f_{s}"] = vmr_i
        derived[f"log_10_f_{s}"] = np.log10(np.maximum(vmr_i, 1e-300))

    return derived


def prepare_chemistry_kernel(cfg, chemistry_kernel, opacity_schemes: dict):
    """Prepare and validate chemistry kernel with inferred species.

    This function infers the required trace species from the opacity configuration,
    validates that necessary parameters are present, and returns an optimized
    chemistry kernel ready for use in the forward model.

    Parameters
    ----------
    cfg : config object
        Configuration object containing params and opac settings.
    chemistry_kernel : callable
        The base chemistry kernel function from vert_chem (e.g., constant_vmr,
        CE_fastchem_jax, CE_rate_jax).
    opacity_schemes : dict
        Dictionary containing opacity scheme strings with keys:
        'line_opac', 'ray_opac', 'cia_opac', 'special_opac'.

    Returns
    -------
    chemistry_kernel : callable
        The prepared chemistry kernel, potentially optimized for JIT compilation.
    trace_species : tuple of str
        Tuple of trace species names inferred from the opacity configuration.
    """
    from .vert_chem import constant_vmr, constant_vmr_clr

    # Infer required species from opacity configuration
    trace_species = infer_trace_species(
        cfg,
        line_opac_scheme_str=opacity_schemes['line_opac'],
        ray_opac_scheme_str=opacity_schemes['ray_opac'],
        cia_opac_scheme_str=opacity_schemes['cia_opac'],
        special_opac_scheme_str=opacity_schemes['special_opac'],
    )

    # If configured opacities require atomic H (CIA H2-H / He-H pairs, or H- ff),
    # constant-VMR chemistry derives H from the H2+He filler using log_10_H_over_H2.
    need_atomic_h = False
    opac_cfg = getattr(cfg, "opac", None)
    cia_names = _extract_species_list(getattr(opac_cfg, "cia", None) if opac_cfg is not None else None)
    for name in cia_names:
        parts = str(name).strip().split("-")
        if len(parts) == 2 and "H" in parts and "H-" not in parts:
            need_atomic_h = True
            break
    if not need_atomic_h:
        special_cfg = getattr(opac_cfg, "special", None) if opac_cfg is not None else None
        if special_cfg not in (None, "None", "none", False):
            try:
                iterator = iter(special_cfg) if not isinstance(special_cfg, bool) else iter(())
            except TypeError:
                iterator = iter((special_cfg,))
            for item in iterator:
                spec = getattr(item, "species", item)
                if str(spec).strip() != "H-":
                    continue
                if bool(getattr(item, "ff", False)):
                    need_atomic_h = True
                break

    if need_atomic_h:
        from .vert_chem import constant_vmr, constant_vmr_clr

        if chemistry_kernel in (constant_vmr, constant_vmr_clr):
            cfg_param_names = {p.name for p in cfg.params}
            if "log_10_H_over_H2" not in cfg_param_names:
                raise ValueError(
                    "Atomic H is required by the configured CIA/special opacities, but parameter "
                    "'log_10_H_over_H2' is missing. Add it to cfg.params to derive H from the "
                    "H2+He filler (constant_vmr/constant_vmr_clr)."
                )

    # For constant VMR: validate and build optimized kernel
    if chemistry_kernel is constant_vmr:
        validate_log10_vmr_params(cfg, trace_species)
        chemistry_kernel = constant_vmr(trace_species)
    elif chemistry_kernel is constant_vmr_clr:
        use_log10_vmr = validate_clr_params(cfg, trace_species)
        chemistry_kernel = constant_vmr_clr(trace_species, use_log10_vmr=use_log10_vmr)

    return chemistry_kernel, trace_species


def load_nasa9_if_needed(cfg: Any, exp_dir: Path) -> None:
    """Load NASA-9 thermo coefficients if RateJAX chemistry requires it.

    This function checks if the configured chemistry scheme requires Gibbs free
    energy data (RateJAX chemical equilibrium modes) and loads the NASA-9 tables
    if needed. If data is already loaded or not required, it returns immediately.

    Parameters
    ----------
    cfg : config object
        Parsed YAML configuration object with `cfg.physics.vert_chem` attribute.
    exp_dir : `~pathlib.Path`
        Experiment directory used to resolve relative paths to NASA-9 data.
    """
    phys = getattr(cfg, "physics", None)
    if phys is None:
        return

    vert_chem_raw = getattr(phys, "vert_chem", None)
    if vert_chem_raw is None:
        return

    vert_chem_name = str(vert_chem_raw).lower()
    ep_names = ("easychem_jax", "easychem")
    fc_grid_names = (
        "fastchem_grid_jax",
        "ce_fastchem_grid",
        "fastchem_ce_grid",
        "ce",
        "chemical_equilibrium",
        "ce_fastchem_jax",
        "fastchem_jax",
    )
    needs_nasa9 = (
        "rate_ce",
        "rate_jax",
        "ce_rate_jax",
        *ep_names,
    )

    if vert_chem_name in fc_grid_names:
        init_fastchem_grid_if_needed(cfg, exp_dir)
        return

    if vert_chem_name not in needs_nasa9:
        return

    from .rate_jax import is_nasa9_cache_loaded, load_nasa9_cache

    if is_nasa9_cache_loaded():
        print("[info] NASA-9 cache already loaded")
    else:
        data_cfg = getattr(cfg, "data", None)
        nasa9_rel_path = getattr(data_cfg, "nasa9", None) if data_cfg is not None else None
        if nasa9_rel_path is None:
            raise ValueError(
                "NASA-9 data path not found in config. Please add 'nasa9: path/to/NASA9' "
                "under 'data:' section in YAML config."
            )

        base_dir = Path.cwd() if exp_dir is None else Path(exp_dir)
        nasa9_path = (
            str(base_dir / nasa9_rel_path)
            if not Path(nasa9_rel_path).is_absolute()
            else nasa9_rel_path
        )

        print(f"[info] Loading NASA-9 thermo tables from {nasa9_path}")
        thermo = load_nasa9_cache(nasa9_path)
        print(f"[info] NASA-9 cache loaded: {len(thermo.data)} species")

    # EP backend needs its own model cache as well; initialize it here so callers
    # that only invoke `load_nasa9_if_needed` (e.g. bestfit scripts) still work.
    if vert_chem_name in ep_names:
        init_element_potentials_if_needed(cfg, exp_dir)
        return


def init_fastchem_grid_if_needed(cfg: Any, exp_dir: Path) -> None:
    """Initialise and cache the FastChem 5D grid chemistry backend if required."""
    phys = getattr(cfg, "physics", None)
    if phys is None:
        return

    vert_chem_name = str(getattr(phys, "vert_chem", "") or "").lower()
    aliases = (
        "fastchem_grid_jax",
        "ce_fastchem_grid",
        "fastchem_ce_grid",
        "ce",
        "chemical_equilibrium",
        "ce_fastchem_jax",
        "fastchem_jax",
    )
    if vert_chem_name not in aliases:
        return

    from .vert_chem import (
        is_fastchem_grid_cache_loaded,
        load_fastchem_grid_cache,
        get_fastchem_grid_cache_info,
    )

    if is_fastchem_grid_cache_loaded():
        print("[info] FastChem-grid cache already loaded")
        return

    fc_cfg = getattr(cfg, "fastchem_grid_jax", None)
    if fc_cfg is None:
        raise ValueError(
            "fastchem_grid_jax config block is required when physics.vert_chem "
            "uses the FastChem-grid backend."
        )

    grid_path_raw = getattr(fc_cfg, "grid_path", None)
    if not grid_path_raw:
        raise ValueError("fastchem_grid_jax.grid_path is required and must point to a 5D NPZ grid.")

    grid_path_obj = Path(str(grid_path_raw))
    if grid_path_obj.is_absolute():
        grid_path = grid_path_obj
    else:
        base_dir = Path.cwd() if exp_dir is None else Path(exp_dir)
        candidate_exp = (base_dir / grid_path_obj).resolve()
        candidate_repo = (Path.cwd() / grid_path_obj).resolve()
        if candidate_exp.exists():
            grid_path = candidate_exp
        else:
            grid_path = candidate_repo
    if not grid_path.exists():
        raise FileNotFoundError(f"FastChem grid file not found: {grid_path}")

    bounds_cfg = getattr(fc_cfg, "bounds", None)
    bounds_mode = str(getattr(bounds_cfg, "mode", "clip")).lower() if bounds_cfg is not None else "clip"
    if bounds_mode != "clip":
        raise ValueError("fastchem_grid_jax.bounds.mode currently supports only: clip")

    param_names = {p.name for p in getattr(cfg, "params", [])}
    for required in ("M_to_H", "C_to_O"):
        if required not in param_names:
            raise ValueError(
                f"fastchem_grid_jax requires parameter '{required}' in cfg.params."
            )

    species_out = infer_active_opacity_species(cfg)
    if not species_out:
        raise ValueError(
            "Could not infer any active opacity species for fastchem_grid_jax. "
            "Check opac.line/opac.ray/opac.cia/opac.special configuration."
        )

    solver_cfg = getattr(fc_cfg, "solver", None)
    solver_kwargs: dict[str, Any] = {}
    if solver_cfg is not None:
        mode = getattr(solver_cfg, "mode", None)
        if mode is not None:
            solver_kwargs["mode"] = str(mode)
    if "mode" not in solver_kwargs:
        solver_kwargs["mode"] = "vmap"

    species_map_override: dict[str, str] = {}
    map_cfg = getattr(fc_cfg, "species_map", None)
    if map_cfg is not None:
        if isinstance(map_cfg, dict):
            items = map_cfg.items()
        else:
            items = vars(map_cfg).items()
        for key, val in items:
            species_map_override[str(key)] = str(val)

    print("[info] Initializing CE scheme: fastchem_grid_jax")
    print(f"[info]   Grid path: {grid_path}")
    print(f"[info]   Output species source: active opacity species ({len(species_out)})")
    print(f"[info]   Solver: mode={solver_kwargs['mode']}")
    print(f"[info]   Bounds: mode=clip")

    load_fastchem_grid_cache(
        grid_path=str(grid_path),
        species_out=list(species_out),
        solver_cfg=solver_kwargs,
        species_map_override=species_map_override if species_map_override else None,
    )
    info = get_fastchem_grid_cache_info()
    if info:
        shape = info.get("shape", ())
        print(
            "[info]   Grid axes: "
            f"T={shape[0]}, P={shape[1]}, M/H={shape[2]}, C/O={shape[3]}, species={shape[4]}"
        )
        print(
            "[info]   Ranges: "
            f"T=[{info['T_range'][0]:.1f}, {info['T_range'][1]:.1f}] K, "
            f"P=[{info['P_range'][0]:.3e}, {info['P_range'][1]:.3e}] bar, "
            f"M/H=[{info['MH_range'][0]:.2f}, {info['MH_range'][1]:.2f}] dex, "
            f"C/O=[{info['CO_range'][0]:.3f}, {info['CO_range'][1]:.3f}]"
        )
        print(f"[info]   Interp axes: {'log10(T,P,C/O)+dex(M/H)' if info.get('use_log_axes', False) else 'linear'}")
        mapped_species = tuple(info.get("species_out", ()))
        print(f"[info]   Mapped species: {len(mapped_species)}")
        if mapped_species:
            print(f"[info]   Species list: {list(mapped_species)}")
        if info.get("unmapped_species"):
            print(f"[warn]   Unmapped species skipped: {list(info['unmapped_species'])}")

        need_bf, need_ff = _special_hminus_requirements(cfg)
        required_special: list[str] = []
        if need_bf:
            required_special.append("H-")
        if need_ff:
            required_special.extend(("H", "e-"))
        if required_special:
            missing_special = [sp for sp in required_special if sp not in mapped_species]
            if missing_special:
                raise ValueError(
                    "H- special opacity is enabled, but required species are not mapped "
                    "from the FastChem CE grid. "
                    f"Missing={missing_special}, mapped={list(mapped_species)}. "
                    "Fix via fastchem_grid_jax.species_map and/or grid contents."
                )
    print("[info] FastChem-grid cache loaded")


def init_element_potentials_if_needed(cfg: Any, exp_dir: Path) -> None:
    """Initialise and cache the element-potentials chemistry backend if required."""
    phys = getattr(cfg, "physics", None)
    if phys is None:
        return

    vert_chem_name = str(getattr(phys, "vert_chem", "") or "").lower()
    if vert_chem_name not in ("easychem_jax", "easychem"):
        return

    from .vert_chem import is_element_potentials_cache_loaded, load_element_potentials_cache

    if is_element_potentials_cache_loaded():
        print("[info] Element-potentials cache already loaded")
        return

    ep_cfg = getattr(cfg, "easychem_jax", None)
    if ep_cfg is None:
        raise ValueError(
            "easychem_jax config block is required when physics.vert_chem is "
            "'easychem_jax'."
        )

    raw_species = list(getattr(ep_cfg, "species", None) or [])
    species_list = []
    for sp in raw_species:
        if isinstance(sp, bool):
            # YAML 1.1 parses unquoted `NO`/`Yes` as booleans; recover the common NO case.
            species_list.append("NO" if sp is False else "YES")
        else:
            species_list.append(str(sp))
    if not species_list:
        raise ValueError(
            "easychem_jax.species must be a non-empty list in the YAML config."
        )

    elements_val = getattr(ep_cfg, "elements", None)
    elements = list(elements_val) if elements_val else None

    solver_cfg = getattr(ep_cfg, "solver", None)
    solver_kwargs = {}
    if solver_cfg is not None:
        for key in ("mode", "max_steps", "tol", "throw", "relax_limit", "prefer_chord"):
            val = getattr(solver_cfg, key, None)
            if val is not None:
                solver_kwargs[key] = val

    data_cfg = getattr(cfg, "data", None)
    nasa9_rel_path = getattr(data_cfg, "nasa9", None) if data_cfg is not None else None
    if nasa9_rel_path is None:
        raise ValueError(
            "NASA-9 data path not found in config. Please add 'nasa9: path/to/NASA9' "
            "under 'data:' section in YAML config."
        )
    base_dir = Path.cwd() if exp_dir is None else Path(exp_dir)
    nasa9_path = (
        str(base_dir / nasa9_rel_path)
        if not Path(nasa9_rel_path).is_absolute()
        else nasa9_rel_path
    )

    p0_bar = float(getattr(ep_cfg, "p0_bar", 1.0))
    e_ref = str(getattr(ep_cfg, "e_ref", "H"))
    nlay = int(getattr(phys, "nlay", 99))

    print("[info] Initializing CE scheme: easychem_jax")
    print(f"[info]   NASA9 path: {nasa9_path}")
    print(f"[info]   Species count: {len(species_list)}")
    print(f"[info]   Elements: {elements if elements is not None else 'auto (inferred)'}")
    print(f"[info]   e_ref: {e_ref}, P0_bar: {p0_bar}")
    print(
        "[info]   Solver: "
        f"mode={solver_kwargs.get('mode', 'scan')}, "
        f"max_steps={solver_kwargs.get('max_steps', 64)}, "
        f"tol={solver_kwargs.get('tol', 1.0e-11)}, "
        f"throw={solver_kwargs.get('throw', False)}, "
        f"relax_limit={solver_kwargs.get('relax_limit', 0.75)}"
    )
    load_element_potentials_cache(
        species_list=species_list,
        elements=elements,
        nlay=nlay,
        solver_kwargs=solver_kwargs,
        nasa9_dir=nasa9_path,
        p0_bar=p0_bar,
        e_ref=e_ref,
    )
    print("[info] Element-potentials cache loaded")


def init_atmodeller_if_needed(cfg: Any, exp_dir: Path) -> None:
    """Initialise the global atmodeller :class:`EquilibriumModel` if required.

    Reads ``atmodeller.species_network`` from the YAML config and builds the
    cached :class:`EquilibriumModel` instance.  If the cache is already loaded,
    or the active chemistry scheme is not ``'atmodeller'``, returns immediately.

    Parameters
    ----------
    cfg : config object
        Parsed YAML configuration object with ``cfg.physics.vert_chem`` and
        ``cfg.atmodeller.species_network`` attributes.
    exp_dir : `~pathlib.Path`
        Experiment directory (unused; kept for API symmetry with
        :func:`load_nasa9_if_needed`).
    """
    phys = getattr(cfg, "physics", None)
    if phys is None:
        return

    vert_chem_name = str(getattr(phys, "vert_chem", "") or "").lower()
    if vert_chem_name != "atmodeller":
        return

    if importlib.util.find_spec("atmodeller") is None:
        raise ImportError(
            "physics.vert_chem is set to 'atmodeller', but the optional 'atmodeller' "
            "package is not installed."
        )

    atm_cfg = getattr(cfg, "atmodeller", None)
    if atm_cfg in (False, "False", "false", "off", "none", "None"):
        print("[info] Atmodeller explicitly disabled in YAML; skipping atmodeller initialization.")
        return

    from .vert_chem import is_atmodeller_cache_loaded, load_atmodeller_cache

    if is_atmodeller_cache_loaded():
        print("[info] Atmodeller cache already loaded")
        return

    species_list = list(getattr(atm_cfg, "species_network", None) or [])
    if not species_list:
        raise ValueError(
            "atmodeller.species_network must be a non-empty list in the YAML config. "
            "Example:\n  atmodeller:\n    species_network:\n      - H2O_g\n      - CO_g"
        )

    # Parse optional solver tuning from atmodeller.solver (all fields optional)
    solver_cfg = getattr(atm_cfg, "solver", None)
    solver_kwargs = {}
    if solver_cfg is not None:
        for key in ("atol", "rtol", "max_steps", "multistart", "multistart_perturbation", "jac"):
            val = getattr(solver_cfg, key, None)
            if val is not None:
                solver_kwargs[key] = val

    nlay = int(cfg.physics.nlay)
    print(f"[info] Building atmodeller EquilibriumModel with {len(species_list)} species, nlay={nlay}")
    if solver_kwargs:
        print(f"[info] Atmodeller solver settings: {solver_kwargs}")
    load_atmodeller_cache(species_list, nlay, solver_kwargs=solver_kwargs)
    print("[info] Atmodeller cache loaded")
