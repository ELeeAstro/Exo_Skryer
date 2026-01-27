"""
build_chem.py
=============
"""

from __future__ import annotations

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
    'load_nasa9_if_needed'
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

    if line_opac_scheme_str.lower() == "lbl":
        add_many(_extract_species_list(getattr(cfg.opac, "line", None)))
    elif line_opac_scheme_str.lower() == "ck":
        ck_mode = getattr(cfg.opac, "ck", None)
        if isinstance(ck_mode, bool):
            ck_block = getattr(cfg.opac, "line", None)
        else:
            ck_block = ck_mode
        add_many(_extract_species_list(ck_block))

    if ray_opac_scheme_str.lower() in ("lbl", "ck"):
        add_many(_extract_species_list(getattr(cfg.opac, "ray", None)))

    if cia_opac_scheme_str.lower() in ("lbl", "ck"):
        for cia_name in _extract_species_list(getattr(cfg.opac, "cia", None)):
            if cia_name == "H-":
                _append_unique(required, "H-")
                continue
            parts = cia_name.split("-")
            if len(parts) == 2:
                _append_unique(required, parts[0])
                _append_unique(required, parts[1])

    if special_opac_scheme_str.lower() not in ("none", "off", "false", "0"):
        for cia_name in _extract_species_list(getattr(cfg.opac, "cia", None)):
            if cia_name == "H-":
                _append_unique(required, "H-")

    trace_species = tuple(s for s in required if s not in ("H2", "He"))
    return trace_species


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
    if vert_chem_name not in ("rate_ce", "rate_jax", "ce_rate_jax"):
        return

    from .rate_jax import is_nasa9_cache_loaded, load_nasa9_cache

    if is_nasa9_cache_loaded():
        print("[info] NASA-9 cache already loaded")
        return

    data_cfg = getattr(cfg, "data", None)
    nasa9_rel_path = getattr(data_cfg, "nasa9", None) if data_cfg is not None else None
    if nasa9_rel_path is None:
        raise ValueError(
            "NASA-9 data path not found in config. Please add 'nasa9: path/to/NASA9' "
            "under 'data:' section in YAML config."
        )

    nasa9_path = (
        str(exp_dir / nasa9_rel_path)
        if not Path(nasa9_rel_path).is_absolute()
        else nasa9_rel_path
    )

    print(f"[info] Loading NASA-9 thermo tables from {nasa9_path}")
    thermo = load_nasa9_cache(nasa9_path)
    print(f"[info] NASA-9 cache loaded: {len(thermo.data)} species")
