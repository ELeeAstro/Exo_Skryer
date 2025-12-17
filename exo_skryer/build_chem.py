"""
build_chem.py
=============
"""

from __future__ import annotations

from typing import Iterable

__all__ = [
    'infer_trace_species',
    'infer_log10_vmr_keys',
    'validate_log10_vmr_params'
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


def validate_log10_vmr_params(cfg, trace_species: tuple[str, ...]) -> None:
    cfg_param_names = {p.name for p in cfg.params}
    missing = [s for s in trace_species if f"log_10_f_{s}" not in cfg_param_names]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing required VMR parameters for: {joined}. "
            f"Add `log_10_f_<species>` entries to cfg.params."
        )
