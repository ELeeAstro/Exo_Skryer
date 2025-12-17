"""
help_print.py
=============
"""

from types import SimpleNamespace
from typing import Any, Dict, List


__all__ = [
    "print_cfg"
]

def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, SimpleNamespace):
        entries = [f"{k}={_format_value(v)}" for k, v in vars(value).items()]
        return ", ".join(entries) if entries else "{}"
    if isinstance(value, dict):
        entries = [f"{k}={_format_value(v)}" for k, v in value.items()]
        return ", ".join(entries) if entries else "{}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if all(isinstance(v, SimpleNamespace) for v in value):
            count = len(value)
            plural = "entry" if count == 1 else "entries"
            return f"{count} {plural}"
        if all(not isinstance(v, (list, tuple, dict, SimpleNamespace)) for v in value) and len(value) <= 5:
            return "[" + ", ".join(str(v) for v in value) + "]"
        return f"{len(value)} items"
    return str(value)


def _print_kv_table(title: str, data: Dict[str, Any]) -> None:
    """Print a simple key/value table."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    if not data:
        print("(none)")
        return

    k_width = max(len(k) for k in data.keys())
    for k, v in data.items():
        print(f"{k.ljust(k_width)}  : {_format_value(v)}")


def _print_params_table(params: List[SimpleNamespace]) -> None:
    """Print the params section as a compact table."""
    print()
    print("=" * 60)
    print("PARAMETERS")
    print("=" * 60)

    if not params:
        print("(no parameters defined)")
        return

    header = ["name", "dist", "transform", "value/low", "high", "init"]
    rows: List[List[str]] = []

    for p in params:
        name = str(getattr(p, "name", ""))
        dist = str(getattr(p, "dist", ""))
        transform = str(getattr(p, "transform", ""))

        if dist == "delta":
            val = getattr(p, "value", "")
            low_str = "" if val is None else str(val)
            high_str = ""
        else:
            low = getattr(p, "low", "")
            high = getattr(p, "high", "")
            low_str = "" if low is None else str(low)
            high_str = "" if high is None else str(high)

        init = getattr(p, "init", "")
        init_str = "" if init is None else str(init)

        rows.append([name, dist, transform, low_str, high_str, init_str])

    # column widths
    cols = list(zip(header, *rows))
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(row):
        return "  ".join(str(x).ljust(w) for x, w in zip(row, widths))

    # print
    print(fmt_row(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))


def _print_opac_list(title: str, items: List[SimpleNamespace]) -> None:
    """Print an opacity category (line/cia/ray/cloud) as a small table."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    if not items:
        print("(none)")
        return

    # Collect all keys present across items
    all_keys = set()
    for it in items:
        all_keys.update(vars(it).keys())

    # We prefer to show 'species' and 'path' first if they exist
    preferred = ["species", "path"]
    keys = [k for k in preferred if k in all_keys] + \
           [k for k in sorted(all_keys) if k not in preferred]

    header = keys
    rows: List[List[str]] = []
    for it in items:
        d = vars(it)
        rows.append([_format_value(d.get(k, "")) for k in keys])

    # column widths
    cols = list(zip(header, *rows))
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(row):
        return "  ".join(str(x).ljust(w) for x, w in zip(row, widths))

    # print
    print(fmt_row(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))


def print_cfg(cfg: SimpleNamespace) -> None:
    """
    Pretty-print the entire configuration (SimpleNamespace tree)
    in a compact, table-like format.

    Sections:
      - obs
      - physics
      - opac (with per-category tables)
      - params
      - sampling (only chosen engine)
      - runtime
    """

    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)

    # -------- OBS --------
    obs = getattr(cfg, "obs", SimpleNamespace())
    _print_kv_table(
        "OBSERVATIONS",
        {k: v for k, v in vars(obs).items()},
    )

    # -------- PHYSICS --------
    physics = getattr(cfg, "physics", SimpleNamespace())
    _print_kv_table(
        "PHYSICS",
        {k: v for k, v in vars(physics).items()},
    )

    # -------- OPAC CONFIG --------
    opac = getattr(cfg, "opac", SimpleNamespace())

    # Top-level opac keys as a quick overview
    _print_kv_table(
        "OPACITY CONFIG (overview)",
        {k: v for k, v in vars(opac).items()},
    )

    # Per-category detailed tables
    # LINE
    line_cfg = getattr(opac, "line", None)
    if isinstance(line_cfg, list):
        _print_opac_list("OPACITY [line]", line_cfg)
    else:
        _print_kv_table("OPACITY [line]", {"line": line_cfg})

    # CIA
    cia_cfg = getattr(opac, "cia", None)
    if isinstance(cia_cfg, list):
        _print_opac_list("OPACITY [cia]", cia_cfg)
    else:
        _print_kv_table("OPACITY [cia]", {"cia": cia_cfg})

    # RAY
    ray_cfg = getattr(opac, "ray", None)
    if isinstance(ray_cfg, list):
        _print_opac_list("OPACITY [ray]", ray_cfg)
    else:
        _print_kv_table("OPACITY [ray]", {"ray": ray_cfg})

    # CLOUD
    cld_cfg = getattr(opac, "cloud", None)
    if isinstance(cld_cfg, list):
        _print_opac_list("OPACITY [cloud]", cld_cfg)
    else:
        _print_kv_table("OPACITY [cloud]", {"cloud": cld_cfg})

    # -------- PARAMS (table) --------
    params = getattr(cfg, "params", []) or []
    _print_params_table(params)

    # -------- SAMPLING (only chosen engine) --------
    sampling = getattr(cfg, "sampling", SimpleNamespace())
    if vars(sampling):
        engine = getattr(sampling, "engine", None)

        # Global engine choice
        _print_kv_table(
            "SAMPLING (ENGINE CHOICE)",
            {"engine": engine} if engine is not None else {},
        )

        if engine is not None:
            chosen = getattr(sampling, engine, None)

            if isinstance(chosen, SimpleNamespace):
                chosen_dict = vars(chosen).copy()

                # Special handling for jaxns.termination
                if engine == "jaxns":
                    term = chosen_dict.pop("termination", None)

                    # Main jaxns settings (without termination)
                    _print_kv_table(
                        f"SAMPLING [{engine}]",
                        chosen_dict,
                    )

                    # Termination as its own sub-section
                    if isinstance(term, SimpleNamespace):
                        _print_kv_table(
                            "SAMPLING [jaxns] - termination",
                            vars(term),
                        )
                else:
                    # Simple engine: print its config as a flat table
                    _print_kv_table(
                        f"SAMPLING [{engine}]",
                        chosen_dict,
                    )
            else:
                _print_kv_table(
                    f"SAMPLING [{engine}]",
                    {},
                )
    else:
        _print_kv_table("SAMPLING", {})

    # -------- RUNTIME --------
    runtime = getattr(cfg, "runtime", SimpleNamespace())
    _print_kv_table(
        "RUNTIME",
        {k: v for k, v in vars(runtime).items()},
    )
