"""
profile_io.py
=============

User-supplied 1D atmospheric profiles for forward models: T-p-mu-Kzz,
volume mixing ratios and condensate clouds, read from text files and used by
the ``file`` kernels (``vert_Tp: file``, ``vert_mu: file``, ``vert_chem:
file``, ``vert_cloud: file``, ``opac_cloud: file``).

Files are whitespace-separated text with a header line naming the columns
(it may start with ``#``; other ``#`` lines are comments). Pressure is in bar
(``p_bar``). Rows may be in any order.

- T-p-mu-Kzz file, one row per layer: ``p_bar  T_K  [mu_amu]  [Kzz_cm2s]``,
  mu the mean molecular weight (g/mol), Kzz in cm^2 s^-1. Its pressures are
  the model layers. ``vert_mu: file`` uses its mu column.
- VMR file, one row per layer: ``p_bar  <species> ...``.
- Cloud file, long format, one row per layer and species:
  ``p_bar  species  q_c  reff_um  [sigma]``, with q_c the condensate mass per
  mass of gas, reff_um the effective (area-weighted mean) radius in microns and
  sigma the lognormal geometric width.

Config (``profile:`` section)::

    profile:
      tp_file: ../../profiles/my_TpmuKzz.txt
      vmr_file: ../../profiles/my_vmr.txt
      cloud_file: ../../profiles/my_clouds.txt
      cloud_method: lxmie       # lxmie | madt | f18
      cloud_dist: lognormal     # lognormal | mono
      cloud_sigma: 2.0          # width where the cloud file has no sigma column
      cloud_nr: 400             # radii in the lognormal efficiency table
      f18: {Q0: 20.0, Q1: 1.0, a: 4.0}

and per cloud species in ``opac.cloud``: ``{species, rho, path}`` (``path``,
the n,k file, for lxmie/madt; optional ``Q0``/``Q1``/``a`` override ``profile.f18``).

All profile data are loaded once at setup. The VMR and cloud profiles are
interpolated onto the model layers in log p (log values, so mixing ratios
spanning decades interpolate sensibly). Outside a file's pressure range the
VMRs keep their end values, and the clouds are zero. The cloud optical properties (per gram of condensate)
are computed once at setup on the cloud file's pressures; each forward-model
call only interpolates them to the layers and sums the species.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import jax
import jax.numpy as jnp

from .data_constants import bar

__all__ = [
    "init_profiles",
    "has_tp_profile",
    "has_kzz_profile",
    "profile_grid",
    "profile_tp",
    "profile_kzz",
    "profile_mu",
    "file_mu_kernel",
    "vmr_species",
    "file_tp_kernel",
    "file_chem_kernel",
    "file_cloud_profile_kernel",
    "file_cloud_opacity_kernel",
    "runtime_cache",
    "cloud_species",
    "CLOUD_CACHE_KEYS",
]

_LOG_FLOOR = 1.0e-30           # floor for log interpolation of VMR / q_c (zeros stay zero)
_LOGNORMAL_N_NODES = 41        # as opacity_cloud._lognormal_cloud
_LOGNORMAL_Z_MAX = 5.0

CLOUD_CACHE_KEYS = ("profile_cloud_kext", "profile_cloud_ksca", "profile_cloud_kscag")

_STATE: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------
def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def read_table(path: Path) -> Dict[str, np.ndarray]:
    """Read a whitespace table with a header line; returns {column: array}.

    The header is the first non-comment line if it is not numeric, else the
    last ``#`` line before the data. Numeric columns become float arrays,
    others string arrays.
    """
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    header = None
    last_comment = None
    rows = []
    for line in lines:
        if not line:
            continue
        if line.startswith("#"):
            if not rows and header is None:
                last_comment = line.lstrip("#").strip()
            continue
        toks = line.split()
        if header is None and not rows and not _is_number(toks[0]):
            header = toks
            continue
        rows.append(toks)
    if header is None:
        header = last_comment.split() if last_comment else None
    if header is None:
        raise ValueError(f"{path}: no header line naming the columns.")
    if not rows:
        raise ValueError(f"{path}: no data rows.")
    if any(len(r) != len(header) for r in rows):
        bad = next(i for i, r in enumerate(rows) if len(r) != len(header))
        raise ValueError(f"{path}: data row {bad + 1} has {len(rows[bad])} values, header has {len(header)} columns.")
    if len(set(header)) != len(header):
        raise ValueError(f"{path}: duplicate column names in header {header}.")
    cols = {}
    for j, name in enumerate(header):
        vals = [r[j] for r in rows]
        try:
            cols[name] = np.asarray([float(v) for v in vals], dtype=np.float64)
        except ValueError:
            cols[name] = np.asarray(vals)
    return cols


def _resolve(path, exp_dir) -> Path:
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() if exp_dir is None else Path(exp_dir)) / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Profile file not found: {p}")
    return p


def _pressure(cols, path) -> np.ndarray:
    if "p_bar" not in cols:
        raise ValueError(f"{path}: needs a 'p_bar' column (pressure in bar).")
    p = cols["p_bar"]
    if p.dtype.kind != "f" or np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise ValueError(f"{path}: p_bar must be positive numbers.")
    return p


# ---------------------------------------------------------------------------
# Loading (setup)
# ---------------------------------------------------------------------------
def _load_tp(path: Path) -> None:
    cols = read_table(path)
    p = _pressure(cols, path)
    if "T_K" not in cols:
        raise ValueError(f"{path}: needs a 'T_K' column.")
    unknown = set(cols) - {"p_bar", "T_K", "mu_amu", "Kzz_cm2s"}
    if unknown:
        raise ValueError(f"{path}: unknown columns {sorted(unknown)} "
                         "(expected p_bar, T_K, optional mu_amu and Kzz_cm2s).")
    order = np.argsort(-p)                      # bottom (high p) first, as the model grid
    p_lay = p[order]
    if np.any(np.diff(p_lay) >= 0):
        raise ValueError(f"{path}: duplicate pressures.")
    if len(p_lay) < 2:
        raise ValueError(f"{path}: needs at least 2 layers.")
    T_lay = cols["T_K"][order]
    if np.any(~np.isfinite(T_lay)) or np.any(T_lay <= 0):
        raise ValueError(f"{path}: T_K must be positive.")

    # Levels: geometric means between layers; the two end levels half a layer
    # beyond the end layers in log p. T on the levels: PCHIP in log p inside,
    # linear extrapolation from the end layers at the two end levels.
    lp = np.log(p_lay)
    lp_lev = np.empty(len(lp) + 1)
    lp_lev[1:-1] = 0.5 * (lp[:-1] + lp[1:])
    lp_lev[0] = lp[0] - 0.5 * (lp[1] - lp[0])
    lp_lev[-1] = lp[-1] + 0.5 * (lp[-1] - lp[-2])
    T_lev = np.empty_like(lp_lev)
    if len(lp) >= 3:
        # Monotone cubic (PCHIP) in log p: no overshoot, closer than linear for smooth profiles
        from scipy.interpolate import PchipInterpolator
        T_lev[1:-1] = PchipInterpolator(lp[::-1], T_lay[::-1])(lp_lev[1:-1])
    else:
        T_lev[1:-1] = 0.5 * (T_lay[:-1] + T_lay[1:])
    T_lev[0] = T_lay[0] + (T_lay[0] - T_lay[1]) * (lp_lev[0] - lp[0]) / (lp[0] - lp[1])
    T_lev[-1] = T_lay[-1] + (T_lay[-1] - T_lay[-2]) * (lp_lev[-1] - lp[-1]) / (lp[-1] - lp[-2])
    T_lev = np.maximum(T_lev, 1.0)

    kzz = None
    if "Kzz_cm2s" in cols:
        kzz = cols["Kzz_cm2s"][order]
        if np.any(~np.isfinite(kzz)) or np.any(kzz <= 0):
            raise ValueError(f"{path}: Kzz_cm2s must be positive.")

    mu = None
    if "mu_amu" in cols:
        mu = cols["mu_amu"][order]
        if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
            raise ValueError(f"{path}: mu_amu must be positive.")

    _STATE["tp"] = {
        "p_lay": p_lay * bar, "p_lev": np.exp(lp_lev) * bar,
        "T_lay": T_lay, "T_lev": T_lev, "mu_lay": mu, "Kzz_lay": kzz, "path": str(path),
    }
    what = "T-p" + ("-mu" if mu is not None else "") + ("-Kzz" if kzz is not None else "")
    print(f"[profile] {what} from {path.name}: {len(p_lay)} layers, "
          f"p = {p_lay.min():.3g}-{p_lay.max():.3g} bar, T = {T_lay.min():.0f}-{T_lay.max():.0f} K")


def _load_vmr(path: Path) -> None:
    from .vert_mu import _SPECIES_MASS

    cols = read_table(path)
    p = _pressure(cols, path)
    order = np.argsort(p)                       # increasing p for interpolation
    vmr = {}
    for name, v in cols.items():
        if name == "p_bar":
            continue
        if v.dtype.kind != "f":
            raise ValueError(f"{path}: column '{name}' is not numeric.")
        v = v[order]
        if np.any(~np.isfinite(v)) or np.any(v < 0) or np.any(v > 1):
            raise ValueError(f"{path}: VMR column '{name}' must be in [0, 1].")
        if name not in _SPECIES_MASS and name != "e-":
            raise ValueError(f"{path}: species '{name}' has no molecular weight in data_constants "
                             "(needed for the mean molecular weight).")
        vmr[name] = v
    if not vmr:
        raise ValueError(f"{path}: no species columns.")
    total = sum(vmr.values())
    if np.any(total > 1.0 + 1e-6):
        raise ValueError(f"{path}: VMRs sum to more than 1 (max {total.max():.6f}).")
    has_bg = "H2" in vmr or "He" in vmr
    _STATE["vmr"] = {"logp": np.log10(p[order]), "vmr": vmr, "fill": not has_bg, "path": str(path)}
    print(f"[profile] VMR from {path.name}: {len(p)} rows, species {sorted(vmr)}"
          + ("; H2/He filled from 1 - sum(VMR) (solar He/H2)" if not has_bg else ""))


def _cloud_species_cfg(cfg) -> Dict[str, object]:
    raw = getattr(getattr(cfg, "opac", None), "cloud", None)
    if raw in (None, "None", "none"):
        return {}
    entries = raw if isinstance(raw, list) else [raw]
    return {str(getattr(e, "species")): e for e in entries}


def _nk_on_grid(path: Path, wl: np.ndarray):
    from .registry_cloud import read_cloud_nk
    n, k, _ = read_cloud_nk(path, wl)
    return n, k


def _efficiency_table(method: str, wl: np.ndarray, r_cm: np.ndarray, n=None, k=None, f18=None):
    """(Q_ext, Q_sca, g), each (nr, nwl), from the chosen method at radii r_cm."""
    from .opacity_cloud import compute_cloud_efficiencies

    wl_j = jnp.asarray(wl)
    scheme = {"lxmie": "lxmie", "madt": "mie_madt", "f18": "f18"}[method]
    params = {}
    if f18 is not None:
        params = {"cld_Q0": jnp.asarray(f18["Q0"]), "cld_Q1": jnp.asarray(f18["Q1"]), "cld_a": jnp.asarray(f18["a"])}
    nk = {} if n is None else {"n": jnp.asarray(n), "k": jnp.asarray(k)}
    one = jax.jit(lambda r: compute_cloud_efficiencies(wl_j, r, params, eff_scheme=scheme, **nk))
    out = [np.asarray(x) for x in zip(*[jax.device_get(one(jnp.asarray(r))) for r in r_cm])]
    return tuple(np.stack(x, axis=0) for x in out)


def _interp_logr(table: np.ndarray, log_r_tab: np.ndarray, log_r: np.ndarray, log_values: bool) -> np.ndarray:
    """Interpolate (nr, nwl) tables in log r at points log_r (m,) -> (m, nwl).

    With ``log_values`` the interpolation is in log Q (exact for power laws such
    as the small-particle Q ~ r^4 regime); otherwise linear in Q (used for g).
    """
    i = np.clip(np.searchsorted(log_r_tab, log_r) - 1, 0, len(log_r_tab) - 2)
    t = np.clip((log_r - log_r_tab[i]) / (log_r_tab[i + 1] - log_r_tab[i]), 0.0, 1.0)[:, None]
    if log_values:
        lo, hi = table[i], table[i + 1]
        pos = (lo > 0) & (hi > 0)
        out = np.exp((1.0 - t) * np.log(np.where(pos, lo, 1.0)) + t * np.log(np.where(pos, hi, 1.0)))
        return np.where(pos, out, (1.0 - t) * lo + t * hi)
    return (1.0 - t) * table[i] + t * table[i + 1]


def _load_cloud(path: Path, cfg, wl: np.ndarray) -> None:
    pcfg = getattr(cfg, "profile", None)
    method = str(getattr(pcfg, "cloud_method", "lxmie")).strip().lower()
    if method not in ("lxmie", "madt", "f18"):
        raise ValueError(f"profile.cloud_method='{method}'. Options: lxmie | madt | f18")
    dist = str(getattr(pcfg, "cloud_dist", "lognormal")).strip().lower().replace("_", "")
    if dist not in ("lognormal", "mono"):
        raise ValueError(f"profile.cloud_dist='{dist}'. Options: lognormal | mono")
    sigma_default = float(getattr(pcfg, "cloud_sigma", 2.0))
    nr = int(getattr(pcfg, "cloud_nr", 400))
    f18_default = getattr(pcfg, "f18", None)

    cols = read_table(path)
    p = _pressure(cols, path)
    for need in ("species", "q_c", "reff_um"):
        if need not in cols:
            raise ValueError(f"{path}: needs columns p_bar, species, q_c, reff_um (and optional sigma).")
    unknown = set(cols) - {"p_bar", "species", "q_c", "reff_um", "sigma"}
    if unknown:
        raise ValueError(f"{path}: unknown columns {sorted(unknown)}.")
    species_col = cols["species"].astype(str)
    q_c, reff = cols["q_c"], cols["reff_um"]
    sigma = cols["sigma"] if "sigma" in cols else np.full_like(q_c, sigma_default)
    if np.any(q_c < 0) or np.any(~np.isfinite(q_c)):
        raise ValueError(f"{path}: q_c must be >= 0.")
    if np.any(reff <= 0):
        raise ValueError(f"{path}: reff_um must be > 0.")
    if np.any(sigma < 1.0):
        raise ValueError(f"{path}: sigma must be >= 1.")

    p_all = np.unique(p)                        # increasing
    species = [str(s) for s in dict.fromkeys(species_col)]
    spec_cfg = _cloud_species_cfg(cfg)
    missing = [s for s in species if s not in spec_cfg]
    if missing:
        raise ValueError(f"{path}: cloud species {missing} have no opac.cloud entry (rho, and n,k path for {method}).")

    nsp, npres, nwl = len(species), len(p_all), len(wl)
    qc_grid = np.zeros((nsp, npres))
    kext = np.zeros((nsp, npres, nwl))
    ksca = np.zeros((nsp, npres, nwl))
    kscag = np.zeros((nsp, npres, nwl))
    z = np.linspace(-_LOGNORMAL_Z_MAX, _LOGNORMAL_Z_MAX, _LOGNORMAL_N_NODES)
    w = np.exp(-z ** 2) * (z[1] - z[0]) / np.sqrt(np.pi)
    w[[0, -1]] *= 0.5

    t0 = time.perf_counter()
    for si, s in enumerate(species):
        e = spec_cfg[s]
        rho = getattr(e, "rho", None)
        if rho is None:
            raise ValueError(f"opac.cloud entry for '{s}' needs 'rho' (bulk density, g cm^-3).")
        rho = float(rho)
        sel = species_col == s
        ps, qs, rs, sg = p[sel], q_c[sel], reff[sel], sigma[sel]
        if len(np.unique(ps)) != len(ps):
            raise ValueError(f"{path}: duplicate pressure rows for cloud species '{s}'.")
        o = np.argsort(ps)
        ps, qs, rs, sg = ps[o], qs[o], rs[o], sg[o]
        ln_s = np.log(sg)
        # Effective radius -> lognormal median radius
        r_g_um = rs * np.exp(-2.5 * ln_s ** 2) if dist == "lognormal" else rs
        r_g = r_g_um * 1e-4
        if dist == "lognormal":
            r_nodes = r_g[:, None] * np.exp(np.sqrt(2.0) * ln_s[:, None] * z[None, :])   # (m, nz)
        else:
            r_nodes = r_g[:, None]

        f18 = None
        n = k = None
        if method == "f18":
            f18 = {key: float(getattr(e, key, None) if getattr(e, key, None) is not None
                              else getattr(f18_default, key, {"Q0": 20.0, "Q1": 1.0, "a": 4.0}[key])
                              if f18_default is not None else {"Q0": 20.0, "Q1": 1.0, "a": 4.0}[key])
                   for key in ("Q0", "Q1", "a")}
        else:
            nk_path = getattr(e, "path", None)
            if nk_path is None:
                raise ValueError(f"opac.cloud entry for '{s}' needs 'path' (n,k file) for cloud_method: {method}.")
            n, k = _nk_on_grid(_resolve(nk_path, _STATE.get("exp_dir")), wl)

        if dist == "mono":
            # Single size: efficiencies at each layer's radius exactly (no table).
            r_u, inv = np.unique(r_g, return_inverse=True)
            Qe, Qs, g = _efficiency_table(method, wl, r_u, n, k, f18)
            area = np.pi * r_u ** 2
            s_ext, s_sca, s_scag = area[:, None] * Qe, area[:, None] * Qs, area[:, None] * Qs * g
            n_per_g = 3.0 / (4.0 * np.pi * rho * r_u ** 3)
            per = (n_per_g[:, None] * s_ext, n_per_g[:, None] * s_sca, n_per_g[:, None] * s_scag)
            layer_props = [(per[0][inv[m]], per[1][inv[m]], per[2][inv[m]]) for m in range(len(ps))]
        else:
            # Lognormal: efficiencies on one log-radius table spanning every layer's
            # distribution (|z| <= 5), then per layer the size integral as a quadrature
            # over the table radii themselves. The table is much denser than the 41
            # transformed-trapezoid nodes of opacity_cloud._lognormal_cloud, which
            # resolves the Mie ripples at large size parameter. Distributions too
            # narrow to span 41 table points use those 41 nodes, interpolating the
            # table in log Q vs log r.
            lr = np.log(r_nodes.ravel())
            lr_tab = np.linspace(lr.min() - 1e-6, lr.max() + 1e-6, max(nr, 2))
            dlr = lr_tab[1] - lr_tab[0]
            r_tab = np.exp(lr_tab)
            Qe, Qs, g = _efficiency_table(method, wl, r_tab, n, k, f18)
            layer_props = []
            for m in range(len(ps)):
                if ln_s[m] > 0:
                    zt = (lr_tab - np.log(r_g[m])) / (np.sqrt(2.0) * ln_s[m])
                    on = np.abs(zt) <= _LOGNORMAL_Z_MAX
                else:
                    on = np.zeros(len(lr_tab), bool)
                if on.sum() >= _LOGNORMAL_N_NODES:
                    wt = np.exp(-zt[on] ** 2) / np.sqrt(np.pi) * dlr / (np.sqrt(2.0) * ln_s[m])
                    wt[[0, -1]] *= 0.5
                    dA = (wt * np.pi * r_tab[on] ** 2)[:, None]
                    Qe_m, Qs_m, g_m = Qe[on], Qs[on], g[on]
                else:
                    rn = r_nodes[m]
                    Qe_m = _interp_logr(Qe, lr_tab, np.log(rn), True)
                    Qs_m = _interp_logr(Qs, lr_tab, np.log(rn), True)
                    g_m = _interp_logr(g, lr_tab, np.log(rn), False)
                    dA = (w * np.pi * rn ** 2)[:, None]
                s_ext, s_sca, s_scag = (dA * Qe_m).sum(0), (dA * Qs_m).sum(0), (dA * Qs_m * g_m).sum(0)
                n_per_g = 3.0 / (4.0 * np.pi * rho * r_g[m] ** 3) * np.exp(-4.5 * ln_s[m] ** 2)
                layer_props.append((n_per_g * s_ext, n_per_g * s_sca, n_per_g * s_scag))

        for m in range(len(ps)):
            j = np.searchsorted(p_all, ps[m])
            qc_grid[si, j] = qs[m]
            kext[si, j], ksca[si, j], kscag[si, j] = layer_props[m]

        # Pressures without a row for this species: q_c = 0 there; hold the nearest
        # optical properties so interpolation between layers is set by q_c alone.
        have = np.isin(p_all, ps)
        idx = np.where(have)[0]
        for j in np.where(~have)[0]:
            near = idx[np.argmin(np.abs(idx - j))]
            kext[si, j], ksca[si, j], kscag[si, j] = kext[si, near], ksca[si, near], kscag[si, near]

    _STATE["cloud"] = {
        "logp": np.log10(p_all), "species": species, "q_c": qc_grid,
        "kext": kext, "ksca": ksca, "kscag": kscag, "path": str(path),
        "method": method, "dist": dist,
    }
    print(f"[profile] clouds from {path.name}: species {species}, {npres} pressures, method {method}, "
          f"{dist}; optical properties computed in {time.perf_counter() - t0:.1f} s")


def init_profiles(cfg, exp_dir=None) -> None:
    """Setup loader for the file kernels: reads the files named in ``cfg.profile``.

    Reloads only when the files or cloud settings change. When a T-p file is
    used, ``cfg.physics.nlay`` is set to its number of layers (other setup
    code sizes caches from it).
    """
    pcfg = getattr(cfg, "profile", None)
    if pcfg is None:
        raise ValueError("The 'file' profile kernels need a 'profile:' section in the config.")
    # Callers without an exp_dir (e.g. build_forward_model from the forward runner)
    # reuse the one from the first setup call, so relative paths resolve the same.
    if exp_dir is None:
        exp_dir = _STATE.get("exp_dir")
    _STATE["exp_dir"] = exp_dir
    phys = cfg.physics

    def want(kind, name="file"):
        return str(getattr(phys, kind, "none")).strip().lower() == name

    tp_file = getattr(pcfg, "tp_file", None)
    if want("vert_Tp"):
        if tp_file is None:
            raise ValueError("vert_Tp: file needs profile.tp_file.")
        path = _resolve(tp_file, exp_dir)
        if _STATE.get("tp", {}).get("path") != str(path):
            _load_tp(path)
        nlay_file = len(_STATE["tp"]["p_lay"])
        nlay_cfg = getattr(phys, "nlay", None)
        if nlay_cfg is not None and int(nlay_cfg) != nlay_file:
            print(f"[profile] physics.nlay={nlay_cfg} replaced by the T-p file's {nlay_file} layers")
        phys.nlay = nlay_file
        if want("vert_mu") and _STATE["tp"]["mu_lay"] is None:
            raise ValueError(f"vert_mu: file needs a mu_amu column in {path.name}.")
    else:
        if want("vert_mu"):
            raise ValueError("vert_mu: file needs vert_Tp: file (mu is read from the T-p-mu-Kzz file).")
        _STATE.pop("tp", None)

    if want("vert_chem"):
        vmr_file = getattr(pcfg, "vmr_file", None)
        if vmr_file is None:
            raise ValueError("vert_chem: file needs profile.vmr_file.")
        path = _resolve(vmr_file, exp_dir)
        if _STATE.get("vmr", {}).get("path") != str(path):
            _load_vmr(path)
    else:
        _STATE.pop("vmr", None)

    if want("opac_cloud") or want("vert_cloud"):
        cloud_file = getattr(pcfg, "cloud_file", None)
        if cloud_file is None:
            raise ValueError("vert_cloud/opac_cloud: file need profile.cloud_file.")
        from .build_opacities import master_wavelength_cut
        wl = np.asarray(master_wavelength_cut(), dtype=np.float64)
        path = _resolve(cloud_file, exp_dir)
        key = (str(path), wl.size, float(wl[0]), float(wl[-1]),
               str(getattr(pcfg, "cloud_method", "lxmie")), str(getattr(pcfg, "cloud_dist", "lognormal")),
               float(getattr(pcfg, "cloud_sigma", 2.0)), int(getattr(pcfg, "cloud_nr", 400)),
               repr(getattr(pcfg, "f18", None)), repr(getattr(getattr(cfg, "opac", None), "cloud", None)))
        if _STATE.get("cloud_key") != key:
            _load_cloud(path, cfg, wl)
            _STATE["cloud_key"] = key
    else:
        _STATE.pop("cloud", None)
        _STATE.pop("cloud_key", None)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------
def has_tp_profile() -> bool:
    return "tp" in _STATE


def has_kzz_profile() -> bool:
    return "tp" in _STATE and _STATE["tp"]["Kzz_lay"] is not None


def profile_grid():
    """(p_lev, p_lay) in dyne cm^-2, bottom first, from the T-p file."""
    tp = _STATE["tp"]
    return jnp.asarray(tp["p_lev"]), jnp.asarray(tp["p_lay"])


def profile_tp():
    tp = _STATE["tp"]
    return jnp.asarray(tp["T_lev"]), jnp.asarray(tp["T_lay"])


def profile_kzz():
    return jnp.asarray(_STATE["tp"]["Kzz_lay"])


def profile_mu():
    return jnp.asarray(_STATE["tp"]["mu_lay"])


def vmr_species() -> tuple:
    v = _STATE["vmr"]
    return tuple(v["vmr"]) + (("H2", "He") if v["fill"] else ())


def cloud_species() -> tuple:
    return tuple(_STATE["cloud"]["species"]) if "cloud" in _STATE else ()


def runtime_cache() -> dict:
    """Cloud optical properties per gram of condensate, (nspec, npres, nwl), for the opac cache."""
    c = _STATE.get("cloud")
    if c is None:
        return {}
    return {
        "profile_cloud_kext": jnp.asarray(c["kext"]),
        "profile_cloud_ksca": jnp.asarray(c["ksca"]),
        "profile_cloud_kscag": jnp.asarray(c["kscag"]),
    }


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def _interp_log(logp_lay, logp_tab, values):
    """Interpolate (n,) values tabulated at logp_tab (increasing) in log10 value; zeros stay zero."""
    lv = jnp.log10(jnp.maximum(jnp.asarray(values), _LOG_FLOOR))
    out = 10.0 ** jnp.interp(logp_lay, jnp.asarray(logp_tab), lv)
    return jnp.where(out > 2.0 * _LOG_FLOOR, out, 0.0)


def file_tp_kernel(p_lev, params):
    """``vert_Tp: file``: temperatures from the T-p file (on its own layers)."""
    del p_lev, params
    return profile_tp()


def file_mu_kernel(params, vmr_lay, nlay):
    """``vert_mu: file``: mean molecular weight from the T-p-mu-Kzz file."""
    del params, vmr_lay, nlay
    return profile_mu()


def file_chem_kernel(p_lay, T_lay, params, nlay):
    """``vert_chem: file``: VMRs from the VMR file, interpolated in log p and log VMR."""
    del T_lay, params, nlay
    from .vert_chem import solar_He

    v = _STATE["vmr"]
    logp = jnp.log10(p_lay / bar)
    out = {s: _interp_log(logp, v["logp"], arr) for s, arr in v["vmr"].items()}
    if v["fill"]:
        # H2 + He fill the remainder at the solar He/H2 ratio (as constant_vmr with no atomic H)
        background = 1.0 - sum(out.values())
        background = jnp.where(background > 0.0, background, jnp.nan)
        N_H = background / (solar_He + 0.5)
        out["H2"] = N_H / 2.0
        out["He"] = solar_He * N_H
    return out


def _cloud_on_layers(p_lay):
    """q_c per species on the layers, (nspec, nlay).

    Unlike T and VMRs, clouds are not extended beyond the cloud file's pressure
    range: layers above its lowest or below its highest pressure get no condensate.
    """
    c = _STATE["cloud"]
    logp = jnp.log10(p_lay / bar)
    lo, hi = float(c["logp"][0]), float(c["logp"][-1])
    inside = (logp >= lo - 1e-9) & (logp <= hi + 1e-9)
    q = jnp.stack([_interp_log(logp, c["logp"], q) for q in c["q_c"]], axis=0)
    return jnp.where(inside[None, :], q, 0.0)


def file_cloud_profile_kernel(p_lay, T_lay, mu_lay, rho_lay, nd_lay, params):
    """``vert_cloud: file``: total condensate q_c per layer (sum over species)."""
    del T_lay, mu_lay, rho_lay, nd_lay, params
    return jnp.sum(_cloud_on_layers(p_lay), axis=0)


def file_cloud_opacity_kernel(state, params):
    """``opac_cloud: file``: (k_ext, ssa, g) from the precomputed per-species optical properties."""
    del params
    c = _STATE["cloud"]
    p_lay = state["p_lay"]
    q = _cloud_on_layers(p_lay)                                    # (nspec, nlay)
    logp = jnp.log10(p_lay / bar)
    logp_tab = jnp.asarray(c["logp"])
    n = logp_tab.shape[0]
    if n == 1:
        i = jnp.zeros_like(logp, dtype=jnp.int32)
        t = jnp.zeros_like(logp)
        nxt = i
    else:
        i = jnp.clip(jnp.searchsorted(logp_tab, logp) - 1, 0, n - 2)
        t = jnp.clip((logp - logp_tab[i]) / (logp_tab[i + 1] - logp_tab[i]), 0.0, 1.0)
        nxt = i + 1

    def on_layers(tab):                                            # (nspec, npres, nwl) -> (nspec, nlay, nwl)
        return (1.0 - t)[None, :, None] * tab[:, i, :] + t[None, :, None] * tab[:, nxt, :]

    k_ext = jnp.sum(q[:, :, None] * on_layers(state["profile_cloud_kext"]), axis=0)
    k_sca = jnp.sum(q[:, :, None] * on_layers(state["profile_cloud_ksca"]), axis=0)
    k_scag = jnp.sum(q[:, :, None] * on_layers(state["profile_cloud_kscag"]), axis=0)
    pos = k_ext > 0.0
    ssa = jnp.where(pos, k_sca / jnp.where(pos, k_ext, 1.0), 0.0)
    g = jnp.where(k_sca > 0.0, k_scag / jnp.where(k_sca > 0.0, k_sca, 1.0), 0.0)
    return k_ext, ssa, g
