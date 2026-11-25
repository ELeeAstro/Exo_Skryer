"""Write the HD 209458b (Barstow 2020 data/setup) retrieval configs for 1D, 1.5D, 2D and 3D.

    python make_configs.py

All four share the Barstow T-p profile, constant VMRs (H2O, Na, K), ck R250 (TRANS) and the
default grey slab cloud (vert_cloud: slab + opac_cloud: grey_profile: log_10_p_top_slab,
log_10_dp_slab [bar], log_10_k_cld_grey [cm^2 g^-1]). The cloud mimics an Iceberg cloud:
  1D        one slab everywhere
  1_5D      a slab per limb (east = evening, west = morning)
  2D        a slab per side (day / night)
  3D_iceberg  one slab, masked in angle by physics.iceberg_cloud (f_cld, phi_0_cld, theta_0_cld)
The data are the single whole-terminator spectrum (Sing et al. 2016), so the multi-D fits are
tests of the machinery, not of limb-resolved data. Sampler: dynesty, 100 live points, dlogz 1.
"""
from pathlib import Path

import argparse
import math

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description="Write the 1D / 1.5D / 2D / 3D_iceberg configs.")
ap.add_argument("--sampler", choices=("dynesty", "nautilus"), default="dynesty")
ap.add_argument("--out", default=".", help="subdirectory of this one to write <geom>/ into (default: here)")
ap.add_argument("--n-live", type=int, default=250, help="nautilus live points (default 250)")
ap.add_argument("--n-eff", type=int, default=1000, help="nautilus effective sample size (default 1000)")
ap.add_argument("--opac", choices=("ck250", "os20k"), default="ck250",
                help="opacities: ck250 (correlated-k R250, ck_mix TRANS) or os20k (opacity sampling R20000)")
ap.add_argument("--dlogz", type=float, default=1.0,
                help="dynesty-equivalent stopping dlogz for nautilus: f_live = 1 - exp(-dlogz) (default 1)")
args = ap.parse_args()
OUT = (HERE / args.out).resolve()
UP = "../" * (len(OUT.relative_to(HERE).parts) + 3)    # from <out>/<geom>/ back to the repo root
OBS = f"{UP}obs_data/HD209458b_transmission_Sing2016.txt"
CK = f"{UP}opac_data/ck"
OS = f"{UP}opac_data/os"
CIA = f"{UP}opac_data/cia"

PHYSICS = """physics:
  nlay: 99
  vert_Tp: Barstow
  vert_alt: p_ref
  vert_chem: constant_vmr_clr
  vert_mu: dynamic
  vert_cloud: slab
  opac_line: {kind}
  opac_ray: {kind}
  opac_cia: {kind}
  opac_cloud: grey_profile
  opac_special: None
  rt_scheme: {rt}
  contri_func: False
  refraction: None
{extra}"""

OPAC_CK = f"""opac:
  wl_master: {CK}/wl_ck_R250.txt
  full_grid: False
  ck: True
  ck_mix: TRANS
  line:
    - {{species: H2O, path: {CK}/H2O_ck_R250.zarr.zip}}
    - {{species: Na,  path: {CK}/Na_ck_R250.zarr.zip}}
    - {{species: K,   path: {CK}/K_ck_R250.zarr.zip}}
  ray:
    - {{species: H2}}
    - {{species: He}}
  cia:
    - {{species: H2-H2, path: {CIA}/H2-H2_2011.zarr.zip}}
    - {{species: H2-He, path: {CIA}/H2-He_2011.zarr.zip}}
  special: None
  cloud: None
"""

OPAC_OS = f"""opac:
  wl_master: {OS}/wl_R20000.txt
  full_grid: False
  ck: False
  line:
    - {{species: H2O, path: {OS}/H2O_R20000.zarr.zip}}
    - {{species: Na,  path: {OS}/Na_R20000.zarr.zip}}
    - {{species: K,   path: {OS}/K_R20000.zarr.zip}}
  ray:
    - {{species: H2}}
    - {{species: He}}
  cia:
    - {{species: H2-H2, path: {CIA}/H2-H2_2011.zarr.zip}}
    - {{species: H2-He, path: {CIA}/H2-He_2011.zarr.zip}}
  special: None
  cloud: None
"""
OPAC = OPAC_OS if args.opac == "os20k" else OPAC_CK


RUNTIME = """
runtime:
  platform: cpu
  cuda_visible_devices: 0
  threads: 12
"""

# Nautilus matched to a dynesty dlogz (--dlogz; n_live from --n-live, live points per bound); dynesty stops at
# ln(1 + Z_live / Z) < dlogz, i.e. live points holding < 1 - exp(-dlogz) of the evidence (0.632 for dlogz 1,
# 0.095 for 0.1), which is f_live;
# n_eff 1000 ~ the ESS dynesty gives with 250 live points (~430-460 at 100).
NAUTILUS = """sampling:
  engine: nautilus
  nautilus:
    n_live: {n_live}
    f_live: {f_live:.4f}
    n_shell: 1
    n_eff: {n_eff}
    discard_exploration: false
    equal_weight: true
    verbose: true
    seed: 42
""" + RUNTIME

SAMPLING = """sampling:
  engine: dynesty
  dynesty:
    nlive: 100
    bound: multi
    sample: auto
    dlogz: 1.0
    maxiter: null
    maxcall: null
    bootstrap: 0
    enlarge: null
    update_interval: null
    dynamic: false
    print_progress: true
    seed: 42

runtime:
  platform: cpu
  cuda_visible_devices: 0
  threads: 12
"""

# name: (low, high, init)
FIXED = {"R_s": 1.18655, "p_bot": 100.0, "p_top": 1e-8, "log_10_p_ref": 1.0}
PRIORS = {
    "log_10_g": (2.7, 3.3, 2.97), "R_p": (1.1, 1.5, 1.35), "T_strat": (100.0, 3000.0, 1200.0),
    "log_10_f_H2O": (-12, -2, -5), "log_10_f_Na": (-12, -2, -6), "log_10_f_K": (-12, -2, -7),
    "log_10_p_top_slab": (-6, 2, -2), "log_10_dp_slab": (-4, 2, 0), "log_10_k_cld_grey": (-8, 2, -2),
    "f_cld": (0.0, 1.0, 0.5), "phi_0_cld": (-180.0, 180.0, 0.0), "theta_0_cld": (-90.0, 90.0, 0.0),
}
GAS = ("log_10_g", "R_p", "log_10_f_H2O", "log_10_f_Na", "log_10_f_K")

# geometry: (rt_scheme, extra physics, fixed tag, {param: tags}, extra fixed params)
SETUPS = {
    "1D": ("transit_1d", "", "", {**{k: ("",) for k in GAS}, "T_strat": ("",),
                                  "log_10_p_top_slab": ("",), "log_10_dp_slab": ("",), "log_10_k_cld_grey": ("",)}, {}),
    "1_5D": ("transit_1_5d", "", "_joint", {**{k: ("_joint",) for k in GAS}, "log_10_dp_slab": ("_joint",),
             "T_strat": ("_east", "_west"), "log_10_p_top_slab": ("_east", "_west"),
             "log_10_k_cld_grey": ("_east", "_west")}, {}),
    "2D": ("transit_2d", "  n_zenith: 4\n", "_joint", {**{k: ("_joint",) for k in GAS}, "log_10_dp_slab": ("_joint",),
           "T_strat": ("_day", "_night"), "log_10_p_top_slab": ("_day", "_night"),
           "log_10_k_cld_grey": ("_day", "_night")}, {"beta_dn": 20.0}),
    "3D_iceberg": ("transit_3d", "  n_zenith: 4\n  iceberg_cloud: True\n", "_joint",
                   {**{k: ("_joint",) for k in GAS},
                    "T_strat": ("_east_day", "_west_day", "_east_night", "_west_night"),
                    **{k: ("_joint",) for k in ("log_10_p_top_slab", "log_10_dp_slab", "log_10_k_cld_grey",
                                                 "f_cld", "phi_0_cld", "theta_0_cld")}}, {"beta_dn": 20.0}),
}


def line_delta(name, v):
    return f"  - {{ name: {name}, dist: delta, value: {v}, transform: identity, init: {v} }}"


def line_free(name, lo, hi, init):
    return f"  - {{ name: {name}, dist: uniform, low: {lo}, high: {hi}, transform: logit, init: {init} }}"


for geom, (rt, extra, ftag, free, xfixed) in SETUPS.items():
    d = OUT / geom
    d.mkdir(parents=True, exist_ok=True)
    lines = [line_delta(k + ftag, v) for k, v in {**FIXED, **xfixed}.items()]
    lines += [line_free(k + t, *PRIORS[k]) for k, tags in free.items() for t in tags]
    n_free = sum(len(t) for t in free.values())
    text = (f"# HD 209458b, Barstow et al. (2020) setup, {geom} geometry ({n_free} free parameters).\n"
            f"# Generated by make_configs.py --sampler {args.sampler} --out {args.out}: edit that script, "
            f"not this file.\n\n"
            f"data:\n  obs: {OBS}\n  obs_east: None\n  obs_west: None\n  stellar: None\n\n"
            + PHYSICS.format(rt=rt, extra=extra, kind="os" if args.opac == "os20k" else "ck") + "\n" + OPAC + "\nparams:\n" + "\n".join(lines) + "\n\n"
            + (NAUTILUS.format(n_live=args.n_live, n_eff=args.n_eff, f_live=1.0 - math.exp(-args.dlogz)) if args.sampler == "nautilus" else SAMPLING))
    (d / "retrieval_config.yaml").write_text(text)
    print(f"{d.relative_to(HERE)}: {n_free} free parameters ({args.sampler})")
