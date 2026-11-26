"""Write the limb-survey (1.5D morning/evening) retrieval setups: one directory per planet, one
config per data set (data sets are retrieved separately, since some are mutually inconsistent).

    python make_limb_configs.py

For each setup it writes <planet>/retrieval_<setup>.yaml and the limb spectra it uses
(<planet>/<setup>_{evening_east,morning_west}.txt, half-annulus depths, with offset-group labels
where an offset is fitted). Edit this script, not the generated files.

Common model (as the HD 209458b multi-D paper runs): Barstow T-p, constant VMRs (centred log-ratio),
slab + grey cloud, os R20000, rt_scheme transit_1_5d.
- Shared between limbs: log g, R_p, chemistry, slab thickness.
- Per limb: T_strat, cloud-top pressure, grey cloud opacity.
- Species chosen per data set from its wavelength coverage.
Sampler: nautilus, n_live 1000, f_live 0.0952 (= dynesty dlogz 0.1), n_eff 10000.

Offsets [ppm, added to the data]:
- Fu et al. 2025 SOSS fits fixed t_c to the white-light value, which leaves the absolute
  evening/morning split degenerate with timing, so the morning limb gets a free offset of
  +-2000 ppm. Only the spectral shapes then constrain the limbs.
- Mukherjee et al. 2025 WASP-94Ab quote +-250 ppm on (R/Rs)^2, i.e. +-125 ppm on the half-annulus
  depth, for the morning limb.
- Maguire et al. 2026 HD 209458b: F444W (separate visit) gets +-100 ppm relative to F322W2.
- Espinoza et al. 2024 (PRISM) and Murphy et al. 2024 (NIRCam, t_c pinned to 0.7 s): no offsets.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OBS = ROOT / "obs_data"
EXP = ROOT / "experiments"
UP = "../../../"        # <planet>/ -> repo root

SPECIES_SETS = {
    "PRISM": ("H2O", "CO2", "CO", "CH4", "SO2", "H2S", "Na", "K"),      # 0.5-5.5 um
    "SOSS": ("H2O", "CO2", "CO", "CH4", "H2S"),                         # 0.87-2.8 um
    "SOSS_cool": ("H2O", "CO2", "CO", "CH4", "H2S", "NH3"),             # cool planets
    "F322W2": ("H2O", "CO2", "CH4", "NH3", "H2S"),                      # 2.45-3.95 um
    "F322W2_F444W": ("H2O", "CO2", "CO", "CH4", "H2S"),                 # 2.4-5.0 um
}

# Planet: R_s [R_sun], log g prior, R_p prior [R_J], T_strat prior [K]
PLANETS = {
    "WASP-39b": dict(R_s=0.931, logg=(2.3, 3.0, 2.63), R_p=(1.00, 1.50, 1.27), T=(300.0, 1800.0, 1100.0)),
    "WASP-107b": dict(R_s=0.67, logg=(2.1, 2.8, 2.45), R_p=(0.80, 1.10, 0.94), T=(300.0, 1500.0, 750.0)),
    "WASP-94Ab": dict(R_s=1.62, logg=(2.4, 2.9, 2.66), R_p=(1.42, 1.72, 1.57), T=(500.0, 2500.0, 1500.0)),
    "WASP-17b": dict(R_s=1.583, logg=(2.2, 2.8, 2.5), R_p=(1.59, 2.15, 1.87), T=(500.0, 2500.0, 1700.0)),
    "HD209458b": dict(R_s=1.2, logg=(2.7, 3.2, 2.97), R_p=(1.10, 1.70, 1.39), T=(300.0, 2500.0, 1400.0)),
}

W94 = EXP / "WASP-94Ab_JWST_trans_1_5D_setup"
H209 = EXP / "HD209_JWST_trans_1_5D_setup"

# setup: (planet, species set, evening file, morning file, offsets {group: (low, high)}, note)
# A group set to None is a reference: an offset fixed to 0.
SETUPS = {
    ("WASP-39b", "PRISM_Espinoza2024"): (
        "PRISM", OBS / "WASP-39b_JWST_evening_east.txt", OBS / "WASP-39b_JWST_morning_west.txt", {},
        "JWST NIRSpec PRISM, Espinoza et al. 2024 (Nature), R=100, catwoman"),
    ("WASP-39b", "SOSS_Fu2025"): (
        "SOSS", OBS / "WASP-39b_NIRISS_SOSS_Fu2025_evening_east.txt", OBS / "WASP-39b_NIRISS_SOSS_Fu2025_morning_west.txt",
        {"evening": None, "morning": (-2000.0, 2000.0)}, "JWST NIRISS SOSS, Fu et al. 2025 (ApJL)"),
    ("WASP-107b", "NIRCam_Murphy2024"): (
        "F322W2", OBS / "WASP-107b_NIRCam_F322W2_Murphy2024_evening_east.txt",
        OBS / "WASP-107b_NIRCam_F322W2_Murphy2024_morning_west.txt", {},
        "JWST NIRCam F322W2, Murphy et al. 2024 (Nat. Astron.), Extended Data Table 2"),
    ("WASP-107b", "SOSS_Fu2025"): (
        "SOSS_cool", OBS / "WASP-107b_NIRISS_SOSS_Fu2025_evening_east.txt", OBS / "WASP-107b_NIRISS_SOSS_Fu2025_morning_west.txt",
        {"evening": None, "morning": (-2000.0, 2000.0)}, "JWST NIRISS SOSS, Fu et al. 2025 (ApJL)"),
    ("WASP-94Ab", "SOSS_Mukherjee2025_FIREFLy"): (
        "SOSS", W94 / "WASP-94Ab_SOSS_FIREFLy_evening_east_half.txt", W94 / "WASP-94Ab_SOSS_FIREFLy_morning_west_half.txt",
        {"evening": None, "morning": (-125.0, 125.0)}, "JWST NIRISS SOSS, Mukherjee et al. 2025, FIREFLy reduction"),
    ("WASP-94Ab", "SOSS_Mukherjee2025_Eureka"): (
        "SOSS", W94 / "WASP-94Ab_SOSS_Eureka_evening_east_half.txt", W94 / "WASP-94Ab_SOSS_Eureka_morning_west_half.txt",
        {"evening": None, "morning": (-125.0, 125.0)}, "JWST NIRISS SOSS, Mukherjee et al. 2025, Eureka! reduction"),
    ("WASP-94Ab", "SOSS_Fu2025"): (
        "SOSS", W94 / "WASP-94Ab_SOSS_Fu_evening_east_half.txt", W94 / "WASP-94Ab_SOSS_Fu_morning_west_half.txt",
        {"evening": None, "morning": (-2000.0, 2000.0)}, "JWST NIRISS SOSS, Fu et al. 2025 (ApJL)"),
    ("WASP-17b", "SOSS_Fu2025"): (
        "SOSS", OBS / "WASP-17b_NIRISS_SOSS_Fu2025_evening_east.txt", OBS / "WASP-17b_NIRISS_SOSS_Fu2025_morning_west.txt",
        {"evening": None, "morning": (-2000.0, 2000.0)}, "JWST NIRISS SOSS, Fu et al. 2025 (ApJL); 2.048 um bin dropped (NaN)"),
    ("HD209458b", "NIRCam_Maguire2026"): (
        "F322W2_F444W", H209 / "HD209b_NIRCam_evening_east_half.txt", H209 / "HD209b_NIRCam_morning_west_half.txt",
        {"F322W2": None, "F444W": (-100.0, 100.0)},
        "JWST NIRCam F322W2 + F444W, Maguire et al. (arXiv:2609.16834), Eureka!/harmonica, circular orbit"),
}


def read_limb(path):
    """(header lines, rows of [wl, hw, depth, err(s)...], response modes, existing groups or None)."""
    head, rows, modes, groups = [], [], [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            head.append(line)
            continue
        tok = line.split()
        nums = []
        for t in tok:
            try:
                nums.append(float(t))
            except ValueError:
                break
        rest = tok[len(nums):]
        rows.append(nums)
        modes.append(rest[0] if rest else "boxcar")
        groups.append(rest[1] if len(rest) > 1 else None)
    return head, rows, modes, (None if all(g is None for g in groups) else groups)


def write_limb(out, src, limb_group, offsets, note):
    """Copy a limb spectrum, setting the offset-group column when this setup fits offsets."""
    head, rows, modes, groups = read_limb(src)
    lines = [f"# Generated by make_limb_configs.py from {src.relative_to(ROOT)}", f"# {note}"] + head
    for r, m, g in zip(rows, modes, groups or [None] * len(rows)):
        nums = " ".join(f"{v:.10g}" for v in r)
        if offsets:
            grp = g if (g is not None and g in offsets) else limb_group
            lines.append(f"{nums} {m} {grp}")
        else:
            lines.append(f"{nums} {m}")
    out.write_text("\n".join(lines) + "\n")
    return len(rows)


def p_delta(name, v):
    return f"  - {{ name: {name}, dist: delta, value: {v}, transform: identity, init: {v} }}"


def p_free(name, lo, hi, init):
    return f"  - {{ name: {name}, dist: uniform, low: {lo}, high: {hi}, transform: logit, init: {init} }}"


VMR_INIT = {"H2O": -3.5, "CO2": -5, "CO": -3.5, "CH4": -6, "SO2": -6, "H2S": -5, "NH3": -6, "Na": -6, "K": -7}

for (planet, setup), (sset, f_even, f_morn, offsets, note) in SETUPS.items():
    d = HERE / planet
    d.mkdir(exist_ok=True)
    pp = PLANETS[planet]
    n_e = write_limb(d / f"{setup}_evening_east.txt", f_even, "evening", offsets, note)
    n_w = write_limb(d / f"{setup}_morning_west.txt", f_morn, "morning", offsets, note)
    species = SPECIES_SETS[sset]

    params = [p_delta("R_s_joint", pp["R_s"]), p_delta("p_bot_joint", 100.0), p_delta("p_top_joint", 1e-8),
              p_delta("log_10_p_ref_joint", 1.0),
              p_free("log_10_g_joint", *pp["logg"]), p_free("R_p_joint", *pp["R_p"])]
    params += [p_free(f"log_10_f_{s}_joint", -12, -1, VMR_INIT[s]) for s in species]
    params += [p_free("log_10_dp_slab_joint", -4, 2, 0)]
    for g, rng in offsets.items():
        params.append(p_delta(f"offset_{g}_joint", 0.0) if rng is None else p_free(f"offset_{g}_joint", rng[0], rng[1], 0.0))
    for side, t0 in (("east", pp["T"][2]), ("west", pp["T"][2] - 100.0)):
        params += [p_free(f"T_strat_{side}", pp["T"][0], pp["T"][1], t0),
                   p_free(f"log_10_p_top_slab_{side}", -6, 2, -2),
                   p_free(f"log_10_k_cld_grey_{side}", -8, 2, -2)]
    n_free = sum(" uniform" in p for p in params)
    lines = "\n".join(f"    - {{species: {s}, path: {UP}opac_data/os/{s}_R20000.zarr.zip}}" for s in species)

    text = f"""# {planet} morning/evening (1.5D) limb retrieval: {note}.
# Setup '{setup}', {n_free} free parameters, {n_e} + {n_w} data points.
# Generated by make_limb_configs.py: edit that script, not this file.
# Offsets [ppm] are added to the data of their group; groups fixed to 0 are the reference.

data:
  obs: None
  obs_east: {setup}_evening_east.txt
  obs_west: {setup}_morning_west.txt
  stellar: None

physics:
  nlay: 99
  vert_Tp: Barstow
  vert_alt: p_ref
  vert_chem: constant_vmr_clr
  vert_mu: dynamic
  vert_cloud: slab
  opac_line: os
  opac_ray: os
  opac_cia: os
  opac_cloud: grey_profile
  opac_special: None
  rt_scheme: transit_1_5d
  contri_func: False
  refraction: None

opac:
  wl_master: {UP}opac_data/os/wl_R20000.txt
  full_grid: False
  ck: False
  line:
{lines}
  ray:
    - {{species: H2}}
    - {{species: He}}
  cia:
    - {{species: H2-H2, path: {UP}opac_data/cia/H2-H2_2011.zarr.zip}}
    - {{species: H2-He, path: {UP}opac_data/cia/H2-He_2011.zarr.zip}}
  special: None
  cloud: None

params:
  # transit_1_5d: every name ends in _joint, _east (evening) or _west (morning).
{chr(10).join(params)}

sampling:
  engine: nautilus
  nautilus:
    n_live: 1000
    f_live: 0.0952        # = dynesty dlogz 0.1 (1 - exp(-0.1))
    n_shell: 1
    n_eff: 10000
    discard_exploration: false
    equal_weight: true
    verbose: true
    seed: 42

runtime:
  platform: cpu
  cuda_visible_devices: 0
  threads: 12
"""
    (d / f"retrieval_{setup}.yaml").write_text(text)
    print(f"{planet}/retrieval_{setup}.yaml: {n_free} free, {n_e}+{n_w} points, species {','.join(species)}"
          + (f", offsets {[g for g, r in offsets.items() if r]}" if offsets else ""))
