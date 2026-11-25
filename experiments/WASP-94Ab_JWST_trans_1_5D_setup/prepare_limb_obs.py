"""Write the WASP-94Ab limb spectra (JWST NIRISS SOSS) in the transit_1_5d observation convention.

Three independent reductions of the same visit, all on the same 59-bin grid (0.87-2.80 um):

- Eureka_1.txt / Eureka_2.txt (ECSV): limb 1 = evening, limb 2 = morning. Checked: Eureka_1 correlates
  with Fu's labelled evening spectrum (r = 0.95). bin_width is a half-width (~half the bin spacing).
- FIREFLy.csv: rp_rs2 = MORNING and rp_rs2_2 = EVENING (the opposite order). Checked by correlation with
  Fu: rp_rs2 vs morning r = +0.87, rp_rs2_2 vs evening r = +0.96. wvl_width_um is the full bin width.
  Rows are in decreasing wavelength.
- Fu_spectrum.txt: rp1 = evening, rp2 = morning (labelled in the header), with lower/upper errors and no
  bin widths. Its wavelength grid is identical to Eureka's, so Eureka's half-widths are used.
- FIREFLy-spherical1D.csv is the whole-terminator (1D) spectrum. It is not written here.

All limb depths are given as (R_limb / R_s)^2, the depth if that limb filled the whole circle (~0.011,
the full WASP-94Ab depth). transit_1_5d compares each limb's data with the half-annulus depth
0.5 * (R_limb / R_s)^2, so depths and uncertainties are multiplied by 0.5 (exact for a half circle).
Output columns: wavelength_um half_width_um depth err_plus err_minus response_mode offset_group
(east = evening, west = morning). The offset group is the limb name, so a morning-only offset
(Mukherjee et al. 2025, +-250 ppm on (R/R_s)^2) can be fitted as offset_morning_joint.

    python prepare_limb_obs.py
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def write(name, limb, wl, hw, depth, err_plus, err_minus, source):
    o = np.argsort(wl)
    out = HERE / f"WASP-94Ab_SOSS_{name}_{limb}_half.txt"
    with open(out, "w") as f:
        f.write(f"# WASP-94Ab {limb.replace('_', ' / ')} limb, JWST NIRISS SOSS, {name} reduction\n")
        f.write(f"# Source: {source} (prepare_limb_obs.py)\n")
        f.write("# Depth and errors x 0.5: source gives (R_limb/R_s)^2; transit_1_5d uses half-annulus depths.\n")
        f.write("# wavelength_um half_width_um transit_depth err_plus err_minus response_mode offset_group\n")
        group = limb.split("_")[0]
        for row in zip(wl[o], hw[o], 0.5 * depth[o], 0.5 * err_plus[o], 0.5 * err_minus[o]):
            f.write("{:.10f} {:.10f} {:.10e} {:.10e} {:.10e} boxcar {}\n".format(*row, group))
    print(f"wrote {out.name} ({wl.size} points)")


# Eureka (ECSV: wavelength, bin_width, value, errorneg, errorpos)
e1 = np.loadtxt(HERE / "Eureka_1.txt", comments="#", skiprows=10)
e2 = np.loadtxt(HERE / "Eureka_2.txt", comments="#", skiprows=10)
write("Eureka", "evening_east", e1[:, 0], e1[:, 1], e1[:, 2], e1[:, 4], e1[:, 3], "Eureka_1.txt")
write("Eureka", "morning_west", e2[:, 0], e2[:, 1], e2[:, 2], e2[:, 4], e2[:, 3], "Eureka_2.txt")

# FIREFLy (rp_rs2 = morning, rp_rs2_2 = evening; full bin widths; symmetric errors)
ff = np.genfromtxt(HERE / "FIREFLy.csv", delimiter=",", names=True)
hw = 0.5 * ff["wvl_width_um"]
write("FIREFLy", "evening_east", ff["wvl_cen_um"], hw, ff["rp_rs2_2"], ff["rp_rs2_2_unc"], ff["rp_rs2_2_unc"],
      "FIREFLy.csv (rp_rs2_2)")
write("FIREFLy", "morning_west", ff["wvl_cen_um"], hw, ff["rp_rs2"], ff["rp_rs2_unc"], ff["rp_rs2_unc"],
      "FIREFLy.csv (rp_rs2)")

# Fu (wavelength, evening depth, lower, upper, morning depth, lower, upper); half-widths from Eureka
fu = np.loadtxt(HERE / "Fu_spectrum.txt", comments="#")
assert np.allclose(fu[:, 0], e1[:, 0], rtol=1e-6), "Fu and Eureka wavelength grids differ"
write("Fu", "evening_east", fu[:, 0], e1[:, 1], fu[:, 1], fu[:, 3], fu[:, 2], "Fu_spectrum.txt (rp1, Evening)")
write("Fu", "morning_west", fu[:, 0], e1[:, 1], fu[:, 4], fu[:, 6], fu[:, 5], "Fu_spectrum.txt (rp2, morning)")
