"""Write the HD 209458b limb spectra in the transit_1_5d observation convention.

Inputs (JWST NIRCam, Eureka!/harmonica, circular orbit), as provided:
    HD209_JWST_NIRCam_Eureka_circular_evening_east.txt
    HD209_JWST_NIRCam_Eureka_circular_morning_west.txt
Columns: wavelength_um, half-width_um, transit_depth, uncertainty, response_mode, offset_group.

The provided limb depths are (R_limb / R_s)^2, i.e. the depth if that limb filled the whole
circle (~0.0146 each, the full HD 209458b depth). transit_1_5d compares each limb's data with
the half-annulus depth 0.5 * (R_limb / R_s)^2, so depths and uncertainties are multiplied by
0.5 (exact for a half circle). Everything else is copied unchanged.

    python prepare_limb_obs.py
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FILES = {
    "evening_east": "HD209_JWST_NIRCam_Eureka_circular_evening_east.txt",
    "morning_west": "HD209_JWST_NIRCam_Eureka_circular_morning_west.txt",
}

for limb, name in FILES.items():
    raw = np.genfromtxt(HERE / name, dtype=str, comments="#")
    wl, hw, depth, err = (raw[:, i].astype(float) for i in range(4))
    mode, group = raw[:, 4], raw[:, 5]
    out = HERE / f"HD209b_NIRCam_{limb}_half.txt"
    with open(out, "w") as f:
        f.write(f"# HD 209458b {limb.replace('_', ' / ')} limb, JWST NIRCam (Eureka!, circular orbit)\n")
        f.write(f"# Source: {name} (prepare_limb_obs.py)\n")
        f.write("# Depth and uncertainty x 0.5: source gives (R_limb/R_s)^2; transit_1_5d uses half-annulus depths.\n")
        f.write("# wavelength_um half_width_um transit_depth uncertainty response_mode offset_group\n")
        for row in zip(wl, hw, 0.5 * depth, 0.5 * err, mode, group):
            f.write("{:.10f} {:.10f} {:.10e} {:.10e} {} {}\n".format(*row))
    print(f"wrote {out.name} ({wl.size} points, {wl.min():.3f}-{wl.max():.3f} um)")
