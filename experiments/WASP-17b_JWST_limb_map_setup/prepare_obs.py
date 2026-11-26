"""Write the WASP-17b limb-angle spectra (Alderson et al. 2026) in Exo Skryer's observation format.

Source: zenodo_21245265/w17_transmission_theta<angle>.txt (Zenodo 10.5281/zenodo.21245265, CC-BY 4.0).
Columns there: wl, wl start, wl end [um], planet radius Rp(theta)/Rs, its 1-sigma error. They are the
Harmonica "transmission strings" radii at limb angle theta (0 = morning / west, 180 = evening / east),
24 bins over 0.59-12 um (NIRISS SOSS, NIRSpec G395H, MIRI LRS).

Each angle is retrieved as its own transit_1d planet (as in the paper), so the observable is the
full-circle-equivalent depth D = (Rp/Rs)^2 with sigma_D = 2 (Rp/Rs) sigma_R.
Output: obs/W17b_limb_theta<NNN>.txt, columns: wavelength_um half_width_um depth depth_err response_mode.

    python prepare_obs.py
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "zenodo_21245265"
OUT = HERE / "obs"


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(SRC.glob("w17_transmission_theta*.txt"),
                   key=lambda p: float(p.stem.split("theta")[1]))
    for f in files:
        theta = float(f.stem.split("theta")[1])
        d = np.loadtxt(f, skiprows=1)              # header line has no '#'
        wl, lo, hi, r, sr = d.T
        depth, err = r**2, 2.0 * r * sr
        out = OUT / f"W17b_limb_theta{int(round(theta)):03d}.txt"
        with open(out, "w") as fh:
            fh.write(f"# WASP-17b transmission spectrum at limb angle theta = {theta:g} deg "
                     "(0 = morning/west, 180 = evening/east)\n")
            fh.write(f"# Source: {SRC.name}/{f.name} (Alderson et al. 2026, Harmonica transmission strings)\n")
            fh.write("# depth = (Rp/Rs)^2 (full-circle equivalent), depth_err = 2 (Rp/Rs) sigma_R\n")
            fh.write("# wavelength_um half_width_um depth depth_err response_mode\n")
            for row in zip(wl, 0.5 * (hi - lo), depth, err):
                fh.write("{:.6f} {:.6f} {:.8e} {:.8e} boxcar\n".format(*row))
        print(f"theta {theta:5.1f}: {len(wl)} bins, depth {depth.min():.5f}-{depth.max():.5f} -> {out.name}")


if __name__ == "__main__":
    main()
