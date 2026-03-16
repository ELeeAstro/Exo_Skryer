#!/usr/bin/env python3
"""
Plot WASP-107b observational data: Welbanks et al. 2024 combined dataset
alongside the two exoTEDRF NIRISS/SOSS orders.
"""

import numpy as np
import matplotlib.pyplot as plt

OBS_DIR = "../../obs_data"

# --- Load Welbanks et al. 2024 combined dataset ---
welbanks = np.loadtxt(f"{OBS_DIR}/WASP-107b_JWST_trans.txt", comments="#", usecols=(0, 1, 2, 3))
wl_w, dlam_w, depth_w, err_w = welbanks[:, 0], welbanks[:, 1], welbanks[:, 2], welbanks[:, 3]

# --- Load exoTEDRF NIRISS/SOSS orders ---
soss_o1 = np.loadtxt(f"{OBS_DIR}/W107b_exoTEDRF_NIRISS_SOSS_O1.txt")
wl_o1, depth_o1, err_o1 = soss_o1[:, 0], soss_o1[:, 1], soss_o1[:, 2]

soss_o2 = np.loadtxt(f"{OBS_DIR}/W107b_exoTEDRF_NIRISS_SOSS_O2.txt")
wl_o2, depth_o2, err_o2 = soss_o2[:, 0], soss_o2[:, 1], soss_o2[:, 2]

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 5))

ax.errorbar(wl_w, depth_w * 100, yerr=err_w * 100,
            fmt="o", ms=3, lw=0.8, color="steelblue", alpha=0.7,
            label="Welbanks+2024 (combined)")

ax.errorbar(wl_o1, depth_o1 * 100, yerr=err_o1 * 100,
            fmt="s", ms=3, lw=0.8, color="tomato", alpha=0.8,
            label="exoTEDRF NIRISS/SOSS Order 1")

ax.errorbar(wl_o2, depth_o2 * 100, yerr=err_o2 * 100,
            fmt="^", ms=3, lw=0.8, color="goldenrod", alpha=0.8,
            label="exoTEDRF NIRISS/SOSS Order 2")

ax.set_xlabel("Wavelength (μm)")
ax.set_ylabel("Transit depth (%)")
ax.set_title("WASP-107b transmission spectrum")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("obs_comparison.png", dpi=150)
print("Saved obs_comparison.png")
plt.show()
