#!/usr/bin/env python3
"""
plot_cloud_comparison.py
========================
Overlay the clear and cloudy emission forward models produced by
``run_forward_model.py`` (one ``spec_<case>_highres.txt`` per config).

Top panel: emergent flux. Bottom panel: cloudy / clear flux ratio.

Usage
-----
    python plot_cloud_comparison.py [--no-show]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent

# (file stem, label, colour, linestyle). Colours: fixed categorical order;
# the clear reference is neutral ink so the clouds carry the hues.
CASES = [
    ("clear",                "Clear",                                        "#2b2b2b", "-"),
    ("cloud_grey_slab",      "Grey slab (slab + grey_profile, EAA)",          "#2a78d6", "-"),
    ("cloud_powerlaw",       "Deck + power-law haze (powerlaw, EAA)",         "#eb6834", "--"),
    ("cloud_F18_exp",        "F18 (exponential, mono, EAA)",                  "#1baf7a", "-."),
    ("cloud_madt_lognormal", "MgSiO$_3$ MADT (exponential, log-normal, Toon89)", "#eda100", "-"),
    ("cloud_lxmie_slab",     "Mg$_2$SiO$_4$ LX-MIE (slab, mono, DDA)",        "#e87ba4", "--"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--outname", default="forward_spectrum_cloud_comparison")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    spectra = {}
    for stem, *_ in CASES:
        path = EXP_DIR / f"spec_{stem}_highres.txt"
        spectra[stem] = np.loadtxt(path)
    wl = spectra["clear"][:, 0]
    f_clear = spectra["clear"][:, 1]

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06},
    )

    for stem, label, colour, ls in CASES:
        d = spectra[stem]
        lw = 1.6 if stem == "clear" else 1.2
        ax.plot(d[:, 0], d[:, 1], color=colour, ls=ls, lw=lw, label=label, zorder=3 if stem == "clear" else 2)
        if stem != "clear":
            ratio = np.divide(d[:, 1], f_clear, out=np.full_like(f_clear, np.nan), where=f_clear > 0)
            axr.plot(d[:, 0], ratio, color=colour, ls=ls, lw=1.2)

    # Recessive grid / axes
    for a in (ax, axr):
        a.grid(True, which="major", color="0.88", lw=0.6)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.spines["left"].set_color("0.5")
        a.spines["bottom"].set_color("0.5")
        a.tick_params(colors="0.3")

    ax.set_ylabel(r"Flux [erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$]", fontsize=12)
    ax.set_title("Brown dwarf emission (T$_{\\rm int}$ = 1000 K, log g = 4.87): cloud schemes", fontsize=13, loc="left")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_ylim(bottom=0)

    axr.axhline(1.0, color="0.5", lw=0.8, ls=":")
    axr.set_ylabel("Cloudy / clear", fontsize=12)
    axr.set_ylim(0, 1.05)
    axr.set_xlabel("Wavelength [µm]", fontsize=12)
    axr.set_xscale("log")

    ticks = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    ticks = ticks[(ticks >= wl.min() * 0.95) & (ticks <= wl.max() * 1.05)]
    axr.set_xticks(ticks)
    axr.set_xticklabels([f"{t:g}" for t in ticks])
    axr.set_xlim(wl.min(), wl.max())

    fig.savefig(EXP_DIR / f"{args.outname}.png", dpi=250, bbox_inches="tight")
    fig.savefig(EXP_DIR / f"{args.outname}.pdf", bbox_inches="tight")
    print(f"[plot] Saved: {EXP_DIR / args.outname}.png, .pdf")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
