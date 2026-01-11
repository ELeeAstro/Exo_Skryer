#!/usr/bin/env python3
"""
compare_samplers.py
===================

Compare Dynesty and PyMultiNest posteriors on the same corner plot.

Usage:
    python compare_samplers.py

Reads: dynesty_results.pkl, pymultinest_post_equal_weights.dat
Saves: corner_comparison.png
"""

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import corner

# Parameter names - UPDATE THESE to match your config!
param_names = [
    r"$\log_{10}g$",
    r"$R_p$",
    r"$T_{\rm strat}$",
    r"$\log_{10}f_{\rm H_2O}$",
    r"$\log_{10}f_{\rm Na}$",
    r"$\log_{10}f_{\rm K}$",
    r"$\log_{10}q_c$",
]

print("="*60)
print("Sampler Comparison")
print("="*60)

# ============================================================================
# Load Dynesty
# ============================================================================
dynesty_file = Path("dynesty_results.pkl")
if dynesty_file.exists():
    print("\n[Dynesty]")
    with open(dynesty_file, "rb") as f:
        results_dy = pickle.load(f)

    samples_dy = results_dy['samples']
    weights_dy = np.exp(results_dy['logwt'] - results_dy['logz'][-1])

    # Resample to equal weights
    n_samples = 10000
    indices = np.random.choice(
        len(samples_dy),
        size=n_samples,
        replace=True,
        p=weights_dy / weights_dy.sum()
    )
    samples_dy_equal = samples_dy[indices]

    logZ_dy = results_dy['logz'][-1]
    logZ_err_dy = results_dy['logzerr'][-1]

    print(f"  logZ = {logZ_dy:.2f} ± {logZ_err_dy:.2f}")
    print(f"  Samples: {len(samples_dy_equal)}")
    has_dynesty = True
else:
    print("\n[Dynesty] NOT FOUND")
    has_dynesty = False

# ============================================================================
# Load PyMultiNest
# ============================================================================
pymn_file = Path("pymultinest_post_equal_weights.dat")
if pymn_file.exists():
    print("\n[PyMultiNest]")
    data_pymn = np.loadtxt(pymn_file)
    samples_pymn = data_pymn[:, 2:]  # Skip weight and -2*logL

    try:
        stats_file = Path("pymultinest_stats.dat")
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            parts = lines[0].split()
            logZ_pymn = float(parts[0])
            logZ_err_pymn = float(parts[1])
    except:
        logZ_pymn = None
        logZ_err_pymn = None

    if logZ_pymn is not None:
        print(f"  logZ = {logZ_pymn:.2f} ± {logZ_err_pymn:.2f}")
    print(f"  Samples: {len(samples_pymn)}")
    has_pymn = True
else:
    print("\n[PyMultiNest] NOT FOUND")
    has_pymn = False

if not (has_dynesty or has_pymn):
    raise FileNotFoundError("No sampler output files found!")

# ============================================================================
# Create comparison corner plot
# ============================================================================
print("\nCreating comparison corner plot...")

fig = plt.figure(figsize=(12, 12))

# Plot Dynesty first (blue, lighter)
if has_dynesty:
    fig = corner.corner(
        samples_dy_equal,
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        levels=(0.68, 0.95),
        color='#1f77b4',
        hist_kwargs={'alpha': 0.6},
        fig=fig,
    )

# Overlay PyMultiNest (orange, darker)
if has_pymn:
    fig = corner.corner(
        samples_pymn,
        labels=param_names if not has_dynesty else None,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=has_dynesty is False,  # Only show titles if Dynesty not shown
        title_fmt=".3f",
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        levels=(0.68, 0.95),
        color='#ff7f0e',
        hist_kwargs={'alpha': 0.6},
        fig=fig,
    )

# Add legend
legend_elements = []
if has_dynesty:
    from matplotlib.lines import Line2D
    legend_elements.append(
        Line2D([0], [0], color='#1f77b4', lw=2,
               label=f'Dynesty (logZ={logZ_dy:.1f}±{logZ_err_dy:.1f})')
    )
if has_pymn and logZ_pymn is not None:
    from matplotlib.lines import Line2D
    legend_elements.append(
        Line2D([0], [0], color='#ff7f0e', lw=2,
               label=f'PyMultiNest (logZ={logZ_pymn:.1f}±{logZ_err_pymn:.1f})')
    )
elif has_pymn:
    from matplotlib.lines import Line2D
    legend_elements.append(
        Line2D([0], [0], color='#ff7f0e', lw=2, label='PyMultiNest')
    )

if legend_elements:
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)

fig.suptitle("Sampler Comparison", fontsize=16, y=1.0)

# Save
output_file = "corner_comparison.png"
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"\nSaved comparison plot to {output_file}")
plt.close()

# ============================================================================
# Print parameter comparison
# ============================================================================
print("\n" + "="*60)
print("Parameter Comparison (median ± 1σ)")
print("="*60)

for i, name in enumerate(param_names):
    print(f"\n{name}:")

    if has_dynesty and i < samples_dy_equal.shape[1]:
        med_dy = np.median(samples_dy_equal[:, i])
        lo_dy = np.percentile(samples_dy_equal[:, i], 16)
        hi_dy = np.percentile(samples_dy_equal[:, i], 84)
        print(f"  Dynesty     : {med_dy:.3f} +{hi_dy-med_dy:.3f} -{med_dy-lo_dy:.3f}")

    if has_pymn and i < samples_pymn.shape[1]:
        med_pymn = np.median(samples_pymn[:, i])
        lo_pymn = np.percentile(samples_pymn[:, i], 16)
        hi_pymn = np.percentile(samples_pymn[:, i], 84)
        print(f"  PyMultiNest : {med_pymn:.3f} +{hi_pymn-med_pymn:.3f} -{med_pymn-lo_pymn:.3f}")

    if has_dynesty and has_pymn and i < min(samples_dy_equal.shape[1], samples_pymn.shape[1]):
        diff = med_pymn - med_dy
        print(f"  Difference  : {diff:+.3f} ({100*diff/med_dy:+.1f}%)")

print("\n" + "="*60)
print("Done!")
