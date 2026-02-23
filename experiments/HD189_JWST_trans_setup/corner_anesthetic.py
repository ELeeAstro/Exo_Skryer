#!/usr/bin/env python3
"""
corner_anesthetic.py
====================

Create a corner plot from nested_samples.csv using the anesthetic package.

Usage:
    python corner_anesthetic.py

Features:
    - 1D KDE line plots on diagonal
    - 2D KDE contour plots in lower triangle
    - Scatter points overlaid on contours
    - Quantile lines and labels
    - Seaborn colorblind palette
"""

from anesthetic import read_chains
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn colorblind palette
sns.set_palette("colorblind")
colors = sns.color_palette("colorblind")

# Read nested samples
samples = read_chains("nested_samples.csv")

# Define parameters to plot
params = ['R_p', 'T_iso', 'log_10_f_H2O']

# Configure plot types for each panel
# Use kde_1d instead of hist_1d for line plots on diagonal
kinds = {
    'diagonal': 'kde_1d',   # 1D KDE line plots on diagonal
    'lower': 'kde_2d',      # Contour plots in lower triangle
}

# Create corner plot with contours
axes = samples.plot_2d(params, kinds=kinds, color=colors[0])

# Overlay scatter points on the same axes
samples.plot_2d(axes, kinds={'lower': 'scatter_2d'}, alpha=0.3, color=colors[0])

# Add quantile lines and labels to diagonal KDE plots
quantiles = [0.16, 0.5, 0.84]
for i, param in enumerate(params):
    # Get the diagonal axis
    ax_diag = axes[param][param]

    # Calculate quantiles for this parameter
    param_quantiles = samples[param].quantile(quantiles)
    q16, q50, q84 = param_quantiles

    # Get y-axis limits for vertical lines
    ymin, ymax = ax_diag.get_ylim()

    # Add vertical lines at each quantile
    for q, qval in zip(quantiles, param_quantiles):
        linestyle = '--' if q == 0.5 else ':'  # Dashed for median, dotted for others
        ax_diag.axvline(qval, color=colors[0], linestyle=linestyle, linewidth=1.5, alpha=0.7)

    # Add text label with median and uncertainties
    # Format: median +upper -lower
    upper = q84 - q50
    lower = q50 - q16
    label_text = f"${q50:.3f}^{{+{upper:.3f}}}_{{-{lower:.3f}}}$"

    # Add text above the top of the plot (in figure coordinates)
    ax_diag.text(0.5, 1.05, label_text,
                 transform=ax_diag.transAxes,
                 fontsize=9,
                 verticalalignment='bottom',
                 horizontalalignment='center')

# Get the figure from the axes
fig = axes.figure if hasattr(axes, 'figure') else plt.gcf()

# Customize plot appearance
fig.suptitle('Posterior Corner Plot', y=0.995, fontsize=14)
plt.tight_layout()

# Save the plot
plt.savefig('corner_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('corner_plot.pdf', bbox_inches='tight')
print("[corner] Saved corner_plot.png and corner_plot.pdf")

# Show the plot
plt.show()
