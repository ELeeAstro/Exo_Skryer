#!/usr/bin/env python3
"""Plot Gliese 229B emission data with error bars."""

import numpy as np
import matplotlib.pyplot as plt

# Read the data file
# Format: wavelength, half_bandwidth, flux, error, response_mode
data = np.loadtxt('Gliese_229B_emission.txt', dtype={'names': ('wave', 'half_bw', 'flux', 'error', 'mode'),
                                                       'formats': ('f8', 'f8', 'f8', 'f8', 'U10')})

wavelength = data['wave']
half_bandwidth = data['half_bw']
flux = data['flux']
flux_error = data['error']

# Get the b parameter bounds
b_min = np.log10(0.1 * np.min(flux_error))
b_max = np.log10(10.0 * np.max(flux_error))

print(b_min, b_max)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot data with error bars
ax.errorbar(wavelength, flux,
            xerr=half_bandwidth,  # x error bars from half-bandwidth
            yerr=flux_error,      # y error bars from flux error
            fmt='o',              # circle markers
            markersize=2,
            elinewidth=0.5,
            capsize=0,
            alpha=0.6,
            label='Gliese 229B (10 pc)')

# Labels and formatting
ax.set_xlabel('Wavelength [µm]', fontsize=14)
ax.set_ylabel('Flux [erg s⁻¹ cm⁻² cm⁻¹]', fontsize=14)
ax.set_title('Gliese 229B Emission Spectrum', fontsize=16)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

# Use log scale for y-axis (flux varies over orders of magnitude)
ax.set_yscale('log')

# Tight layout
fig.tight_layout()

# Show the plot
plt.show()
