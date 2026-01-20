#!/usr/bin/env python
"""
Convert WASP-69b emission data from Schlawin et al. 2024 TSV format
to the standard format used by Exo_Skryer (same as HD189733b_emission_JWST.txt).

Combines data from all TSV files covering different wavelength ranges.
"""

import numpy as np
import matplotlib.pyplot as plt

# Input files (covering different wavelength ranges)
input_files = [
    "1_table_WASP-69-b-Schlawin-et-al.-2024.tsv",  # NIRSpec: 2-5 um
    "2_table_WASP-69-b-Schlawin-et-al.-2024.tsv",  # MIRI: 5-12 um
]
output_file = "WASP-69b_emission_JWST.txt"

# Read and combine all TSV files
all_data = []
for input_file in input_files:
    try:
        # Columns: wavelength, bandwidth, eclipse_depth, err_pos, err_neg
        data = np.genfromtxt(input_file, delimiter='\t', skip_header=1, usecols=(0, 1, 2, 3, 4))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        all_data.append(data)
        print(f"Read {len(data)} points from {input_file}")
    except Exception as e:
        print(f"Warning: Could not read {input_file}: {e}")

# Combine all data
combined_data = np.vstack(all_data)

# Sort by wavelength
sort_idx = np.argsort(combined_data[:, 0])
combined_data = combined_data[sort_idx]

wavelength = combined_data[:, 0]              # Central wavelength (um)
bandwidth = combined_data[:, 1]               # Full bandwidth (um)
eclipse_depth_pct = combined_data[:, 2]       # Eclipse depth (%)
err_pos_pct = combined_data[:, 3]             # Positive error (%)
err_neg_pct = np.abs(combined_data[:, 4])     # Negative error (%), take absolute value

# Convert to required format
half_bandwidth = bandwidth / 2.0                      # Half-bandwidth
eclipse_depth = eclipse_depth_pct / 100.0             # Fractional eclipse depth
err_pos = err_pos_pct / 100.0                         # Fractional positive error
err_neg = err_neg_pct / 100.0                         # Fractional negative error

# Write output file (matching HD189733b format)
with open(output_file, 'w') as f:
    f.write("#Central_WL(μm)   Half_BW(μm)    Eclipse_Depth      Err_Pos       Err_Neg       Mode      \n")
    f.write("#" + "-" * 90 + "\n")

    for i in range(len(wavelength)):
        f.write(f"{wavelength[i]:<17.4f}{half_bandwidth[i]:<15.5f}{eclipse_depth[i]:<19.6g}{err_pos[i]:<14.3g}{err_neg[i]:<14.3g}boxcar    \n")

print(f"\nTotal: {len(wavelength)} data points combined")
print(f"Wavelength range: {wavelength.min():.3f} - {wavelength.max():.3f} um")
print(f"Output written to: {output_file}")

# Plot the spectrum
fig, ax = plt.subplots(figsize=(12, 6))

# Use asymmetric error bars
ax.errorbar(wavelength, eclipse_depth * 100, yerr=[err_neg * 100, err_pos * 100],
            fmt='o', markersize=4, capsize=2, color='darkorange',
            ecolor='moccasin', alpha=0.8, label='WASP-69b (Schlawin et al. 2024)')

ax.set_xlabel('Wavelength (μm)', fontsize=12)
ax.set_ylabel('Eclipse Depth (%)', fontsize=12)
ax.set_title('WASP-69b JWST Emission Spectrum', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
