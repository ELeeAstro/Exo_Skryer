#!/usr/bin/env python
"""
Convert WASP-107b observational data from Welbanks et al. 2024 TSV format
to the standard format used by Exo_Skryer (same as WASP-17b_JWST.txt).

Combines data from all 5 TSV files covering different wavelength ranges.
"""

import numpy as np
import matplotlib.pyplot as plt

# Input files (covering different wavelength ranges)
input_files = [
    "1_table_WASP-107-b-Welbanks-et-al.-2024.tsv",  # MIRI LRS: 5-14 um
    "2_table_WASP-107-b-Welbanks-et-al.-2024.tsv",  # NIRSpec G395H: 2.5-4 um
    "3_table_WASP-107-b-Welbanks-et-al.-2024.tsv",  # NIRSpec G395H: 4-5 um
    "4_table_WASP-107-b-Welbanks-et-al.-2024.tsv",  # NIRSpec G140H: 1.1-1.6 um
    "5_table_WASP-107-b-Welbanks-et-al.-2024.tsv",  # NIRSpec G140H: 0.9-1.1 um
]
output_file = "WASP-107b_JWST.txt"

# Offset to apply to MIRI LRS data (in ppm, converted to %)
MIRI_OFFSET_PPM = 282.0
MIRI_OFFSET_PCT = MIRI_OFFSET_PPM / 10000.0  # Convert ppm to % (285 ppm = 0.0285%)

# Read and combine all TSV files
all_data = []
for input_file in input_files:
    try:
        data = np.genfromtxt(input_file, delimiter='\t', skip_header=1, usecols=(0, 1, 2, 3))
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Apply offset to MIRI LRS data (file 1)
        if "1_table" in input_file:
            data[:, 2] += MIRI_OFFSET_PCT  # Add offset to transit depth column
            print(f"Read {len(data)} points from {input_file} (MIRI LRS, +{MIRI_OFFSET_PPM:.0f} ppm offset applied)")
        else:
            print(f"Read {len(data)} points from {input_file}")

        all_data.append(data)
    except Exception as e:
        print(f"Warning: Could not read {input_file}: {e}")

# Combine all data
combined_data = np.vstack(all_data)

# Sort by wavelength
sort_idx = np.argsort(combined_data[:, 0])
combined_data = combined_data[sort_idx]

wavelength = combined_data[:, 0]          # Central wavelength (um)
bandwidth = combined_data[:, 1]           # Half-bandwidth (um) - TSV column labeled "BANDWIDTH"
transit_depth_pct = combined_data[:, 2]   # Transit depth (%)
error_pct = np.abs(combined_data[:, 3])   # Error (%), take absolute value

# Convert to required format
# Note: The BANDWIDTH column in the TSV files is already the half-bandwidth,
# NOT the full bandwidth. Using it directly without dividing by 2.
half_bandwidth = bandwidth                          # Already half-bandwidth
transit_depth = transit_depth_pct / 100.0           # Fractional transit depth
uncertainty = error_pct / 100.0                     # Fractional uncertainty

# Write output file
with open(output_file, 'w') as f:
    f.write("# WASP-107b observational data from Welbanks et al. 2024\n")
    f.write("# Format: wavelength(um)  delta_wavelength(um)  transit_depth  uncertainty  response_mode\n")
    f.write("# Combined JWST data: NIRSpec G140H, NIRSpec G395H, MIRI LRS\n")
    f.write(f"# MIRI LRS data shifted by +{MIRI_OFFSET_PPM:.0f} ppm to align with NIRSpec\n")
    f.write("# Data sorted by wavelength (lowest to highest)\n")
    f.write("# " + "=" * 80 + "\n")

    for i in range(len(wavelength)):
        f.write(f"{wavelength[i]:.6f} {half_bandwidth[i]:.6f} {transit_depth[i]:.8f} {uncertainty[i]:.8f} boxcar\n")

print(f"\nTotal: {len(wavelength)} data points combined")
print(f"Wavelength range: {wavelength.min():.3f} - {wavelength.max():.3f} um")
print(f"Output written to: {output_file}")

# Plot the spectrum with wavelength bin coverage visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

# Top panel: Transit depth spectrum
ax1 = axes[0]
ax1.errorbar(wavelength, transit_depth * 100, yerr=uncertainty * 100,
             fmt='o', markersize=4, capsize=2, color='royalblue',
             ecolor='lightblue', alpha=0.8, label='WASP-107b (Welbanks et al. 2024)')

ax1.set_xlabel('Wavelength (μm)', fontsize=12)
ax1.set_ylabel('Transit Depth (%)', fontsize=12)
ax1.set_title('WASP-107b JWST Transmission Spectrum', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Bottom panel: Wavelength bin coverage (to verify no gaps)
ax2 = axes[1]

# Calculate bin edges
bin_lo = wavelength - half_bandwidth
bin_hi = wavelength + half_bandwidth

# Plot each bin as a horizontal bar
for i in range(len(wavelength)):
    color = 'C0' if half_bandwidth[i] < 0.02 else ('C1' if half_bandwidth[i] < 0.05 else 'C2')
    ax2.barh(0.5, bin_hi[i] - bin_lo[i], left=bin_lo[i], height=0.8,
             color=color, alpha=0.6, edgecolor='black', linewidth=0.3)

# Mark gaps (if any)
gaps = []
for i in range(len(wavelength) - 1):
    gap = bin_lo[i+1] - bin_hi[i]
    if gap > 0.001:  # Gap > 1 nm
        gaps.append((bin_hi[i], bin_lo[i+1], gap))
        ax2.axvspan(bin_hi[i], bin_lo[i+1], color='red', alpha=0.3)

ax2.set_xlabel('Wavelength (μm)', fontsize=12)
ax2.set_ylabel('Bins', fontsize=12)
ax2.set_title(f'Wavelength Bin Coverage (gaps shown in red) - {len(gaps)} gaps found', fontsize=12)
ax2.set_ylim(0, 1)
ax2.set_yticks([])
ax2.grid(True, alpha=0.3, axis='x')

# Print gap summary
if gaps:
    print(f"\nWARNING: {len(gaps)} gaps found between wavelength bins:")
    for start, end, size in gaps[:10]:
        print(f"  Gap at {start:.4f}-{end:.4f} μm (size: {size*1000:.2f} nm)")
    if len(gaps) > 10:
        print(f"  ... and {len(gaps)-10} more gaps")
else:
    print("\nSUCCESS: No gaps between wavelength bins - bins are contiguous!")

plt.tight_layout()
plt.show()
