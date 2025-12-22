#!/usr/bin/env python3
"""Convert Gliese 229B emission data to retrieval format with proper flux scaling."""

import numpy as np

# Constants
pc_to_cm = 3.086e18  # 1 parsec in cm
R_jupiter_cm = 7.1492e9  # Jupiter radius in cm
distance_pc = 10.0  # Data calibrated to 10 pc
R_gliese = 0.94 * R_jupiter_cm  # 0.94 Jupiter radii

# Read the original file, skipping header (first 36 lines)
data = np.loadtxt('Gliese_229B_emission_original.txt', skiprows=36)

# Extract columns
wavelength_um = data[:, 0]  # Already in microns
flux_erg_s_cm2_A_10pc = data[:, 1]  # In erg/s/cm^2/Å at 10 pc
error_erg_s_cm2_A_10pc = data[:, 2]  # In erg/s/cm^2/Å at 10 pc

# Convert from 10 pc to surface flux
# F_surface = F_observed * (d/R)^2
distance_cm = distance_pc * pc_to_cm
flux_scaling = (distance_cm / R_gliese)**2

flux_erg_s_cm2_A_surf = flux_erg_s_cm2_A_10pc * flux_scaling
error_erg_s_cm2_A_surf = error_erg_s_cm2_A_10pc * flux_scaling

# Convert flux from erg/s/cm^2/Å to erg/s/cm^2/cm
# 1 cm = 10^8 Å, so multiply by 10^8
flux_erg_s_cm2_cm = flux_erg_s_cm2_A_surf * 1e8
error_erg_s_cm2_cm = error_erg_s_cm2_A_surf * 1e8

# Define instrument boundaries from header information
# 1. HST STIS optical: 0.52 - 1.02 µm
# 2. JHK spectrum: 1.025 - 2.52 µm
# 3. L spectrum: 2.98 - 4.15 µm
# 4. M spectrum: 4.5 - 5.1 µm

instrument_boundaries = [
    (0.52, 1.02, "HST STIS optical"),
    (1.025, 2.52, "JHK spectrum"),
    (2.98, 4.15, "L spectrum"),
    (4.5, 5.1, "M spectrum")
]

# Assign each wavelength point to an instrument segment
segment_ids = np.zeros(len(wavelength_um), dtype=int)
for i, wave in enumerate(wavelength_um):
    for seg_id, (wmin, wmax, name) in enumerate(instrument_boundaries):
        if wmin <= wave <= wmax:
            segment_ids[i] = seg_id
            break

# Find contiguous segments
segment_starts = [0]
segment_instrument = [segment_ids[0]]
for i in range(1, len(segment_ids)):
    if segment_ids[i] != segment_ids[i-1]:
        segment_starts.append(i)
        segment_instrument.append(segment_ids[i])
segment_starts.append(len(wavelength_um))

print(f"Detected {len(segment_starts)-1} instrument segments:")
for i in range(len(segment_starts)-1):
    start = segment_starts[i]
    end = segment_starts[i+1]
    inst_id = segment_instrument[i]
    inst_name = instrument_boundaries[inst_id][2]
    print(f"  Segment {i+1}: {wavelength_um[start]:.3f} - {wavelength_um[end-1]:.3f} µm "
          f"({end-start} points) - {inst_name}")

# Calculate half-bandwidths for each segment
half_bandwidth = np.zeros_like(wavelength_um)

for i in range(len(segment_starts)-1):
    start = segment_starts[i]
    end = segment_starts[i+1]
    segment_wave = wavelength_um[start:end]
    segment_half_bw = np.zeros(len(segment_wave))

    if len(segment_wave) == 1:
        # Single point: estimate from typical spacing in dataset
        segment_half_bw[0] = 0.005  # 0.005 micron = 50 Å
    else:
        for j in range(len(segment_wave)):
            if j == 0:
                # First point: use spacing to next point
                spacing = segment_wave[1] - segment_wave[0]
                segment_half_bw[j] = spacing / 2.0
            elif j == len(segment_wave) - 1:
                # Last point: use spacing to previous point
                spacing = segment_wave[-1] - segment_wave[-2]
                segment_half_bw[j] = spacing / 2.0
            else:
                # Middle points: use average of spacing to neighbors
                # Half-bandwidth = (wave[i+1] - wave[i-1]) / 4
                segment_half_bw[j] = (segment_wave[j+1] - segment_wave[j-1]) / 4.0

    half_bandwidth[start:end] = segment_half_bw

# Write the output file
with open('Gliese_229B_emission.txt', 'w') as f:
    for i in range(len(wavelength_um)):
        f.write(f"{wavelength_um[i]:.6e}  {half_bandwidth[i]:.6e}  "
                f"{flux_erg_s_cm2_cm[i]:.6e}  {error_erg_s_cm2_cm[i]:.6e}  None\n")

print(f"\nConversion complete:")
print(f"  Total data points: {len(wavelength_um)}")
print(f"  Flux scaling factor (d/R)^2: {flux_scaling:.6e}")
print(f"  Distance: {distance_pc} pc = {distance_cm:.6e} cm")
print(f"  Radius: {0.94} R_Jup = {R_gliese:.6e} cm")
print(f"\nHalf-bandwidth statistics by instrument:")
for i in range(len(segment_starts)-1):
    start = segment_starts[i]
    end = segment_starts[i+1]
    inst_id = segment_instrument[i]
    inst_name = instrument_boundaries[inst_id][2]
    seg_hbw = half_bandwidth[start:end]
    print(f"  {inst_name}:")
    print(f"    Min: {np.min(seg_hbw):.6e} µm, Max: {np.max(seg_hbw):.6e} µm, "
          f"Median: {np.median(seg_hbw):.6e} µm")
print(f"\nOutput written to: Gliese_229B_emission.txt")
