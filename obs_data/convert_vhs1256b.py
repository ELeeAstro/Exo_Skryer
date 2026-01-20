#!/usr/bin/env python3
"""
Convert VHS1256b_V2.txt to retrieval format (observed_data.csv).

Input format: wavelength, flux, flux_error, observing_mode
Output format: lam_um, dlam_um, depth, depth_sigma, response_mode
"""

import numpy as np
from pathlib import Path

def convert_vhs1256b_to_retrieval_format(
    input_file: Path,
    output_file: Path,
):
    """Convert VHS1256b high-res spectrum to binned retrieval format."""

    # Read the data, skipping comment lines and NaN values
    data = []
    skipped_nan = 0
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                wl = float(parts[0])
                flux = float(parts[1])
                flux_err = float(parts[2])
                mode = parts[3].strip() if len(parts) > 3 else ""

                # Skip lines with NaN values
                if np.isnan(wl) or np.isnan(flux) or np.isnan(flux_err):
                    skipped_nan += 1
                    continue

                data.append([wl, flux, flux_err, mode])

    print(f"Read {len(data)} data points")
    if skipped_nan > 0:
        print(f"Skipped {skipped_nan} data points with NaN values")

    # Convert to structured array to keep instrument mode
    wl = np.array([d[0] for d in data])
    flux = np.array([d[1] for d in data])
    flux_err = np.array([d[2] for d in data])
    modes = [d[3] for d in data]

    # Convert flux units from W/m²/µm to erg/s/cm²/cm
    # Conversion factor: 1 W/m²/µm = 10^7 erg/s/cm²/cm
    # Because: 1 W = 10^7 erg/s, 1 m² = 10^4 cm², 1 µm = 10^-4 cm
    # So: 10^7 / 10^4 * 10^4 = 10^7
    flux_conversion = 1e7
    flux = flux * flux_conversion
    flux_err = flux_err * flux_conversion

    print(f"Wavelength range: {wl[0]:.4f} - {wl[-1]:.4f} microns")
    print(f"Converted flux units from W/m²/µm to erg/s/cm²/cm (factor: {flux_conversion:.0e})")

    # Group data by instrument
    unique_modes = []
    mode_groups = {}
    current_mode = None

    for i, mode in enumerate(modes):
        if mode != current_mode:
            current_mode = mode
            unique_modes.append(mode)
            mode_groups[mode] = []
        mode_groups[mode].append(i)

    print(f"\nFound {len(unique_modes)} instrument modes:")
    for mode in unique_modes:
        indices = mode_groups[mode]
        wl_range = (wl[indices[0]], wl[indices[-1]])
        print(f"  {mode}: {len(indices)} points, λ = {wl_range[0]:.4f} - {wl_range[1]:.4f} µm")

    # Calculate bin half-widths (dlam_um) for each instrument separately
    dlam = np.zeros_like(wl)

    for mode, indices in mode_groups.items():
        indices = np.array(indices)
        n_pts = len(indices)

        if n_pts == 1:
            # Single point: estimate from typical resolution
            dlam[indices[0]] = wl[indices[0]] * 0.005  # Assume R~200
        elif n_pts == 2:
            # Two points: use spacing
            spacing = wl[indices[1]] - wl[indices[0]]
            dlam[indices[0]] = spacing / 2.0
            dlam[indices[1]] = spacing / 2.0
        else:
            # Multiple points: calculate from adjacent spacing
            # First point
            dlam[indices[0]] = (wl[indices[1]] - wl[indices[0]]) / 2.0

            # Middle points
            for j in range(1, n_pts - 1):
                i = indices[j]
                dlam[i] = (wl[indices[j+1]] - wl[indices[j-1]]) / 4.0

            # Last point
            dlam[indices[-1]] = (wl[indices[-1]] - wl[indices[-2]]) / 2.0

    # Sort all arrays by wavelength (low to high)
    sort_idx = np.argsort(wl)
    wl = wl[sort_idx]
    dlam = dlam[sort_idx]
    flux = flux[sort_idx]
    flux_err = flux_err[sort_idx]

    print(f"\nData sorted by wavelength: {wl[0]:.4f} - {wl[-1]:.4f} microns")

    # Write output in retrieval format (space-separated)
    with open(output_file, 'w') as f:
        # Write header
        f.write("lam_um dlam_um depth depth_sigma response_mode\n")

        # Write all data points except last as boxcar
        for i in range(len(wl) - 1):
            f.write(f"{wl[i]:.6g} {dlam[i]:.6g} {flux[i]:.6g} {flux_err[i]:.6g} boxcar\n")

        # Write last point with MIRI_MRS_4A response
        i = len(wl) - 1
        f.write(f"{wl[i]:.6g} {dlam[i]:.6g} {flux[i]:.6g} {flux_err[i]:.6g} MIRI_MRS_4A\n")

    print(f"Wrote {len(wl)} data points to {output_file}")
    print(f"All points use 'boxcar' except the last which uses 'MIRI_MRS_4A'")


if __name__ == "__main__":
    input_file = Path("/Users/gl20y334/Desktop/Exo_Skryer/obs_data/VHS1256b_V2.txt")
    output_file = Path("/Users/gl20y334/Desktop/Exo_Skryer/obs_data/VHS1256b_observed_data.txt")

    convert_vhs1256b_to_retrieval_format(input_file, output_file)

    print(f"\nConversion complete!")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
