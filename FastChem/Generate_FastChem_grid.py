"""
Generate FastChem chemistry grid matching opacity T-P grid.
Loops over M/H and C/O ratios, modifying element abundances.
Outputs NPZ files for use in retrieval calculations.
"""
import pyfastchem
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy import constants as const
import shutil
import zarr
from zarr.storage import ZipStore


def round_in_log(x):
    log_floor = np.floor(np.log10(x))
    pre_exponent = np.round(x * 10 ** -log_floor, 1)
    return pre_exponent * 10 ** log_floor


def safe_log10(arr, floor=1e-300):
    """
    Compute log10 safely for positive values while preserving NaNs.
    """
    x = np.asarray(arr, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    finite = np.isfinite(x)
    out[finite] = np.log10(np.maximum(x[finite], floor))
    return out


def load_tp_grid_from_opacity(path):
    """
    Load linear temperature [K] and pressure [bar] grids from an opacity table.

    Supports:
    - Zarr directory stores
    - Zarr zip stores (.zarr.zip)
    - legacy NPZ files with temperature/pressure arrays
    """
    path = os.path.expanduser(path)

    if path.endswith(".npz"):
        data = np.load(path)
        T = np.asarray(data["temperature"], dtype=float)
        p = np.asarray(data["pressure"], dtype=float)
        return T, p

    if path.endswith(".zarr.zip"):
        with ZipStore(path, mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            T = np.asarray(root["temperature"][:], dtype=float)
            p = np.asarray(root["pressure"][:], dtype=float)
            return T, p

    root = zarr.open_group(path, mode="r")
    T = np.asarray(root["temperature"][:], dtype=float)
    p = np.asarray(root["pressure"][:], dtype=float)
    return T, p


def modify_abundances(input_file, output_file, M_H, C_O_ratio):
    """
    Modify element abundances for given [M/H] and C/O ratio.

    Parameters:
    -----------
    input_file : str
        Path to base abundance file (e.g., asplund_2020.dat)
    output_file : str
        Path to save modified abundance file
    M_H : float
        Metallicity [M/H] = log10(M/M_solar)
    C_O_ratio : float
        Carbon to oxygen ratio (linear, not log)
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # First pass: find oxygen abundance
    O_abundance_base = None
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] == 'O':
            O_abundance_base = float(parts[1])
            break

    if O_abundance_base is None:
        raise ValueError("Oxygen abundance not found in input file!")

    # Calculate modified oxygen abundance
    O_abundance_new = O_abundance_base + M_H

    # Second pass: modify all abundances
    modified_lines = []

    for line in lines:
        # Keep comments and empty lines
        if line.startswith('#') or line.strip() == '':
            modified_lines.append(line)
            continue

        parts = line.split()
        if len(parts) < 2:
            modified_lines.append(line)
            continue

        element = parts[0]
        abundance = float(parts[1])

        # Don't modify H, He, or electron
        if element in ['H', 'He', 'e-']:
            modified_lines.append(line)
            continue

        # Apply metallicity enhancement to all metals
        new_abundance = abundance + M_H

        # Special handling for C to achieve desired C/O ratio
        if element == 'C':
            # Set C abundance to achieve desired C/O ratio
            # C/O = 10^(log_C - log_O)
            # log_C = log(C/O) + log_O
            C_abundance = np.log10(C_O_ratio) + O_abundance_new
            modified_lines.append(f"{element}  {C_abundance:.8f}\n")
        else:
            modified_lines.append(f"{element}  {new_abundance:.8f}\n")

    with open(output_file, 'w') as f:
        f.writelines(modified_lines)

    return output_file

# Load T and p grid from opacity data to ensure exact matching.
# Update this path if you want FastChem to follow a different opacity reference file.
opacity_grid_file = '../opac_data/ck/H2O_ck_R1000.zarr.zip'
T, p = load_tp_grid_from_opacity(opacity_grid_file)

# Setup M/H and C/O grid
log_M_H = np.linspace(-2, 3, 20)  # [M/H] from -2 to +3 dex
log_C_O = np.linspace(-1, 1, 10)  # log(C/O) from -1 to +1 dex
M_H_grid = log_M_H  # These are already in log space
C_O_grid = 10.0**log_C_O  # Convert to linear C/O ratio

print(f"Temperature grid: {len(T)} points")
print(f"Pressure grid: {len(p)} points")
print(f"log_M_H grid: {log_M_H}")
print(f"log_C_O grid: {log_C_O}")
print(f"Metallicity grid: {len(M_H_grid)} points, [M/H] = {M_H_grid}")
print(f"C/O ratio grid: {len(C_O_grid)} points, C/O = {C_O_grid}")
print(f"Total grid points: {len(T)} x {len(p)} x {len(M_H_grid)} x {len(C_O_grid)} = {len(T) * len(p) * len(M_H_grid) * len(C_O_grid)}")

# Fastchem output dir
output_dir = './'
temp_dir = './temp_abundances/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Species to track
plot_species = ['H2O1', 'C1O2', 'C1O1', 'C1H4', 'H3N1']
plot_species_labels = ['H2O', 'CO2', 'CO', 'CH4', 'NH3']

# Base abundance file
base_abundance_file = '/Users/elspeth/fastchem/input/element_abundances/asplund_2020.dat'
logK_file = '/Users/elspeth/fastchem/input/logK/logK.dat'
input_file = '/Users/elspeth/fastchem/input/logK/parameters_py.dat'

# Get grid dimensions
n_T = len(T)
n_p = len(p)
n_MH = len(M_H_grid)
n_CO = len(C_O_grid)

# Pre-allocate arrays for results (5D: T x P x M/H x C/O x Species)
# We'll get n_species after initializing FastChem
n_species = None

print("\nInitializing FastChem to get species count...")
temp_abund = temp_dir + 'temp_init.dat'
modify_abundances(base_abundance_file, temp_abund, 0.0, 0.55)
fastchem_init = pyfastchem.FastChem(temp_abund, logK_file, 1)
n_species = fastchem_init.getGasSpeciesNumber()
species_names = [fastchem_init.getGasSpeciesSymbol(i) for i in range(n_species)]
print(f"Number of gas species: {n_species}")

# Pre-allocate full 5D arrays
mixing_ratios_5d = np.zeros((n_T, n_p, n_MH, n_CO, n_species))
mean_molecular_weight_5d = np.zeros((n_T, n_p, n_MH, n_CO))
success_flags_5d = np.zeros((n_T, n_p, n_MH, n_CO), dtype=bool)

# Loop through M/H and C/O
total_iterations = n_MH * n_CO * n_T * n_p
current_iteration = 0
n_success_total = 0
n_failed_total = 0

print("\nComputing FastChem grid over T, P, [M/H], and C/O...")
print("="*70)

for i_mh, M_H in enumerate(M_H_grid):
    for i_co, C_O in enumerate(C_O_grid):
        # Create modified abundance file for this M/H and C/O
        abund_file = temp_dir + f'abundances_MH{M_H:.2f}_CO{C_O:.3f}.dat'
        modify_abundances(base_abundance_file, abund_file, M_H, C_O)

        # Initialize FastChem with new abundances
        fastchem = pyfastchem.FastChem(abund_file, logK_file, 1)
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()

        print(f"\n[M/H]={M_H:+.2f}, C/O={C_O:.3f} ({i_mh+1}/{n_MH}, {i_co+1}/{n_CO})")

        # Loop through T-P grid
        for i_t in range(n_T):
            for i_p in range(n_p):
                # Set T and P
                input_data.temperature = np.array([T[i_t]])
                input_data.pressure = np.array([p[i_p]])

                # Calculate
                fastchem_flag = fastchem.calcDensities(input_data, output_data)

                if fastchem_flag == pyfastchem.FASTCHEM_SUCCESS:
                    # Convert number densities to mixing ratios
                    gas_number_density = p[i_p] * 1e6 / (const.k_B.cgs.value * T[i_t])
                    mixing_ratios_5d[i_t, i_p, i_mh, i_co, :] = np.array(output_data.number_densities[0]) / gas_number_density
                    mean_molecular_weight_5d[i_t, i_p, i_mh, i_co] = output_data.mean_molecular_weight[0]
                    success_flags_5d[i_t, i_p, i_mh, i_co] = True
                    n_success_total += 1
                else:
                    mixing_ratios_5d[i_t, i_p, i_mh, i_co, :] = np.nan
                    mean_molecular_weight_5d[i_t, i_p, i_mh, i_co] = np.nan
                    success_flags_5d[i_t, i_p, i_mh, i_co] = False
                    n_failed_total += 1

                current_iteration += 1

        # Progress for this M/H, C/O combination
        progress_pct = 100 * current_iteration / total_iterations
        print(f"  Progress: {current_iteration}/{total_iterations} ({progress_pct:.1f}%) - Success: {n_success_total}, Failed: {n_failed_total}")

print(f"\n{'='*70}")
print(f"FastChem calculation complete:")
print(f"  Total calculations: {total_iterations}")
print(f"  Success: {n_success_total}/{total_iterations} ({100*n_success_total/total_iterations:.1f}%)")
print(f"  Failed: {n_failed_total}/{total_iterations} ({100*n_failed_total/total_iterations:.1f}%)")

# Save to NPZ file
npz_filename = output_dir + 'fastchem_grid_5d.npz'
np.savez_compressed(
    npz_filename,
    temperature=T,
    pressure=p,
    M_H=M_H_grid,
    C_O=C_O_grid,
    mixing_ratios=mixing_ratios_5d,
    mean_molecular_weight=mean_molecular_weight_5d,
    success_flags=success_flags_5d,
    species_names=species_names
)
print(f"\nSaved full 5D grid to: {npz_filename}")
print(f"  Shape: T({n_T}) x P({n_p}) x [M/H]({n_MH}) x C/O({n_CO}) x Species({n_species})")
print(f"  File size: {os.path.getsize(npz_filename) / 1024**2:.1f} MB")

# Save an additional log-space version (M/H is already dex/log10 by definition)
npz_log_filename = output_dir + 'fastchem_grid_5d_log10.npz'
np.savez_compressed(
    npz_log_filename,
    log10_temperature=safe_log10(T),
    log10_pressure=safe_log10(p),
    log10_M_H=np.array(M_H_grid, dtype=float),   # already in dex = log10(metallicity factor)
    log10_C_O=safe_log10(C_O_grid),
    log10_mixing_ratios=safe_log10(mixing_ratios_5d),
    log10_mean_molecular_weight=safe_log10(mean_molecular_weight_5d),
    success_flags=success_flags_5d,
    species_names=species_names
)
print(f"Saved log10 5D grid to: {npz_log_filename}")
print(f"  File size: {os.path.getsize(npz_log_filename) / 1024**2:.1f} MB")

# Also save selected species for easier access
plot_species_indices = []
for species in plot_species:
    # Find species index
    try:
        idx = species_names.index(species)
        plot_species_indices.append(idx)
    except ValueError:
        print(f"Warning: Species {species} not found in FastChem")

if plot_species_indices:
    selected_mixing_ratios = mixing_ratios_5d[:, :, :, :, plot_species_indices]
    np.savez_compressed(
        output_dir + 'fastchem_grid_selected.npz',
        temperature=T,
        pressure=p,
        M_H=M_H_grid,
        C_O=C_O_grid,
        mixing_ratios=selected_mixing_ratios,
        species_names=np.array(plot_species),
        species_labels=np.array(plot_species_labels)
    )
    print(f"Saved selected species to: {output_dir}fastchem_grid_selected.npz")

    np.savez_compressed(
        output_dir + 'fastchem_grid_selected_log10.npz',
        log10_temperature=safe_log10(T),
        log10_pressure=safe_log10(p),
        log10_M_H=np.array(M_H_grid, dtype=float),  # already dex
        log10_C_O=safe_log10(C_O_grid),
        log10_mixing_ratios=safe_log10(selected_mixing_ratios),
        species_names=np.array(plot_species),
        species_labels=np.array(plot_species_labels)
    )
    print(f"Saved selected species (log10) to: {output_dir}fastchem_grid_selected_log10.npz")

# Create verification plot for solar metallicity and C/O ~ 0.55
if n_success_total > 0 and plot_species_indices:
    print("\nGenerating verification plot...")

    # Find indices closest to solar values
    T_target = 1000.0
    MH_target = 0.0  # Solar metallicity
    CO_target = 0.55  # Solar-ish C/O

    T_idx = np.argmin(np.abs(T - T_target))
    MH_idx = np.argmin(np.abs(M_H_grid - MH_target))
    CO_idx = np.argmin(np.abs(C_O_grid - CO_target))

    T_actual = T[T_idx]
    MH_actual = M_H_grid[MH_idx]
    CO_actual = C_O_grid[CO_idx]

    plt.figure(figsize=(10, 6))

    # Plot at specified T, M/H, C/O
    for i, idx in enumerate(plot_species_indices):
        plt.plot(mixing_ratios_5d[T_idx, :, MH_idx, CO_idx, idx], p,
                label=plot_species_labels[i], marker='o')

    plt.xscale('log')
    plt.yscale('log')
    plt.gca().invert_yaxis()
    plt.xlabel("Mixing ratios")
    plt.ylabel("Pressure (bar)")
    plt.title(f"FastChem: T={T_actual:.1f}K, [M/H]={MH_actual:+.2f}, C/O={CO_actual:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir + 'fastchem_sample.png', dpi=150)
    print(f"Saved plot to: {output_dir}fastchem_sample.png")
    plt.show()

# Clean up temporary abundance files
print(f"\nCleaning up temporary files in {temp_dir}...")
shutil.rmtree(temp_dir)

print("\n" + "="*70)
print("Grid generation complete!")
print("="*70)
