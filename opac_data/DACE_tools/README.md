# DACE_tools

Scripts for converting per-molecule opacity binaries downloaded from the
[DACE Opacity Database](https://dace.unige.ch/opacityDatabase/) into the
Zarr v3 format used by Exo_Skryer.

---

## Input data format

The DACE database distributes pre-computed line-by-line (LBL) cross-sections
as raw IEEE-754 single-precision binary files (`.bin`).
Each file contains cross-sections at a single (T, P) point on a fixed
wavenumber grid with spacing Δν = 0.01 cm⁻¹.

File naming convention:

- **Most species:** `Out_<wn_start>_<wn_end>_<T>_<P>.bin`
- **Fe / FeII:** `<T>_<P>.bin`

The pressure grid is fixed across all DACE tables (34 levels):

| Range | Values (log₁₀ bar) |
|---|---|
| Low pressure | −8.00 → −0.33 |
| High pressure | 0.00 → +3.00 |

Temperatures available depend on the molecule and are inferred automatically
from the files present in the binary directory.

Cross-sections are in units of cm² molecule⁻¹ (most species) or cm² g⁻¹
(Fe, FeII). The scripts convert to cm² molecule⁻¹ using
`σ = σ_raw × mol_weight / Avogadro` where needed.

---

## Input files

Both scripts share the same plain-text input file structure
(`input_OS.txt` / `input_ck.txt`), with comments starting with `#`:

```
# wl start (micron)
0.3
# wl end (micron)
30.0
# resolution (R = λ/Δλ)
250
# wavelength output file
wl_ck_R250.txt
# output base name (.zarr and .zarr.zip will be produced)
H2O_ck_R250.txt
# input form (currently unused, set to 1)
1
# species  mol_weight  T_min  T_max  wn_start  wn_end  binary_dir
H2O 18.01528 50 6100 0 42000 ../opacities/H2O_EXOMOL
```

The `wn_start` / `wn_end` values select which DACE binary files to open.
Temperature bounds (`T_min`, `T_max`) are informational; the actual
temperature list is discovered by globbing the binary directory.

---

## Gen_OS_table_R_zarr.py — Opacity Sampling (OS)

Produces a high-resolution opacity-sampling (OS) table on a constant-R
wavelength grid directly from the DACE high-resolution binaries.

**Processing steps:**

1. Builds a constant-resolving-power wavenumber grid from `wl_start` to
   `wl_end` at resolution R (i.e. Δν/ν = 1/R at each centre point).
2. For every (T, P) pair, reads the corresponding DACE binary and converts
   cross-sections to cm² molecule⁻¹.
3. Interpolates the native high-resolution data (0.01 cm⁻¹ spacing) onto the
   output wavenumber grid using `np.interp`, flooring values at 1×10⁻⁹⁹ before
   taking log₁₀.
4. Writes the full (nT × nP × nλ) log₁₀ cross-section cube to Zarr.

**Output Zarr schema:**

| Key | Shape | dtype | Description |
|---|---|---|---|
| `temperature` | (nT,) | float64 | Temperature grid (K) |
| `pressure` | (nP,) | float64 | Pressure grid (bar) |
| `wavelength` | (nλ,) | float64 | Wavelength grid (µm), ascending |
| `cross_section` | (nT, nP, nλ) | float32 | log₁₀ cross-section (cm² molecule⁻¹) |
| attr `molecule` | — | str | Species name |

---

## Gen_ck_table_R_zarr.py — Correlated-k (c-k)

Produces a correlated-k table on the same constant-R wavelength grid,
integrating the native opacity distribution within each spectral bin using
Gauss-Legendre quadrature.

**Processing steps:**

1. Builds the same constant-R wavenumber grid and computes midpoint bin
   edges for robust assignment of native high-resolution points to each spectral bin.
2. Constructs 16-point Gauss-Legendre quadrature nodes and weights, split
   8 + 8 at g = 0.9 (higher density of points in the opaque tail of the
   distribution).
3. For every (T, P) pair, reads the DACE binary and converts to
   cm² molecule⁻¹.
4. Within each spectral bin, sorts the cross-sections to form the cumulative
   distribution function (CDF) g(k), then samples log₁₀ k at the
   Gauss-Legendre g-points by linear interpolation.
5. Writes the full (nT × nP × nλ × ng) k-coefficient cube to Zarr.

**Output Zarr schema:**

| Key | Shape | dtype | Description |
|---|---|---|---|
| `temperature` | (nT,) | float64 | Temperature grid (K) |
| `pressure` | (nP,) | float64 | Pressure grid (bar) |
| `wavelength` | (nλ,) | float64 | Wavelength bin centres (µm), ascending |
| `g_points` | (ng,) | float64 | Gauss-Legendre quadrature nodes in [0, 1] |
| `g_weights` | (ng,) | float64 | Quadrature weights (normalised, sum to 1) |
| `cross_section` | (nT, nP, nλ, ng) | float32 | log₁₀ k-coefficient (cm² molecule⁻¹) |
| attr `molecule` | — | str | Species name |

---

## Output files

Both scripts produce two output files from the same base name:

| File | Description |
|---|---|
| `<name>.zarr/` | Zarr v3 directory store — fast local random access |
| `<name>.zarr.zip` | Zarr v3 zip store — single portable file for archiving / sharing |

Compression: Blosc/lz4, level 1, byte-shuffle — optimised for read speed
over file size.

Both formats are read transparently by `registry_ck.py` and `registry_line.py`.
If only `.zarr` is specified in the Exo_Skryer YAML config but the directory
is absent, the registries will automatically fall back to the `.zarr.zip` file.
