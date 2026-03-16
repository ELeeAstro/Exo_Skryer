.. _dace_tools:

*************************************
Building Opacity Tables from DACE
*************************************

This page describes how to use the scripts in ``opac_data/DACE_tools/`` to
convert per-molecule opacity binaries downloaded from the
`DACE Opacity Database <https://dace.unige.ch/opacityDatabase/>`__ into the
Zarr v3 format used by Exo_Skryer.

Two table types are supported:

* **Opacity sampling (OS)** — full high-resolution cross-section cube, used
  with ``physics.opac_line: os``.
* **Correlated-k (c-k)** — k-coefficient cube with Gauss-Legendre quadrature,
  used with ``physics.opac_line: ck``.

Both formats are produced by the same two scripts and registered in the YAML
config identically.

.. contents:: On this page
   :local:
   :depth: 2


Overview
========

.. code-block:: text

   DACE Binary Files  (.bin, Δν = 0.01 cm⁻¹)
           │
           │  input_os.txt / input_ck.txt
           ▼
   Gen_OS_table_R_zarr.py   →  <species>_R<R>.zarr  +  <species>_R<R>.zarr.zip
   Gen_ck_table_R_zarr.py   →  <species>_ck_R<R>.zarr  +  <species>_ck_R<R>.zarr.zip
           │
           │  wl_R<R>.txt / wl_ck_R<R>.txt  (shared wavelength grid)
           ▼
   opac_data/os/   or   opac_data/ck/
           │
           ▼
   retrieval_config.yaml  →  opac.line entries


Prerequisites
=============

1. **Download DACE binaries.**
   Visit https://dace.unige.ch/opacityDatabase/ and download the per-molecule
   ``.bin`` archive for each species you need.  Unpack into a directory
   (e.g. ``opac_data/opacities/H2O_EXOMOL/``).  Each ``.bin`` file represents
   one (T, P) point on a fixed wavenumber grid (Δν = 0.01 cm⁻¹).

2. **Install dependencies.**
   The scripts require ``numpy``, ``zarr`` (v3), and ``numcodecs``::

      pip install numpy "zarr>=3" numcodecs

3. **Locate the tools.**
   All scripts and input templates live in::

      opac_data/DACE_tools/
      ├── Gen_OS_table_R_zarr.py
      ├── Gen_ck_table_R_zarr.py
      ├── input_os.txt          ← template for OS tables
      └── input_ck.txt          ← template for c-k tables


DACE Input Data Format
======================

The DACE database distributes pre-computed line-by-line cross-sections as raw
IEEE-754 single-precision binary files (``.bin``), one file per (T, P) point.

**File naming:**

* Most species: ``Out_<wn_start>_<wn_end>_<T>_<P>.bin``
* Fe / FeII: ``<T>_<P>.bin``

**Fixed pressure grid** (34 levels, log₁₀ bar):

=============  ======================
Range          Values (log₁₀ bar)
=============  ======================
Low pressure   −8.00 → −0.33
High pressure   0.00 → +3.00
=============  ======================

The temperature list is *not* fixed — it is inferred automatically by globbing
the binary directory.

**Units:** cm² molecule⁻¹ for most species; cm² g⁻¹ for Fe and FeII.
The scripts convert to cm² molecule⁻¹ using
σ = σ_raw × mol_weight / N_A where needed.


Input File Format
=================

Both scripts read the same plain-text input format. Lines beginning with
``#`` are comments. After the 14-line header, the scripts process every
uncommented species line until the first blank line after the active block.

.. code-block:: text

   # wl start (micron)
   0.3
   # wl end (micron)
   30.0
   # resolution (R = λ/Δλ)
   250
   # wavelength output file
   wl_ck_R250.txt
   # output base name template (.zarr and .zarr.zip will be produced)
   H2O_ck_R250.txt
   # input form (currently unused, set to 1)
   1
   # species  mol_weight  T_min  T_max  wn_start  wn_end  binary_dir
   H2O 18.01528 50 6100 0 42000 ../opacities/H2O_EXOMOL
   CO  28.0101  50 6100 0 23000 ../opacities/CO_EXOMOL

Field descriptions:

``wl_start``, ``wl_end``
  Wavelength range in microns for the output table.

``R``
  Spectral resolving power (R = λ/Δλ). Typical values:

  * OS tables: 10 000 – 20 000
  * c-k tables: 250 – 1000

``wl_file``
  Name of the wavelength-grid output file (a plain-text column of wavelengths
  in microns). This file is shared across all species generated at the same R
  and must be referenced as ``opac.wl_master`` in the YAML config.

``out_name``
  Output-name template. If exactly one species is active, the scripts use this
  name directly. If multiple species are active, they auto-generate
  per-species names such as ``H2O_R20000.zarr`` or ``H2O_ck_R250.zarr``.

``species``
  Molecule identifier string (informational; stored as a Zarr attribute).

``mol_weight``
  Molecular weight in g mol⁻¹.

``T_min``, ``T_max``
  Informational temperature bounds; the actual temperature list is discovered
  from the binary files present in ``binary_dir``.

``wn_start``, ``wn_end``
  Wavenumber range (cm⁻¹) used to select which ``.bin`` files to open.
  Set ``wn_start = 0`` and ``wn_end`` to cover the full molecule range.

``binary_dir``
  Path (relative to the script, or absolute) to the directory of DACE ``.bin``
  files for this species.

To process multiple species in one run, add one active species line per
molecule in a contiguous block. Comment out lines with ``#`` to skip them, and
leave a blank line after the active block to stop processing.


Generating Opacity Sampling (OS) Tables
========================================

Run the OS script from within ``opac_data/DACE_tools/``::

   python Gen_OS_table_R_zarr.py

or explicitly::

   python Gen_OS_table_R_zarr.py --input input_os.txt

**What the script does:**

1. Builds a constant-R wavenumber grid from ``wl_start`` to ``wl_end``
   (Δν/ν = 1/R at each centre point).
2. For every (T, P) pair, reads the corresponding DACE binary and converts
   cross-sections to cm² molecule⁻¹.
3. Interpolates the native 0.01 cm⁻¹ data onto the output grid using
   ``np.interp`` in log-space, flooring values at 1×10⁻⁹⁹ before taking
   log₁₀.
4. Writes the (nT × nP × nλ) log₁₀ cross-section cube to Zarr.

**Output Zarr schema:**

.. list-table::
   :header-rows: 1
   :widths: 20 25 10 45

   * - Key
     - Shape
     - dtype
     - Description
   * - ``temperature``
     - (nT,)
     - float64
     - Temperature grid (K)
   * - ``pressure``
     - (nP,)
     - float64
     - Pressure grid (bar)
   * - ``wavelength``
     - (nλ,)
     - float64
     - Wavelength grid (µm), ascending
   * - ``cross_section``
     - (nT, nP, nλ)
     - float32
     - log₁₀ cross-section (cm² molecule⁻¹)
   * - attr ``molecule``
     - —
     - str
     - Species name

.. note::

   Exo_Skryer reads the linear ``temperature`` and ``pressure`` arrays from the
   Zarr store and computes ``log10(T)`` and ``log10(P)`` internally for
   interpolation. The files do not need to provide separate log-axis arrays.


Generating Correlated-k (c-k) Tables
======================================

Run the c-k script from within ``opac_data/DACE_tools/``::

   python Gen_ck_table_R_zarr.py

or explicitly::

   python Gen_ck_table_R_zarr.py --input input_ck.txt

**What the script does:**

1. Builds the same constant-R wavenumber grid and computes midpoint bin
   edges to robustly assign native high-resolution points to each spectral bin.
2. Constructs 16-point Gauss-Legendre quadrature nodes and weights, split
   8 + 8 at g = 0.9 (higher point density in the optically thick tail of the
   k-distribution).
3. For every (T, P) pair, reads the DACE binary and converts to
   cm² molecule⁻¹.
4. Within each spectral bin, sorts the cross-sections to form the cumulative
   distribution function (CDF) g(k), then samples log₁₀(k) at the
   Gauss-Legendre g-points by linear interpolation.
5. Writes the (nT × nP × nλ × ng) k-coefficient cube to Zarr.

**Output Zarr schema:**

.. list-table::
   :header-rows: 1
   :widths: 20 25 10 45

   * - Key
     - Shape
     - dtype
     - Description
   * - ``temperature``
     - (nT,)
     - float64
     - Temperature grid (K)
   * - ``pressure``
     - (nP,)
     - float64
     - Pressure grid (bar)
   * - ``wavelength``
     - (nλ,)
     - float64
     - Wavelength bin centres (µm), ascending
   * - ``g_points``
     - (ng,)
     - float64
     - Gauss-Legendre quadrature nodes in [0, 1]
   * - ``g_weights``
     - (ng,)
     - float64
     - Quadrature weights (normalised, sum to 1)
   * - ``cross_section``
     - (nT, nP, nλ, ng)
     - float32
     - log₁₀ k-coefficient (cm² molecule⁻¹)
   * - attr ``molecule``
     - —
     - str
     - Species name

.. note::

   As with OS tables, Exo_Skryer loads the linear ``temperature`` and
   ``pressure`` axes and derives the log grids internally in the opacity
   registries.


Output Files
============

Both scripts produce two output files from the same base name:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Description
   * - ``<name>.zarr/``
     - Zarr v3 directory store — fastest for local random access
   * - ``<name>.zarr.zip``
     - Zarr v3 zip store — single portable file for archiving / sharing

Compression: Blosc/lz4, level 1, byte-shuffle — optimised for read speed
over file size.

Both formats are read transparently by ``registry_ck.py`` and
``registry_line.py``.  If the YAML config points to a ``.zarr`` directory
that is absent, the registry automatically falls back to the ``.zarr.zip``
file in the same location.

Move the output files into the appropriate subdirectory. For multi-species
runs, move each generated species table plus the shared wavelength file::

   mv H2O_R20000.zarr     opac_data/os/
   mv H2O_R20000.zarr.zip opac_data/os/
   mv wl_R20000.txt       opac_data/os/

   mv H2O_ck_R250.zarr     opac_data/ck/
   mv H2O_ck_R250.zarr.zip opac_data/ck/
   mv wl_ck_R250.txt       opac_data/ck/


Registering Tables in the YAML Config
======================================

Reference the new files under ``opac`` in your ``retrieval_config.yaml``
(or ``forward_config.yaml``):

**Opacity sampling example:**

.. code-block:: yaml

   physics:
     opac_line: os
     opac_ray: os
     opac_cia: os

   opac:
     wl_master: ../../opac_data/os/wl_R20000.txt
     full_grid: false
     ck: false
     ck_mix: RORR

     line:
       - {species: H2O, path: ../../opac_data/os/H2O_R20000.zarr}
       - {species: CO,  path: ../../opac_data/os/CO_R20000.zarr}

**Correlated-k example:**

.. code-block:: yaml

   physics:
     opac_line: ck
     opac_ray: ck
     opac_cia: ck

   opac:
     wl_master: ../../opac_data/ck/wl_ck_R250.txt
     full_grid: false
     ck: true
     ck_mix: TRANS     # or RORR

     line:
       - {species: H2O, path: ../../opac_data/ck/H2O_ck_R250.zarr}
       - {species: CO,  path: ../../opac_data/ck/CO_ck_R250.zarr}

.. note::

   The ``wl_master`` file must be the one produced by the same script run
   that generated the line opacity tables — they share an identical wavelength
   grid.  Mixing wavelength files from different R values or different runs
   will cause a shape mismatch at runtime.


Supported Species Reference
============================

The table below lists the species present in the ``input_ck.txt`` and
``input_os.txt`` templates, along with their molecular weights and the
recommended wavenumber range.  Uncomment the relevant line in the input file
to generate a table for that species.

.. list-table::
   :header-rows: 1
   :widths: 10 15 12 12 12 12 27

   * - Species
     - Line list
     - Mol. wt.
     - T min (K)
     - T max (K)
     - wn range (cm⁻¹)
     - Notes
   * - H₂O
     - EXOMOL
     - 18.01528
     - 50
     - 6100
     - 0 – 42 000
     -
   * - CO
     - EXOMOL
     - 28.0101
     - 50
     - 6100
     - 0 – 23 000
     -
   * - CO₂
     - EXOMOL
     - 44.0095
     - 50
     - 2900
     - 0 – 20 000
     -
   * - CO₂
     - HITEMP
     - 44.0095
     - 50
     - 2900
     - 0 – 18 000
     - Alternative line list
   * - CH₄
     - EXOMOL
     - 16.0425
     - 50
     - 1900
     - 0 – 12 000
     -
   * - CH₄
     - HITEMP
     - 16.0425
     - 50
     - 2500
     - 0 – 14 000
     - Alternative line list
   * - NH₃
     - EXOMOL
     - 17.03052
     - 50
     - 1900
     - 0 – 20 000
     -
   * - H₂S
     - EXOMOL
     - 34.0809
     - 50
     - 2900
     - 0 – 35 000
     -
   * - SO₂
     - ExoAmes
     - 64.0638
     - 50
     - 1900
     - 0 – 8 000
     -
   * - SO₃
     - EXOMOL
     - 80.0632
     - 50
     - 1000
     - 0 – 5 000
     -
   * - SO
     - EXOMOL
     - 48.0644
     - 50
     - 4900
     - 0 – 45 000
     -
   * - OH
     - EXOMOL
     - 17.00734
     - 50
     - 6100
     - 0 – 50 000
     -
   * - OH
     - HITEMP
     - 17.00734
     - 200
     - 6100
     - 0 – 43 409
     - Alternative line list
   * - HCN
     - EXOMOL
     - 27.0253
     - 50
     - 3900
     - 0 – 18 000
     -
   * - C₂H₂
     - EXOMOL
     - 26.0373
     - 50
     - 2900
     - 0 – 10 000
     -
   * - C₂H₄
     - EXOMOL
     - 28.0532
     - 50
     - 1500
     - 0 – 8 000
     -
   * - Na
     - Kitzmann
     - 22.98977
     - 100
     - 4000
     - 0 – 36 000
     -
   * - K
     - Kitzmann
     - 39.09830
     - 100
     - 4000
     - 0 – 36 000
     -
   * - SiO
     - EXOMOL
     - 44.0849
     - 200
     - 6100
     - 0 – 66 481
     -
   * - CaH
     - EXOMOL
     - 41.0859
     - 50
     - 4500
     - 0 – 30 000
     -
   * - TiO
     - EXOMOL
     - 63.8664
     - 50
     - 4900
     - 0 – 30 000
     -
   * - VO
     - EXOMOL
     - 66.9409
     - 200
     - 6100
     - 0 – 35 000
     -
   * - FeH
     - EXOMOL
     - 56.8529
     - 200
     - 6100
     - 0 – 16 136
     -
   * - Fe
     - Kitzmann
     - 55.8450
     - 200
     - 6100
     - 0 – 0
     - Special: cm² g⁻¹ input; different filename scheme
   * - FeII
     - Kitzmann
     - 55.8450
     - 200
     - 6100
     - 0 – 0
     - Special: cm² g⁻¹ input; different filename scheme
   * - SH
     - EXOMOL
     - 33.0729
     - 200
     - 4900
     - 0 – 37 556
     -
   * - PO
     - EXOMOL
     - 46.97316
     - 50
     - 4500
     - 0 – 12 000
     -
   * - PH₃
     - EXOMOL
     - 33.99758
     - 50
     - 2900
     - 0 – 10 000
     -
   * - CS₂
     - HITRAN
     - 76.1407
     - 50
     - 2500
     - 0 – 6 500
     -
   * - OCS
     - EXOMOL
     - 60.0751
     - 50
     - 1900
     - 0 – 10 000
     -
   * - HF
     - EXOMOL
     - 20.00634
     - 200
     - 4900
     - 0 – 32 352
     -
   * - HCl
     - HITRAN
     - 36.4609
     - 50
     - 4500
     - 0 – 21 000
     -


Design Notes
============

**Constant-R wavelength grid**
  The grid is constructed so that Δν/ν = 1/R at each centre point (not
  constant spacing in wavenumber).  This is the standard astronomical
  convention and means the wavelength file can be shared across species
  with the same R.

**16-point Gauss-Legendre quadrature split at g = 0.9**
  Eight nodes are placed below g = 0.9 (the mostly transparent region,
  where fewer quadrature points are needed) and eight above (the opaque
  tail, where the integrand varies rapidly).  This split improves flux
  accuracy compared to a uniform 16-point scheme at the same cost.

**log₁₀ storage**
  All cross-sections are stored as log₁₀(σ) with a minimum floor of
  log₁₀(10⁻⁹⁹) = −99.  This avoids log(0) issues and reduces dynamic-range
  demands on float32 storage.

**Zarr v3 format**
  Blosc/lz4 compression with byte-shuffle is used as it prioritises
  read throughput over compression ratio, which is important for the
  forward model's random-access interpolation pattern.
