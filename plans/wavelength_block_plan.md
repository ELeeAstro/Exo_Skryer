# Plan: compute the spectrum in wavelength blocks

## Goal

Make the per-particle working memory of the forward model (and its gradient)
depend on a chosen **block size**, not on the number of wavelengths. Batched
samplers (BlackJAX NSS/SMC, pocoMC, NUTS) can then run large batches at high
resolution (OS R10000 and above, full JWST ranges, many species) on one GPU.

This is the "chunking" step left out of the solver work so far. It applies to
every RT path; OS emission is the first target because it is the largest case.

## Why it is needed

Measured on CPU (`memory_analysis()`, float64), Gliese 229B test model, OS
R10000, H2O/Na/K, 99 layers, 18 570 wavelengths after cutting to the data:

| Per particle | Forward | Gradient |
|---|---|---|
| EAA, after the solver restructuring | 89 MB | 753 MB |
| Toon89 | 369 MB | 1.17 GB |
| DDA | 148 MB | 0.90 GB |
| AD | 180–209 MB | 1.3–2.1 GB |
| sc_ric | 807 MB | 2.4 GB |
| Static tables (shared by the batch) | 393 MB | 393 MB |

One (nlay, nwl) float64 array is 14.7 MB here. The forward model builds many
of them before the RT starts:

- `opacities()` in [build_common.py](../exo_skryer/build_common.py) returns a dict
  of full arrays: line, Rayleigh, CIA, special, cloud extinction, cloud ssa,
  cloud g.
- The RT wrapper then forms k_tot, dtau, ssa, g and the level Planck function
  (nlev, nwl), and the solver's own working arrays come on top.
- Under `grad`, many of these are kept for the backward pass.

The solver changes only shrank the last item. Everything else scales with
nwl, so doubling the resolution or the wavelength range doubles the memory
per particle.

## What makes blocking possible

For OS and ck alike, wavelengths don't interact anywhere between the
atmosphere and the high-resolution spectrum:

- The atmosphere (T–p, chemistry, μ, altitude, densities, cloud profile)
  depends only on layer.
- Each opacity source is evaluated wavelength by wavelength from its tables.
- Each RT solver is independent per wavelength (and per g-point for ck).
- The only step that mixes wavelengths is the instrument binning/convolution
  (`apply_response_functions_cached`), and it acts on the assembled (nwl,)
  spectrum, which is small.

So the model can be written as

```
atm   = atmosphere(params)                          # layer-only, ~100 KB
D_b   = column(atm, params, wl_b, opac_b)           # opacities + RT for block b
D     = concat_b(D_b)                               # (nwl,), drop padding
D_bin = apply_response_functions_cached(D, bandpass)
```

with `column` run in a `lax.map` over blocks under `jax.checkpoint`. Only one
block's (nlay, nblock) arrays exist at a time, forward or backward.

## Design

### 1. The `column` function

`build_model.py` already runs `atmosphere -> opacities -> spec.rt ->
apply_response_functions_cached`. The change is to put the middle two steps
in one function of a block:

```python
def column(atm, params, wl_b, opac_b, stellar_b):
    state_b = {**atm, "wl": wl_b, "nwl": BLOCK, "stellar_flux": stellar_b}
    comps = opacities(state_b, params, opac_b)
    D_b, cf_b = spec.rt(state_b, params, comps, opac_b)
    return D_b, cf_b
```

`opacities` and `spec.rt` are **unchanged**: they get a state whose `wl` is the
block, and an `opac_b` dict with the **same keys** as now, holding each table's
slice for that block. Kernels that only use `state["wl"]` and wavelength-last
tables work as they are. That is most of them, but each one is checked in step 3.

### 2. Table layout

Tables are reshaped once at load time so that a block is one contiguous
leading-axis slice:

- Pad nwl up to a multiple of `BLOCK` by repeating the last wavelength (the
  padded points are dropped before binning).
- For every table with a wavelength axis, reshape `(..., nwl)` to
  `(nblock, ..., BLOCK)`.
- Tables without a wavelength axis (T/P grids, g weights, species indices,
  cloud n/k on their own grid if not on `wl`) are passed through unchanged.

A small registry in the opacity loaders declares each `opac` key's wavelength
axis (or none), e.g. `{"line_sigma_cube": -1, "cia_retained_sigma_cube": -1,
"ray_sigma_linear_table": -1, "line_log10_pressure_grid": None, ...}`. An
unknown key is an error at build time, so a new table can't silently be passed
whole into every block.

The blocked tables are then the `xs` of the block loop:

```python
def body(xs):
    wl_b, opac_b, stellar_b = xs
    return column(atm, params, wl_b, opac_b, stellar_b)

D_blocks, cf_blocks = lax.map(jax.checkpoint(body), (wl_blk, opac_blk, stellar_blk))
```

Scanning over a leading axis needs no `dynamic_slice`, and nothing has to be
differentiated with respect to the tables, so the backward pass stores only
`atm` and `params` per block.

The static table memory is unchanged (same data, different shape). Where the
loaders currently keep a wavelength-last copy for other uses (diagnostics,
plotting), the blocked form replaces it, so tables aren't stored twice.

### 3. Kernel audit (things that might span wavelengths)

Every opacity and RT kernel must be checked for anything that isn't
per-wavelength. Known candidates:

| Item | Where | What to check |
|---|---|---|
| `*_master_wavelength` arrays | CIA, Rayleigh, H⁻ | Whether they are compared against or interpolated onto `state["wl"]`; if so, block them too |
| Cloud optics (Mie/LX-MIE, MADT, F18, power law) | `opacity_cloud.py` | Per-wavelength in `wl`; n/k data on `wl` must be blocked |
| Refraction cutoff | `refraction.py` | Whether the mask uses a reduction over wavelength (e.g. a reference wavelength or a minimum) |
| Stellar flux | runtime arg, (nwl,) | Block it with the tables |
| `jnp.zeros((nlay, nwl))` shapes from `state["nwl"]` | many kernels | Must be the block size (it is static, so this works once `state_b["nwl"] = BLOCK`) |
| `contri_func` | RT kernels | Per block (nlay, BLOCK), assembled to (nlay, nwl) as output only; normalisation is per wavelength already |
| 1.5D transit | `RT_trans_1_5D_*` | Limb columns per wavelength; same treatment |
| Albedo | `RT_alb_1D.py` | Check its scattering/geometry for any wavelength coupling |

Anything that does span wavelengths either moves to `atmosphere()` (if it is
wavelength-independent in effect) or after the assembly (if it acts on the
spectrum).

### 4. ck paths

- **RORR/PRAS:** the mixed (nlay, nwl, ng) opacity becomes (nlay, BLOCK, ng)
  per block. The g-point loop inside the RT (now checkpointed, see
  `RT_em_1D_ck.py`) stays as is.
- **TRANS:** the lazy per-species loop (already the default) runs inside each
  block. Per-particle memory becomes independent of nwl, ng and the number of
  species.

### 5. Precision

Line cross sections are already stored as float32 on the device
(`registry_line.py`). Check CIA, Rayleigh and ck tables, and store them as
float32 where they aren't, interpolating in float32 and converting to float64
for the exponentials and the RT. Accuracy check: spectra against float64
tables over 20 prior draws.

This is independent of blocking and can be done first or skipped.

### 6. Configuration

```yaml
opac:
  wl_block: 2048     # wavelengths per block; 0 = whole grid in one block (current behaviour)
```

`wl_block: 0` must reproduce the current model exactly, as a regression path
and for diagnostics.

## Expected effect

Per-particle working memory ≈ (fixed layer-only part) + (current
wavelength-dependent part) × BLOCK / nwl. For OS R10000 with BLOCK = 2048,
that is about 1/9 of the current wavelength-dependent memory, and it stays the
same if the resolution, range or number of species grows. These are estimates;
step 4 of the test plan measures them.

Time:

- **Forward, CPU:** about the same; possibly faster from better cache use.
- **Gradient:** about one extra forward pass from the per-block checkpoint
  (1.3–2× as for TRANS, where it ended up faster overall because less memory
  was moved).
- **GPU:** the work in flight is batch × BLOCK × nlay. A smaller block needs a
  larger batch to keep the GPU busy, so the choice is a split of the same
  total, not a loss. The right default comes from the GPU measurement.

## Order of work

Each step ends with the regression check (spectra and gradients against the
saved references) before moving on.

1. **References and measurement.** Save spectra and gradients (`mem_probe.py
   --check`) for a matrix of configs: OS emission (EAA, Toon89, sc_ric), ck
   emission RORR (with and without `f_cloud`), OS transit, ck TRANS transit
   (with refraction), 1.5D transit, albedo if used. Record memory and time for
   each.
2. **Kernel audit** (design step 3). List each kernel's wavelength-dependent
   inputs and fix any wavelength-spanning code. No behaviour change yet.
3. **Wavelength-axis registry and table blocking** in the loaders, with
   `wl_block: 0` giving one block of the full grid. Must match the references
   exactly.
4. **`column` and the block loop** in `build_model.py` (and
   `build_model_1_5D.py`), with `jax.checkpoint`. Check BLOCK = nwl, a divisor,
   a non-divisor (tests the padding) and a small value such as 64.
5. **Diagnostics:** `return_highres`, `contri_func`, `component_spectra.py` and
   the plotting scripts in `experiments/*/` run on the assembled outputs. Check
   each still works.
6. **Memory and time on CPU** for the whole config matrix, blocks
   {0, 4096, 2048, 512}.
7. **GPU run** on the cluster: peak memory
   (`jax.devices()[0].memory_stats()["peak_bytes_in_use"]`) and time per
   likelihood at the batch sizes used for NSS/SMC. Find the largest batch that
   fits for each block size and the throughput (likelihoods per second). Set
   the default `wl_block` from this.
8. **Float32 tables** (design step 5), if still wanted after step 7.

## Test plan

| Check | Pass criterion |
|---|---|
| `wl_block: 0` vs current code | Identical spectra; gradients ≤ 1e-13 relative |
| Any block size vs `wl_block: 0` | Spectra ≤ 1e-13 relative (same operations, only reordered); gradients ≤ 1e-12 |
| Padding (non-divisor block) | As above; binned spectrum unchanged |
| `contri_func` | Identical to unblocked |
| Per-particle memory slope | Falls roughly in proportion to BLOCK / nwl in the wavelength-dependent part |
| `check_model_traces` on every `experiments/*/retrieval_config.yaml` | Builds and traces |
| GPU | Larger batch fits at equal or better likelihoods per second |

## Risks

- **Hidden coupling between wavelengths** in a kernel (a reduction over `wl`,
  a reference wavelength, normalisation). The audit in step 2 and the block-size
  tests in step 4 are there to catch this.
- **Interface size:** the dict of full opacity arrays is used by contribution
  functions, `component_spectra.py` and several plotting scripts. Keeping the
  same keys in `opac_b` and assembling outputs keeps them working, and
  `wl_block: 0` is the fallback.
- **GPU occupancy** with small blocks and small batches; addressed by step 7
  and by choosing the default from measurement.
- **Compile time:** the block body is compiled once, so it should not grow, and
  may shrink.
- **Loader memory:** reshaping the tables must not keep a second copy on the
  device.

## Alternatives considered

- **Chunking only the RT:** leaves the full opacity arrays, which are a large
  part of the memory, so it gains much less.
- **`vmap` over blocks:** vectorises all blocks at once, so the memory is the
  same as now.
- **Thinning the OS grid** (every n-th point): valid opacity sampling if the
  data resolution allows, and complementary to blocking. It is a science choice
  per dataset, not a code change.
- **Custom VJPs** for the interpolation and simple solvers: cuts gradient
  memory further, but only matters for NUTS; can follow later.

## Decisions for you

1. The default `wl_block` (proposed: decide from the GPU run, with 0 as the
   default until then).
2. Whether `wl_block: 0` stays as a supported path long-term or only for
   testing.
3. Whether float32 interpolation of the non-line tables is acceptable if the
   accuracy check passes.
4. Which path to do first after OS emission: ck TRANS transit (the NSS/SMC runs
   that hit memory limits) or ck emission.

## Implemented for emission (2026-11-10)

`opac.wl_block` (default 0 = whole grid, as before) and
`opac.interp_precision` (default float64) are implemented for
`rt_scheme: emission_1d`, OS and ck. Other RT schemes raise
`NotImplementedError` if `wl_block > 0`.

How it differs from the design above:

- No reshaped copy of the tables and no padding. Each block reads its slice
  with `lax.dynamic_slice` inside the block loop
  ([wl_blocks.py](../exo_skryer/wl_blocks.py)). When the block size doesn't
  divide nwl, the last block is shifted back to end at nwl and its overlap
  with the previous block is dropped on assembly.
- `opacities` and `spec.rt` are unchanged: [build_model.py](../exo_skryer/build_model.py)
  calls them with a block's `state` (`wl`, `nwl`, `stellar_flux`, cloud n/k)
  and `opac` slices.
- Every runtime table is registered with its wavelength axis in
  `wl_blocks.WAVELENGTH_AXIS` (or in `NO_WAVELENGTH_AXIS`); an unregistered
  table is an error at build time.

Kernel audit for emission: none of the line, CIA, Rayleigh, H⁻ or cloud
kernels, nor the OS/ck emission wrappers, couples wavelengths (the
`*_master_wavelength` arrays are only shape checks; cloud node splines are
evaluated per wavelength from parameters).

### Results (CPU, float64, `memory_analysis()`; raw output in `mem_wl_blocks.txt`)

OS R10000 Gliese 229B test model (18 570 wavelengths, EAA), per particle,
and call time at B = 8:

| `wl_block` | fwd MB | grad MB | fwd call | grad call | spectrum vs unblocked | gradient |
|---|---|---|---|---|---|---|
| 0 | 89.0 | 758 | 583 ms | 1675 ms | — | — |
| 4096 | 13.4 | 161 | 675 ms | 2281 ms | identical | 2e-13 |
| 2048 | 6.8 | 81 | 643 ms | 2209 ms | identical | 3e-13 |
| 1000 (non-divisor) | 3.4 | 40 | 597 ms | 2335 ms | identical | 3e-13 |
| 64 | 1.4 | 2.7 | 647 ms | 1988 ms | identical | 3e-13 |

ck R250 (464 wavelengths, F18 cloud with `f_cloud`, EAA): forward 14.7 →
2.5 MB and gradient 1143 → 243 MB per particle with `wl_block: 100`;
spectrum identical, gradients 9e-11. Forward call 1.84 → 2.71 s at B = 8.

High-resolution spectrum, binned spectrum and contribution functions with
blocks: identical to unblocked (OS `wl_block: 1000`, ck `wl_block: 100`).

`interp_precision: float32` vs float64:

| Model | Spectrum | Gradient | grad MB per particle |
|---|---|---|---|
| OS | 7e-6 | 4e-5 | 758 → 685 |
| ck | 6e-5 | 1.4e-3 | 1143 → 1107 |

It works and the spectral error is far below data uncertainties, but on CPU
it saves little memory and no time. It is left off by default; it is worth
timing on the GPU.

Still to do:

- GPU measurements (step 7) to choose a default `wl_block`.
- The ck gradient is still 243 MB per particle at `wl_block: 100`, dominated
  by the RORR mixing in `compute_ck_opacity` (its layer loop isn't
  checkpointed).
- Transit (OS, ck, TRANS, 1.5D, refraction cutoff audit).
