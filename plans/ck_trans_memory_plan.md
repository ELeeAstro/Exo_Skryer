# Plan: cut per-particle memory in the `ck_mix: TRANS` transit path

## Goal

Reduce the GPU memory used per likelihood evaluation, so batched samplers
(BlackJAX NSS/SMC, NUTS) can run larger batches. The static opacity tables are
not the problem: they are stored once and shared by the whole batch (152.5 MB
for the HD209 test config). The memory that grows with the batch is the
per-particle working arrays, and the largest one in the TRANS path is built
up front even though it is only ever used one species at a time.

## What happens now

1. `compute_ck_opacity_perspecies` ([opacity_ck.py:286](../exo_skryer/opacity_ck.py#L286))
   interpolates every species at every layer with a `vmap`, and returns
   `line_perspecies` with shape **(nspec, nlay, nwl, ng), float64**.
   - HD209 config: 3 × 99 × 509 × 16 × 8 B = **19.3 MB per particle**,
     about 620 MB at `batch_size: 32`.
   - Full-grid run with 10 species (1154 wavelengths): about 146 MB per
     particle, about 4.7 GB at batch 32.
2. `_integrate_g_points_trans` ([RT_trans_1D_ck_trans.py:96](../exo_skryer/RT_trans_1D_ck_trans.py#L96))
   then loops over species with `lax.fori_loop` and reads only
   `sigma_perspecies[spec_idx]`, which is (nlay, nwl, ng), on each step.
3. With `f_cloud`, `compute_transit_depth_1d_ck_trans` calls
   `_integrate_g_points_trans` **twice** (cloudy column and clear column). The
   species transmission product `T_prod` doesn't depend on the cloud, so it
   is computed twice for no reason. Only the continuum term differs.

RORR/PRAS (the default, and the emission path) already loop over layers with
`fori_loop` in `compute_ck_opacity` ([opacity_ck.py:273](../exo_skryer/opacity_ck.py#L273)),
so they don't build the 4D array. This plan is about TRANS only. Step 6
extends the measurement to emission.

## The change

Interpolate each species **inside** the species loop, so the 4D array never
exists:

- Add `interp_ck_species(opac, state, spec_idx)` to `opacity_ck.py`. It
  returns one species' mass opacity on the layers, (nlay, nwl, ng), using the
  existing `_interp_species_log`. It indexes the table with
  `opac["ck_sigma_cube"][spec_idx]` and `opac["ck_log10_temperature_grids"][spec_idx]`,
  where `spec_idx` is a traced loop index (`lax.dynamic_index_in_dim`).
- Lazy mode: the opacity builder in `build_common.py` (around line 513) stores
  only what the RT needs to interpolate later, not `line_perspecies`:
  `vmr_perspecies`, and the layer log P / log T (already in `state`).
- `_integrate_g_points_trans` gets a callable `kappa_of(spec_idx)` in place of
  `sigma_perspecies`. In eager mode it is `lambda s: sigma_perspecies[s]`; in
  lazy mode it calls `interp_ck_species`.
- Compute `T_prod` **once** and use it for both the cloudy and the clear
  column: `T_total = exp(-tau_cont) * T_prod`, with only `tau_cont` depending
  on the cloud. This is a separate speed-up that applies in both modes.
- Gradients: `fori_loop` with a fixed trip count is turned into a `scan` under
  reverse-mode AD, which stores each step's (nlay, nwl, ng) values for the
  backward pass. That is the same 4D array again, just built one step at a
  time. Wrap the loop body in `jax.checkpoint` so only the carry `T_prod` is
  stored and the body is recomputed on the way back. Without this, lazy mode
  saves nothing under `grad`, i.e. for NUTS.

Keep both versions selectable with a config switch, default off until it is
verified:

```yaml
opac:
  ck_mix: TRANS
  ck_trans_lazy: false   # true = interpolate per species inside the RT loop
```

## Test plan

All tests run on CPU first (quick) with the HD209 config
(`experiments/HD209_Barstow_2020_trans_setup`, ck_mix TRANS, 3 species,
`f_cloud`). A copy goes in `experiments/_tmp_mem_test/` so the original isn't
touched.

### 1. Measurement script: `tests/mem_probe.py`

Builds the forward model and log-likelihood the same way `run_retrieval`
does, then for each batch size B in {1, 8, 32}:

```python
P = draw_prior(B)                                   # B parameter sets from the prior
f   = jax.jit(lambda P: jax.lax.map(fm, P, batch_size=B))
fg  = jax.jit(lambda P: jax.lax.map(jax.grad(loglike), P, batch_size=B))
for name, fn in (("fwd", f), ("grad", fg)):
    ma = fn.lower(P).compile().memory_analysis()
    print(name, B, ma.temp_size_in_bytes, ma.argument_size_in_bytes, ma.output_size_in_bytes)
```

- `temp_size_in_bytes` is the working memory for one batch, which is the
  quantity to reduce. `argument_size_in_bytes` includes the static tables and
  should not change.
- Also report the wall time of 5 calls after compiling, for each mode.
- Report the slope of `temp` against B: that's the per-particle memory, the
  number that decides the largest batch that fits.

### 2. Baseline

Run `mem_probe.py` with `ck_trans_lazy: false` before changing any code.
Save the table to `plans/mem_baseline.txt`.

Expected: `temp` grows by roughly 20 MB per particle (forward) and more under
`grad`.

### 3. Correctness

With `ck_trans_lazy: true`, for 20 prior draws:

- Transit depth: relative difference ≤ 1e-12 from eager mode (same
  operations, only reordered).
- Gradient of the log-likelihood: relative difference ≤ 1e-10.
- Separately, check that computing `T_prod` once gives the same `f_cloud`
  result as before (both modes).

### 4. Memory and speed with lazy mode on

Rerun `mem_probe.py`. Pass criteria:

| Quantity | Expected change |
|---|---|
| `fwd` temp slope per particle | down by ≈ 19 MB (the 4D array) |
| `grad` temp slope per particle | down by ≥ 19 MB, with `jax.checkpoint` on the body |
| `argument_size` | unchanged |
| Forward time | within ±20% (the interpolation moves into the loop; computing `T_prod` once should make up for it) |
| `grad` time | may rise (the body is recomputed); record it |

If `grad` time rises a lot, make the checkpoint optional
(`ck_trans_lazy_remat: true/false`) so NSS/SMC (forward only) never pay for it.

### 5. On the GPU cluster

`memory_analysis()` on CPU shows the scaling, but the GPU peak is what limits
the batch. On the cluster:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python tests/mem_probe.py --gpu
```

and read `jax.devices()[0].memory_stats()["peak_bytes_in_use"]` after one
batched call, for eager and lazy at the batch sizes you actually use. Then
find the largest `batch_size` that runs in each mode. That's the number that
matters for NSS/SMC.

### 6. Same measurement for emission

Point `mem_probe.py` at an emission config (`experiments/forward_model_em`
with `em_scheme: sc_ric` or `ad`) without code changes, to find which buffers
dominate there. RORR already loops over layers, so the candidates are the
mixed (nlay, nwl, ng) opacities and the RT working arrays. Use
`jax.jit(...).lower(...).compile().as_text()` to find the largest buffers if
the totals alone don't make it clear. That decides whether a similar change is
worth making for emission.

## Order of work

1. `tests/mem_probe.py`, plus the baseline numbers (no code changes).
2. Compute `T_prod` once for `f_cloud` (small, independent, both modes).
3. `interp_ck_species` and the `kappa_of` callable, behind `ck_trans_lazy`.
4. `jax.checkpoint` on the loop body.
5. Correctness checks (step 3 of the test plan), then memory and speed (step 4).
6. GPU run, then the emission measurement.

## Risks

- **Interpolation repeated for cloudy and clear columns:** avoided by
  computing `T_prod` once.
- **Recomputation under `grad`:** the cost of `jax.checkpoint` is extra time,
  measured in step 4.
- **XLA may already reuse buffers:** XLA may already schedule the eager
  version so the full 4D array isn't live at once. The baseline would then
  show a smaller slope than 19 MB, and the change is worth less. Step 2
  answers this before any code is written.
- **Contribution functions:** `contri_func` isn't supported with TRANS now,
  so nothing changes there.

## Results (CPU, `memory_analysis()`, float64, HD209 config, 2026-11-09)

Implemented as planned: `make_ck_species_kappa` / `ck_vmr_perspecies` in
`opacity_ck.py`, the `opac.ck_trans_lazy` switch in `build_common.py`, and in
`RT_trans_1D_ck_trans.py` a single species product shared by the cloudy and
clear columns, with `jax.checkpoint` on the loop body in lazy mode. Raw output
is in `plans/mem_*.txt`.

Working memory per particle (slope of `temp` against batch size), and time per
call at B = 32:

| Version | fwd MB/particle | grad MB/particle | grad temp at B = 32 | fwd call | grad call |
|---|---|---|---|---|---|
| Baseline (old code) | 40.3 | 283.9 | 9.10 GB | 237 ms | 1674 ms |
| Eager, product computed once | 40.3 | 264.5 | 8.48 GB | 208 ms | 1282 ms |
| Lazy, no `jax.checkpoint` | 14.6 | 252.1 | 8.07 GB | 179 ms | 1404 ms |
| **Lazy + `jax.checkpoint`** | **14.6** | **75.4** | **2.41 GB** | **183 ms** | **921 ms** |

7 species (CH4, CO, CO2, NH3 added), B = 1 and 8:

| Version | fwd MB/particle | grad MB/particle |
|---|---|---|
| Eager | 91.9 | 576.0 |
| Lazy + `jax.checkpoint` | 15.0 | 77.0 |

Correctness against the baseline, 20 prior draws: spectrum identical (max
relative difference 0), gradients within 1.2e-13.

Findings:

- Lazy mode cuts per-particle memory by 64% (forward) and 73% (gradient) with
  3 species, and by 84% / 87% with 7 species. In lazy mode the memory no longer
  depends on the number of species.
- `jax.checkpoint` is what makes the gradient saving happen: without it, lazy
  mode saves only 11% under `grad`.
- It is also faster: the gradient takes 45% less time and the forward pass 23%
  less at B = 32, since less memory is written and read.
- The static tables (`args`, 151 MB, or 350 MB with 7 species) are unchanged
  and are not the limiting factor.

Not done yet: the GPU peak-memory run (step 5 of the test plan) and the
emission measurement (step 6).

## Made default (2026-11-10)

Extra checks before switching, lazy vs eager, 20 prior draws each:

| Case | Spectrum | Gradient |
|---|---|---|
| PCHIP T interpolation | identical | ≤ 2e-15 of each parameter's gradient scale |
| No `f_cloud` | identical | 6e-15 |
| Refraction cutoff | identical | 1e-13 |
| 7 species + PCHIP + refraction, no `f_cloud` | identical | 2e-14 |

Lazy interpolation with `jax.checkpoint` is now the only TRANS path; the
`opac.ck_trans_lazy` switch and the precomputed `line_perspecies` input to the
RT are removed. `compute_ck_opacity_perspecies` stays as a diagnostic function
(built on `make_ck_species_kappa`), and
`experiments/WASP-17b_JWST_trans_setup/component_spectra.py` now isolates a
species by zeroing the other species' VMRs (a zero-VMR species transmits
exactly 1; matches computing the species alone to 5e-16).
