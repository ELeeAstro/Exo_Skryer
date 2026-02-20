******************************
YAML Configuration Reference
******************************

This page documents the structure of Exo_Skyrer's ``retrieval_config.yaml`` and
lists the available *string/enum* options for each configuration field.

Exo_Skryer loads YAML with :py:func:`yaml.safe_load` and converts nested dicts to
dot-accessible namespaces (see ``exo_skryer/read_yaml.py``). In YAML, prefer
using ``null`` for "no value" rather than the literal string ``"None"``.

High-level structure
====================

The configuration has these top-level blocks:

* ``data``: input paths (obs, stellar, NASA9)
* ``physics``: model scheme selectors (T-P, chemistry, RT, opac toggles)
* ``opac``: opacity registries and wavelength grid controls
* ``params``: retrieval parameter list (priors + fixed values)
* ``sampling``: sampler engine selection and hyperparameters
* ``runtime``: platform selection (cpu/gpu) and basic runtime knobs

Minimal skeleton
----------------

.. code-block:: yaml

   data:
     obs: path/to/obs.txt
     stellar: null
     nasa9: null

   physics:
     nlay: 99
     vert_Tp: Guillot
     vert_alt: p_ref
     vert_chem: constant_vmr_clr
     vert_mu: dynamic
     vert_cloud: None

     opac_line: ck
     opac_ray: None
     opac_cia: ck
     opac_cloud: None
     opac_special: None

     rt_scheme: transit_1d
     emission_mode: planet
     em_scheme: eaa
     contri_func: false
     refraction: None

   opac:
     wl_master: ../../opac_data/ck/wl_ck_R1000.txt
     full_grid: false
     ck: true
     ck_mix: RORR
     line: []
     ray: null
     cia: []
     special: null
     cloud: null

   params: []

   sampling:
     engine: dynesty
     dynesty: {nlive: 1000, dlogz: 0.1}

   runtime:
     platform: cpu
     threads: 1


data
====

``data.obs``
  Path to the observational data text file. Required.

  *Type*: string (absolute path, or relative to the experiment directory).

  *Format*: whitespace/CSV-like with at least 4 numeric columns::

    wl[um]  dwl[um]  y  dy

  Optional extra columns:

  * 5th numeric column: asymmetric error (``dy_minus``); the code uses
    ``max(dy_plus, dy_minus)``.
  * Next column: per-point response mode (e.g. ``boxcar`` or a custom bandpass
    name).
  * Next column: offset group label; if present, define ``offset_<group>`` in
    ``params`` (in **ppm**) to fit per-instrument/group offsets.

``data.stellar``
  Optional stellar spectrum file used for planet emission contrast (only needed
  when ``physics.rt_scheme: emission_1d`` and ``physics.emission_mode: planet``).

  *Type*: string path or ``null``.

  *Format*: at least two columns ``wl[um]  flux``. Interpolated in log-space.

``data.nasa9``
  Path to the NASA-9 thermo-coefficient folder, required for RateJAX chemical
  equilibrium (``physics.vert_chem: CE_rate_jax`` / ``ce_rate_jax`` aliases).

  *Type*: string path or ``null``.


physics
=======

Most ``physics`` fields are **scheme selectors** (strings). The selected scheme
determines which retrieval parameters (in ``params``) you must provide.

``physics.nlay``
  Number of atmospheric layers (integer). Default in code is 99.

``physics.vert_Tp`` (temperature profile)
  Selects the vertical temperature-pressure parameterization.

  *Supported values* (case-insensitive):

  * ``isothermal`` (alias: ``constant``)
  * ``Guillot``
  * ``Line``
  * ``Barstow``
  * ``Milne``
  * ``Modified_Milne`` (aliases: ``milne_2``, ``milne_modified``, ``modified_milne``)
  * ``picket_fence``
  * ``MandS`` (alias: ``mands``)

  *Typical required parameters* (see :doc:`vert_Tp` for full details):

  * ``isothermal``: ``T_iso``
  * ``Guillot``: ``T_int, T_eq, log_10_k_ir, log_10_gam_v, log_10_g, f_hem``
  * ``Line``: ``T_int, T_eq, f_hem, log_10_k_ir, log_10_g, log_10_gam_v1, log_10_gam_v2, alpha``
  * ``Barstow``: ``T_strat`` (plus other fixed constants in the implementation)
  * ``Milne``: ``T_int, log_10_k_ir, log_10_g``
  * ``Modified_Milne``: ``T_int, log_10_k_ir, log_10_g, T_ratio, log_10_p_t, beta``
  * ``picket_fence``: ``T_int, T_eq, log_10_k_ir, log_10_gam_v, log_10_R, Beta, log_10_g, f_hem``
  * ``MandS``: ``a1, a2, log_10_P1, log_10_P2, log_10_P3, T_ref``

``physics.vert_alt`` (altitude / hydrostatic integration)
  Selects the altitude/height calculation.

  *Supported values*:

  * ``hypsometric`` (aliases: ``constant``, ``constant_g``, ``fixed``)
  * ``hypsometric_variable_g`` (aliases: ``variable``, ``variable_g``)
  * ``hypsometric_variable_g_pref`` (alias: ``p_ref``)

  *Typical required parameters* (see :doc:`vert_alt`):

  * ``hypsometric``: ``log_10_g``
  * ``hypsometric_variable_g``: ``log_10_g, R_p``
  * ``hypsometric_variable_g_pref``: ``log_10_g, R_p, log_10_p_ref``

``physics.vert_chem`` (chemistry)
  Selects the chemistry profile model.

  *Supported values*:

  * ``constant_vmr`` (aliases: ``constant``)
  * ``constant_vmr_clr`` (aliases: ``constant_clr``, ``clr``)
  * ``CE_fastchem_jax`` (aliases: ``ce``, ``chemical_equilibrium``, ``fastchem_jax``) *(placeholder / not implemented)*
  * ``CE_rate_jax`` (aliases: ``rate_ce``, ``rate_jax``, ``ce_rate_jax``)
  * ``quench_approx`` (aliases: ``quench``)

  Notes:

  * For constant-VMR modes, the **trace species list is inferred** from the
    opacity configuration (line/ray/CIA/special). You then must provide either:
    ``log_10_f_<species>`` (constant VMR), or ``clr_<species>`` / ``log_10_f_<species>``
    (CLR constant VMR).
  * If CIA includes pairs with atomic H (e.g. ``H2-H`` or ``He-H``) **or** if
    H⁻ free-free is enabled, atomic hydrogen is required and you must include
    ``log_10_H_over_H2`` in ``params`` (constant-VMR modes derive ``H`` from the
    H2+He filler).
  * ``CE_rate_jax`` requires ``data.nasa9`` and parameters ``M/H`` and ``C/O``.
  * ``quench_approx`` uses RateJAX equilibrium plus quenching; requires at least
    ``M/H, C/O, Kzz, log_10_g``.

``physics.vert_mu`` (mean molecular weight)
  *Supported values*:

  * ``dynamic`` (aliases: ``variable``, ``vmr``): compute from VMR each layer
  * ``constant`` (aliases: ``fixed``): requires parameter ``mu``
  * ``auto``: use ``mu`` if present, else compute from VMR

``physics.vert_cloud`` (vertical cloud mass profile)
  *Supported values*:

  * ``None`` / ``none`` / ``off`` / ``no_cloud``: no clouds
  * ``exponential_decay_profile`` (aliases: ``exponential``, ``exp_decay``)
  * ``slab_profile`` (alias: ``slab``)
  * ``const_profile`` (aliases: ``const``, ``constant``)

  *Typical required parameters* (see :doc:`vert_cloud`):

  * ``exponential_decay_profile``: ``log_10_q_c, log_10_alpha_cld, log_10_p_base``
  * ``slab_profile``: ``log_10_q_c, log_10_p_top_slab, log_10_dp_slab``
  * ``const_profile``: ``log_10_q_c``

``physics.opac_line``
  Enables/disables line opacity and selects method.

  *Supported values*: ``lbl``, ``ck``, ``None``.

``physics.opac_ray``
  Rayleigh scattering toggle/mode.

  *Supported values*: ``lbl``, ``ck``, ``None``.

``physics.opac_cia``
  CIA toggle/mode.

  *Supported values*: ``lbl``, ``ck``, ``None``.

``physics.opac_cloud``
  Cloud opacity model selector.

  *Supported values*:

  * ``None`` (disable cloud opacity)
  * ``grey``
  * ``deck_and_powerlaw`` (alias: ``powerlaw``)
  * ``F18``
  * ``direct_nk`` (alias: ``nk``)
  * ``madt_rayleigh`` (aliases: ``madt-rayleigh``, ``mie_madt``)
  * ``lxmie`` (aliases: ``mie_full``, ``full_mie``)

  Cloud-opacity parameters depend on the chosen scheme (see :doc:`opacity_cloud`).

``physics.opac_special``
  Special opacity toggle (currently H⁻ bf/ff). This only controls whether the
  special opacity *kernel* runs; the special opacity tables themselves are
  enabled/disabled under ``opac.special``.

  *Supported values*: ``lbl``, ``ck``, ``on`` (all enable), or ``None``/``off``/``false``/``0``.

``physics.rt_scheme``
  Radiative transfer mode.

  *Supported values*: ``transit_1d`` or ``emission_1d``.

``physics.emission_mode``
  Only relevant for ``physics.rt_scheme: emission_1d``.

  *Supported values*:

  * ``planet``: compute planet/star contrast (uses ``data.stellar`` if provided)
  * ``brown_dwarf`` (aliases: ``browndwarf``, ``bd``): no stellar flux required

``physics.em_scheme``
  Only relevant for ``physics.rt_scheme: emission_1d``.

  *Supported values*: ``eaa`` / ``alpha_eaa`` / ``toon89`` / ``toon89_picaso``.

``physics.contri_func``
  Boolean. If true, the forward model also computes a contribution function and
  includes it in outputs when running the forward model directly.

``physics.refraction``
  Transmission-only refraction toggle. Only supported for ``rt_scheme: transit_1d``.

  *Supported values*:

  * ``None``/``off``/``false``/``0``: disable
  * ``cutoff`` / ``refractive_cutoff`` / ``refraction_cutoff``: apply refractive cutoff

  If enabled, you must include a delta parameter ``a_sm`` (semi-major axis in AU)
  and provide Rayleigh data (``opac.ray``).

``physics.cloud_dist``
  Static cloud size-distribution selector (not sampled).

  *Supported values*:

  * monodisperse: ``1`` / ``mono`` / ``monodisperse``
  * lognormal: ``2`` / ``lognormal`` / ``log-normal`` / ``log_normal`` / ``ln``


opac
====

The ``opac`` block controls the wavelength grid and which opacity tables are
loaded into registries.

``opac.wl_master``
  Master wavelength grid definition.

  *Supported values*:

  * string path to a wavelength file (``.txt`` or ``.npy``)
  * an explicit YAML array of wavelengths
  * omitted/``null``: fall back to ``obs['wl']``

``opac.full_grid``
  Boolean. If false (default), the code cuts the master grid to only wavelengths
  that fall within the observation bins. If true, it keeps the full master grid.

``opac.ck``
  Controls loading correlated-k tables. In most configs this is boolean:

  * ``true``: load correlated-k tables from ``opac.line`` (and use CK wavelength grid)
  * ``false``: load LBL tables from ``opac.line``

  Advanced: ``opac.ck`` can also be a list of ck-table entries instead of a bool
  (see ``exo_skryer/registry_ck.py``).

``opac.ck_mix``
  CK mixing rule used when ``opac.ck: true``.

  *Supported values* (case-insensitive): ``RORR`` (default), ``PRAS``, ``TRANS``.

  Notes:

  * ``TRANS`` is only supported for ``physics.rt_scheme: transit_1d``.

``opac.line``
  List of line opacity entries. Each entry is a mapping (flow-style shown)::

    - {species: H2O, path: ../../opac_data/lbl/H2O_R20000.npz}

  Fields:

  * ``species``: absorber name (string; must match the table contents)
  * ``path``: table file path (relative to experiment dir or absolute)

  File formats:

  * CK: ``.npz``, ``.h5``, ``.hdf5``
  * LBL: ``.npz``

``opac.ray``
  List of Rayleigh species entries. Each entry is typically::

    - {species: H2}

  Supported Rayleigh species names are defined in ``exo_skryer/registry_ray.py``.
  Common choices include: ``H2``, ``He``, ``H``, ``e-``, ``N2``, ``O2``, ``CO``,
  ``CO2``, ``CH4``, ``NH3``.

``opac.cia``
  List of CIA pair entries::

    - {species: H2-H2, path: ../../opac_data/cia/H2-H2_2011.npz}

  Notes:

  * Do not list ``H-`` here; H⁻ is handled under ``opac.special``.

``opac.special``
  Special opacity sources (currently only H⁻ continuum). Example::

    special:
      - {species: H-, bf: true, ff: false}

  Fields:

  * ``species``: currently must be ``H-``
  * ``bf``: enable bound-free table generation
  * ``ff``: enable free-free table generation

  If ``ff: true``, you must include both:

  * ``log_10_ne_over_ntot`` in ``params`` (proxy for electron fraction)
  * ``log_10_H_over_H2`` in ``params`` (to derive atomic H in constant-VMR modes)

  Backwards compatibility:

  * If ``opac.cia`` contains an entry with ``species: H-``, this enables H⁻ bound-free
    (and optional free-free via an ``ff`` flag), but this is deprecated.

``opac.cloud``
  Optional cloud refractive-index (n,k) table to cache on the model wavelength grid.
  Used by some cloud schemes (e.g. ``lxmie`` / ``madt_rayleigh``). Example::

    cloud:
      - {path: ../../opac_data/nk/silicate_nk.txt}

  The file format is documented in ``exo_skryer/registry_cloud.py``.


params
======

``params`` is a list of parameter definitions. Each element is typically written
in flow style::

  - { name: log_10_g, dist: uniform, low: 2.5, high: 3.5, transform: logit, init: 3.0 }

Common fields
-------------

``name`` (required)
  Parameter name (string). Names are passed directly into the forward model as
  keys in the ``params`` dict.

``dist`` (required)
  Prior / parameter type. Supported values depend on ``sampling.engine``:

  * Common across engines: ``uniform``, ``delta``
  * Some engines also support: ``normal``/``gaussian``, ``lognormal``,
    ``log_uniform``, ``beta``, ``gamma``

``low``, ``high``
  Bounds for uniform-like priors.

``mu``, ``sigma``
  Mean and stddev for normal/lognormal priors (engine-dependent).

``value``
  Used when ``dist: delta`` (fixed parameter). If ``value`` is not provided,
  some code paths fall back to ``init``.

``transform``
  Sampler-space transform hint. Supported values in this codebase are:
  ``identity`` and ``logit`` (and ``log`` for NUTS with ``log_uniform``).

  Notes:

  * Nested samplers treat transforms differently: e.g. Dynesty ignores
    ``transform`` for ``uniform`` because it already samples on a unit cube.
  * JAXNS / BlackJAX NS can use ``transform: logit`` to sample in an unconstrained
    latent space while preserving a uniform prior in physical space.

``init``
  Initial value (used for some sampler initialisation / warmup).

Special parameter conventions
-----------------------------

``log_10_f_<species>`` / ``clr_<species>``
  Abundance parameters used by constant chemistry schemes.

``log_10_H_over_H2``
  Only needed when atomic H must be present (CIA pairs with H, or H⁻ free-free).

``log_10_ne_over_ntot``
  Required for H⁻ free-free special opacity.

``offset_<group>``
  Optional additive offset applied to observed ``y`` for each offset group in the
  obs file. Units are **ppm** (internally divided by 1e6 to become fractional).

``c``
  Optional jitter parameter used in the Gaussian likelihood (log10 of sigma_jit).
  If omitted from YAML entirely, several samplers inject a silent default
  ``c = -99`` (effectively no jitter).


sampling
========

``sampling.engine``
  Selects the sampling engine.

  *Supported values*:

  * ``dynesty``
  * ``jaxns``
  * ``blackjax_ns``
  * ``ultranest``
  * ``pymultinest``
  * ``polychord``
  * ``nuts`` (MCMC; requires ``sampling.nuts.backend``)

``sampling.nuts.backend``
  Only used when ``sampling.engine: nuts``.

  *Supported values*: ``numpyro`` or ``blackjax``.

Engine blocks
-------------

The remaining keys under ``sampling`` are engine-specific configuration blocks.
Fields not listed here may be ignored.

``sampling.dynesty``
  Common fields: ``nlive, bound, sample, dlogz, maxiter, maxcall, bootstrap, enlarge,
  update_interval, dynamic, print_progress, seed``.

``sampling.jaxns``
  Common fields: ``max_samples, num_live_points, s, k, c, shell_fraction, difficult_model,
  parameter_estimation, gradient_guided, init_efficiency_threshold, verbose, posterior_samples, seed``.

  Optional nested block: ``sampling.jaxns.termination`` with fields
  ``ess, evidence_uncert, dlogZ, max_samples, max_num_likelihood_evaluations, rtol, atol``.

``sampling.blackjax_ns``
  Common fields: ``num_live_points, num_inner_steps, num_delete, dlogz_stop, seed,
  likelihood_batch_size, jit``.

``sampling.ultranest``
  Common fields: ``num_live_points, min_num_live_points, dlogz, max_iters, verbose, show_status``.

``sampling.pymultinest``
  Common fields: ``n_live_points, evidence_tolerance, sampling_efficiency, n_iter_before_update,
  null_log_evidence, max_modes, mode_tolerance, seed, verbose, resume, importance_nested_sampling,
  multimodal, const_efficiency_mode``.

``sampling.polychord``
  Common fields: ``nlive, num_repeats, num_repeats_mult, nprior, do_clustering, feedback,
  precision_criterion, max_ndead, boost_posterior, read_resume, write_resume, write_live,
  write_dead, write_stats, equals, compression_factor, seed``.

``sampling.nuts``
  Common fields: ``backend, warmup, draws, seed, chains``.
  Note: the current BlackJAX NUTS driver supports ``chains: 1`` only.


runtime
=======

``runtime.platform``
  *Supported values*: ``cpu`` or ``gpu``.

``runtime.cuda_visible_devices``
  Only used when ``runtime.platform: gpu``. String like ``\"0\"`` or ``\"0,1\"``.

``runtime.threads``
  Only used when ``runtime.platform: cpu``; passed to NumPyro host device count.

``runtime.tf_gpu_allocator``
  Optional string. If set, exported as ``TF_GPU_ALLOCATOR`` when on GPU.
