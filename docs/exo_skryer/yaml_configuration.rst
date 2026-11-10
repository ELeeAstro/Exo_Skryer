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

* ``data``: input paths (obs, obs_east/obs_west, stellar, NASA9)
* ``physics``: model scheme selectors (T-P, chemistry, RT, opac toggles)
* ``opac``: opacity registries and wavelength grid controls
* ``params``: retrieval parameter list (priors + fixed values)
* ``easychem_jax``: explicit species/elements/solver config for EasyChem chemistry (optional)
* ``fastchem_grid_jax``: FastChem 5D grid interpolation config (optional)
* ``atmodeller``: explicit species-network/solver config for Atmodeller chemistry (optional)
* ``quench_approx``: quenched-chemistry species config (optional)
* ``sampling``: sampler engine selection and hyperparameters
* ``runtime``: platform selection (CPU/CUDA/ROCm) and basic runtime knobs

Minimal skeleton
----------------

.. code-block:: yaml

   data:
     obs: path/to/obs.txt
     obs_east: null
     obs_west: null
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
     cloud_dist: mono

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
     T_interp: linear
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

``data.obs_east`` / ``data.obs_west``
  Separate east/west limb observation files for ``physics.rt_scheme: transit_1_5d``.
  Both must be set for 1.5D transit retrievals.

  *Type*: string path or ``null``.

  *Format*: same as ``data.obs``.

``data.stellar``
  Optional stellar spectrum file used for planet emission contrast (only needed
  when ``physics.rt_scheme: emission_1d`` and ``physics.emission_mode: planet``).

  *Type*: string path or ``null``.

  *Format*: at least two columns ``wl[um]  flux``. Interpolated in log-space.

``data.nasa9``
  Path to the NASA-9 thermo-coefficient folder, required for RateJAX chemical
  equilibrium (``physics.vert_chem: CE_rate_jax`` / ``ce_rate_jax`` aliases) and
  for the EasyChem backend (``physics.vert_chem: easychem_jax``).

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
  * ``file``: temperatures from ``profile.tp_file`` (see `profile`_). The file's
    pressures become the model layers, so ``physics.nlay``, ``p_bot`` and
    ``p_top`` are not used.

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
  * ``ce`` (aliases: ``chemical_equilibrium``, ``ce_fastchem_jax``, ``fastchem_jax``)
  * ``fastchem_grid_jax`` (aliases: ``ce_fastchem_grid``, ``fastchem_ce_grid``)
  * ``rate_ce`` (aliases: ``rate_jax``, ``ce_rate_jax``)
  * ``easychem_jax`` (alias: ``easychem``)
  * ``quench_approx`` (aliases: ``quench``)
  * ``atmodeller``
  * ``file``: VMRs from ``profile.vmr_file`` (see `profile`_)

  Notes:

  * For constant-VMR modes, the **trace species list is inferred** from the
    opacity configuration (line/ray/CIA/special). You then must provide either:
    ``log_10_f_<species>`` (constant VMR), or ``clr_<species>`` / ``log_10_f_<species>``
    (CLR constant VMR).
  * If CIA includes pairs with atomic H (e.g. ``H2-H`` or ``He-H``) **or** if
    H⁻ free-free is enabled, atomic hydrogen is required and you must include
    ``log_10_H_over_H2`` in ``params`` (constant-VMR modes derive ``H`` from the
    H2+He filler).
  * ``ce`` uses the FastChem-grid backend and requires parameters ``M_to_H`` and
    ``C_to_O`` plus a top-level ``fastchem_grid_jax`` block with required NPZ
    ``grid_path``.
  * ``fastchem_grid_jax`` requires parameters ``M_to_H`` and ``C_to_O`` and a top-level
    ``fastchem_grid_jax`` block with required NPZ ``grid_path``.
  * ``rate_ce`` requires ``data.nasa9`` and parameters ``M_to_H`` and ``C_to_O``.
  * ``easychem_jax`` requires ``data.nasa9``, parameters ``M_to_H`` and ``C_to_O``,
    and a top-level ``easychem_jax`` config block with explicit species list.
  * ``atmodeller`` requires parameters ``M_to_H`` and ``C_to_O`` and a top-level
    ``atmodeller`` config block.
  * ``quench_approx`` uses the FastChem-grid backend plus quenching; requires at
    least ``M_to_H, C_to_O, log_10_Kzz, log_10_g`` plus the
    ``fastchem_grid_jax`` and ``quench_approx`` config blocks.
  * For ``physics.rt_scheme: transit_1_5d``, these required names may be supplied
    as explicit 1.5D variants such as ``M_to_H_joint`` or paired
    ``M_to_H_east``/``M_to_H_west``.

``physics.vert_mu`` (mean molecular weight)
  *Supported values*:

  * ``dynamic`` (aliases: ``variable``, ``vmr``): compute from VMR each layer
  * ``constant`` (aliases: ``fixed``): requires parameter ``mu``
  * ``auto``: use ``mu`` if present, else compute from VMR
  * ``file``: the ``mu_amu`` column of ``profile.tp_file`` (needs ``vert_Tp: file``)

``physics.vert_cloud`` (vertical cloud mass profile)
  *Supported values*:

  * ``None`` / ``none`` / ``off`` / ``no_cloud``: no clouds
  * ``exponential_decay_profile`` (aliases: ``exponential``, ``exp_decay``)
  * ``slab_profile`` (alias: ``slab``)
  * ``const_profile`` (aliases: ``const``, ``constant``)
  * ``file``: total condensate q_c from ``profile.cloud_file`` (see `profile`_)

  *Typical required parameters* (see :doc:`vert_cloud`):

  * ``exponential_decay_profile``: ``log_10_q_c, log_10_alpha_cld, log_10_p_base``
  * ``slab_profile``: ``log_10_q_c, log_10_p_top_slab, log_10_dp_slab``
  * ``const_profile``: ``log_10_q_c``

``physics.opac_line``
  Enables/disables line opacity and selects method.

  *Supported values*: ``os``, ``ck``, ``None``.

``physics.opac_ray``
  Rayleigh scattering toggle/mode.

  *Supported values*: ``os``, ``ck``, ``None``.

``physics.opac_cia``
  CIA toggle/mode.

  *Supported values*: ``os``, ``ck``, ``None``.

``physics.opac_cloud``
  Cloud opacity model selector.

  *Supported values*:

  * ``None`` (disable cloud opacity)
  * ``file``: per-species clouds from ``profile.cloud_file`` (see `profile`_)
  * ``grey_const``: grey opacity in every layer
  * ``grey_profile``: grey opacity only where ``vert_cloud`` gives ``q_c_lay > 0`` (alias: ``grey_slab``)
  * ``deck_and_powerlaw`` (alias: ``powerlaw``)
  * ``F18``
  * ``direct_nk`` (alias: ``nk``)
  * ``nk_f18_blend`` (alias: ``nk_f18``)
  * ``f18_skew`` (alias: ``f18_skewnormal``)
  * ``madt_rayleigh`` (aliases: ``madt-rayleigh``, ``mie_madt``)
  * ``lxmie`` (aliases: ``mie_full``, ``full_mie``)

  Cloud-opacity parameters depend on the chosen scheme (see :doc:`opacity_cloud`).

``physics.opac_special``
  Special opacity toggle (currently H⁻ bf/ff). This only controls whether the
  special opacity *kernel* runs; the special opacity tables themselves are
  enabled/disabled under ``opac.special``.

  *Supported values*: ``os``, ``ck``, ``on`` (all enable), or ``None``/``off``/``false``/``0``.

``physics.rt_scheme``
  Radiative transfer mode.

  *Supported values*: ``transit_1d``, ``transit_1_5d``, or ``emission_1d``.

  Notes for ``transit_1_5d``:

  * Use ``data.obs_east`` and ``data.obs_west`` to fit separate limb spectra.
    If only ``data.obs`` is supplied, the model compares the hemispheric mean
    ``0.5 * east + 0.5 * west`` to the combined observation.
  * Every YAML parameter name must end in ``_joint``, ``_east``, or ``_west``.
  * Use ``_joint`` for shared quantities; use paired ``_east``/``_west`` entries
    for limb-specific quantities.
  * ``M_p``-based radius/gravity inference is not supported; provide explicit
    ``log_10_g_east``/``log_10_g_west`` or a shared ``log_10_g_joint``.
  * ``physics.contri_func: true`` is not supported.

``physics.emission_mode``
  Only relevant for ``physics.rt_scheme: emission_1d``.

  *Supported values*:

  * ``planet``: compute planet/star contrast (uses ``data.stellar`` if provided)
  * ``brown_dwarf`` (aliases: ``browndwarf``, ``bd``): no stellar flux required

``physics.em_scheme``
  Only relevant for ``physics.rt_scheme: emission_1d``.

  *Supported values*:

  * ``eaa`` / ``alpha_eaa`` (default): extended absorption approximation. Fastest;
    approximate scattering, less accurate at moderate-high single scattering albedo.
  * ``toon89`` / ``toon89_picaso``: Toon et al. (1989) two-stream method.
  * ``dda`` / ``4dda``: four-stream discrete ordinates adding method. Most accurate
    for multiple scattering (e.g. Mie clouds), at roughly Toon89 cost.

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

  The boundary is interpolated between layer midpoints, so the ray holding it
  is partly opaque; the cutoff moves smoothly with the atmosphere (finite
  gradients for gradient-based samplers).

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
  * ``false``: load OS (opacity-sampling) tables from ``opac.line``

  Advanced: ``opac.ck`` can also be a list of ck-table entries instead of a bool
  (see ``exo_skryer/registry_ck.py``).

``opac.ck_mix``
  CK mixing rule used when ``opac.ck: true``.

  *Supported values* (case-insensitive): ``RORR`` (default), ``PRAS``, ``TRANS``.

  Notes:

  * ``TRANS`` is only supported for transit radiative transfer
    (``transit_1d`` and ``transit_1_5d``).
  * ``transit_1_5d`` supports ``RORR`` and ``TRANS`` when using CK opacity, but
    does not currently support ``PRAS``.

``opac.T_interp``
  Interpolation of the line (OS and CK), CIA and H- tables along temperature.
  Pressure is always interpolated linearly in log P (the layer grid is fixed).

  *Supported values*:

  * ``linear`` (default): linear in log T, as in earlier versions.
  * ``pchip``: monotone piecewise-cubic Hermite (SciPy ``PchipInterpolator``
    slopes) in log T. The opacity is then C1 in temperature: no overshoot
    between grid nodes and no jump in the gradient at them, which helps
    gradient-based samplers (NUTS) and optimisers.

``opac.line``
  List of line opacity entries. Each entry is a mapping (flow-style shown)::

    - {species: H2O, path: ../../opac_data/os/H2O_R20000.zarr.zip}

  Fields:

  * ``species``: absorber name (string; must match the table contents)
  * ``path``: table file path (relative to experiment dir or absolute)

  File formats:

  * CK: ``.zarr.zip``, ``.zarr``, ``.h5``, ``.hdf5``, ``.npz``
  * OS: ``.zarr.zip``, ``.zarr``, ``.npz``

``opac.ray``
  List of Rayleigh species entries. Each entry is typically::

    - {species: H2}

  Supported Rayleigh species names are defined in ``exo_skryer/registry_ray.py``.
  Common choices include: ``H2``, ``He``, ``H``, ``e-``, ``N2``, ``O2``, ``CO``,
  ``CO2``, ``CH4``, ``NH3``.

``opac.cia``
  List of CIA pair entries::

    - {species: H2-H2, path: ../../opac_data/cia/H2-H2_2011.zarr.zip}

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

  With ``opac_cloud: file``, ``opac.cloud`` lists every species in the cloud
  file, each with its bulk density ``rho`` (g cm\ :sup:`-3`) and, for
  ``cloud_method: lxmie`` or ``madt``, its n,k ``path``. With ``f18`` an entry
  may set its own ``Q0``, ``Q1``, ``a``::

    cloud:
      - {species: Mg2SiO4, path: ../../nk_data/Mg2SiO4_amorph.txt, rho: 3.27}
      - {species: Fe,      path: ../../nk_data/Fe.txt,             rho: 7.87}


profile
=======

User-supplied atmospheric profiles, used by the ``file`` options of
``physics.vert_Tp``, ``vert_mu``, ``vert_chem``, ``vert_cloud`` and
``opac_cloud``. Each option is independent, e.g. a file T-p with
``vert_chem: fastchem_grid_jax``, or file clouds on an analytic T-p.

All files are whitespace-separated text with a header line naming the columns
(it may start with ``#``; other ``#`` lines are comments), pressure in bar, rows
in any order. Profiles are interpolated onto the model layers in log p (VMRs
and q_c in log value, zeros kept). Beyond a file's pressure range the VMRs
keep their end values; clouds are zero there.

``profile.tp_file``
  T-p-mu-Kzz profile, one row per layer::

    p_bar      T_K       mu_amu    Kzz_cm2s
    1.0e-6     850.0     2.33      1.0e10

  ``mu_amu`` (mean molecular weight, g mol\ :sup:`-1`) and ``Kzz_cm2s`` (eddy
  diffusion, cm\ :sup:`2` s\ :sup:`-1`) are optional. With ``vert_Tp: file``
  the file's pressures are the model layers; levels are placed at the
  geometric means between layers (the two end levels half a layer beyond the
  end layers in log p), with temperatures from a monotone cubic (PCHIP) in
  log p, extrapolated linearly at the two end levels. Emission at short
  wavelengths (the Wien tail) is sensitive to these level temperatures: a
  profile given only at layers gives spectra within ~1% there of the same
  analytic profile evaluated at the exact levels. ``mu_amu`` is used
  with ``vert_mu: file``. ``Kzz_cm2s`` is passed to ``quench_approx`` in place
  of ``log_10_Kzz``.

``profile.vmr_file``
  Volume mixing ratios, one row per layer::

    p_bar      H2O       CO        CH4       NH3

  Every opacity species needs a column. Without ``H2``/``He`` columns the
  remainder ``1 - sum(VMR)`` is filled with H2 + He at the solar ratio, as
  ``constant_vmr`` does.

``profile.cloud_file``
  Condensate clouds in long format, one row per layer and species::

    p_bar      species    q_c        reff_um   sigma
    1.0e-2     Mg2SiO4    1.2e-6     0.85      2.0
    1.0e-2     Fe         3.0e-7     1.40      2.0

  ``q_c`` is the condensate mass per mass of gas, ``reff_um`` the effective
  radius (µm) and ``sigma`` the lognormal geometric width (optional, default
  ``profile.cloud_sigma``). A species absent at a pressure has no condensate
  there. The optical properties are computed once at setup; species are
  combined by summing extinction and taking scattering-weighted single
  scattering albedo and asymmetry.

``profile.cloud_method``
  ``lxmie`` (default; Lorenz-Mie), ``madt`` (Rayleigh + MADT blend) or
  ``f18`` (Fisher & Heng 2018, coefficients from ``profile.f18``).

``profile.cloud_dist``
  ``lognormal`` (default; r_g = r_eff exp(-2.5 ln\ :sup:`2` sigma)) or
  ``mono`` (single radius r_eff).

``profile.cloud_sigma``
  Width used where the cloud file has no ``sigma`` column (default 2.0).

``profile.cloud_nr``
  Radii in the log-radius efficiency table (default 400). For
  ``lognormal`` the size integral is a quadrature over these table radii,
  which resolves the Mie ripples at large size parameter much better than the
  41 fixed nodes of the ``lxmie``/``madt_rayleigh`` schemes (at 400 radii,
  within ~0.3% of a converged integral; 800 gives ~1e-4). The table is
  computed once at setup; LX-MIE costs about 26 s per species at 1154
  wavelengths with 400 radii, scaling with the number of wavelengths. For
  ``mono`` the efficiencies are computed at each layer's radius exactly.

``profile.f18``
  ``{Q0, Q1, a}`` for ``cloud_method: f18`` (default ``{Q0: 20.0, Q1: 1.0, a: 4.0}``).

Example::

  physics:
    vert_Tp: file
    vert_mu: file
    vert_chem: file
    vert_cloud: file
    opac_cloud: file

  profile:
    tp_file: ../../profiles/my_bd_TpmuKzz.txt
    vmr_file: ../../profiles/my_bd_vmr.txt
    cloud_file: ../../profiles/my_bd_clouds.txt
    cloud_method: lxmie
    cloud_dist: lognormal


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
  * ``nautilus``
  * ``pymultinest``
  * ``nessai`` (needs the ``nessai`` extra, which installs PyTorch)
  * ``blackjax_smc``
  * ``pocomc`` (needs the pixi ``pocomc`` environment, or ``pip install pocomc``)
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
  Common fields: ``max_samples, num_live_points, s, k, c, allocation_target, replacement_width,
  checkpoint_dir, checkpoint_cadence, verbose, posterior_samples, seed``.

  ``allocation_target`` is ``uniform`` (default), ``evidence_improving`` or ``posterior_improving``.
  ``posterior_improving`` reaches an ESS target with fewer likelihood calls but gives a looser
  ``logZ``, so avoid it for model comparison. ``replacement_width`` is the number of chains run in
  parallel and defaults to ``10 * D``. ``max_samples`` is a hard limit, so a run can stop before its
  termination targets are met; the driver prints a warning if it does.

  A relative ``checkpoint_dir`` is taken from the experiment directory. A run resumes automatically
  from any checkpoint found there, and that checkpoint's state takes precedence over the YAML, so
  use a new directory for a new configuration. Timed saves, at most every ``checkpoint_cadence``
  seconds (default 3600), happen only at the end of a pass. To save part-way through a pass, send
  ``SIGINT``; ``SIGTERM`` does not save.

  Optional nested block: ``sampling.jaxns.termination`` with fields
  ``ess, evidence_uncert, dlogZ, max_samples, max_num_likelihood_evaluations``.

``sampling.blackjax_ns``
  Common fields: ``num_live_points, num_inner_steps, num_delete, batch_size, dlogz_stop,
  posterior_samples, seed, jit``. ``dlogz_stop`` is ``log(1 + Z_live / Z)``, as ``dlogz`` is
  for dynesty. ``batch_size`` (default 32) is how many particles are evaluated together. It
  sets peak memory and speed but not the result, so raise it until memory runs out;
  ``0`` evaluates all of them at once.

``sampling.ultranest``
  Common fields: ``num_live_points, min_num_live_points, dlogz, max_iters, verbose, show_status``.

``sampling.pymultinest``
  Common fields: ``n_live_points, evidence_tolerance, sampling_efficiency, n_iter_before_update,
  null_log_evidence, max_modes, mode_tolerance, seed, verbose, resume, importance_nested_sampling,
  multimodal, const_efficiency_mode``.

``sampling.nautilus``
  Common fields: ``n_live, n_update, split_threshold, n_networks, n_batch, n_like_new_bound, seed,
  filepath, resume, f_live, n_shell, n_eff, n_like_max, discard_exploration, timeout,
  equal_weight, verbose``.

``sampling.nessai``
  Common fields: ``nlive, stopping, seed, max_iteration, output, resume, checkpointing,
  checkpoint_interval, pytorch_threads, plot, log_level``. ``stopping`` is a ``dlogZ``
  criterion. Unbounded priors are truncated at 6 sigma. Set ``resume: true`` to continue
  a run from the checkpoint in ``output``.

``sampling.blackjax_smc``
  Common fields: ``num_particles, num_mcmc_steps, target_ess, max_iterations, batch_size,
  posterior_samples, seed, resume, checkpoint, checkpoint_interval``. This is persistent
  sampling with a random-walk kernel, so it needs no gradients. ``target_ess`` is a multiple
  of ``num_particles``, measured over all kept particles. The reported ``logZ_err`` only
  covers importance-weight scatter and is a lower limit, so repeat seeds for a full error.
  ``batch_size`` (default 32) works as for ``blackjax_ns``.

``sampling.pocomc``
  Common fields: ``n_effective, n_active, n_total, n_evidence, batch_size, posterior_samples,
  seed, output, resume, save_every, pytorch_threads, progress``. The likelihood is evaluated
  in batches of ``batch_size`` (default 32; ``null`` means ``n_active``). Set ``resume: true``
  to continue from the newest state file in ``output``.

``sampling.nuts``
  Common fields: ``backend, warmup, draws, seed, chains``.
  NumPyro only: ``forward_mode`` (default ``false``) selects forward-mode
  differentiation. Reverse mode is several times faster at typical parameter
  counts.
  Note: the current BlackJAX NUTS driver supports ``chains: 1`` only.


runtime
=======

``runtime.platform``
  *Supported values*: ``cpu``, ``gpu``, ``cuda``, or ``rocm``.
  ``gpu`` is retained as an alias for ``cuda``. Select ``rocm`` explicitly for
  an AMD GPU or APU.

``runtime.cuda_visible_devices``
  Only used with ``runtime.platform: gpu`` or ``cuda``. String like ``"0"`` or
  ``"0,1"``.

``runtime.rocm_visible_devices``
  Only used with ``runtime.platform: rocm``. String like ``"0"`` or ``"0,1"``;
  exported as ``ROCR_VISIBLE_DEVICES`` on Linux.

``runtime.preallocate`` / ``runtime.mem_fraction``
  Optional accelerator memory controls. Preallocation defaults to ``false`` and
  the memory fraction defaults to ``0.75`` for CUDA and ROCm. A lower fraction
  can be useful on an APU because system memory is shared with the CPU.

``runtime.threads``
  Only used when ``runtime.platform: cpu``; passed to NumPyro host device count.

``runtime.tf_gpu_allocator``
  Optional string. If set, exported as ``TF_GPU_ALLOCATOR`` when on GPU.
