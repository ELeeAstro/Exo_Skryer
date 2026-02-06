.. _gettingstarted:

***************
Getting started
***************

This guide assumes you have already completed the :doc:`install` steps.

Why Exo Skryer?
---------------

Contemporary retrieval frameworks for sub-stellar atmospheres have numerous requirements to fulfill to be useful tools for analysis and scientific discovery:

* Robust parameter estimation
* Statistically consistent Bayesian evidence comparison
* Fast completion times
* Flexibility and scalability
* Physically interpretable results

In the era of JWST where medium resolution, large wavelength range coverage observational data is available, 
retrieval results have now become a standard methodology to publish alongside new observational data.
Retrieval models for high spectral resolution data are also being developed, which will become boosted by the near-future operation of the ELT.

Exo Skryer attempts to solve computational issues associated with in-depth retrieval modelling, without using excessive amounts of high performance computing (HPC) power.
Exo Skryer uses the JAX extension to Python to accelerate both the sampling and forward model evaluations, enabling efficient, scalable operation on CPUs and GPUs.
This enables complex retrieval modelling to be performed in good time on desktop computers with a GPU.

A simple, online app can help you get an initial YAML configuration file started:
`Exo Skryer YAML configuration tool <https://exoskryer.streamlit.app/>`__


The current version of Exo Skryer offers the following nested sampling options:

* pymultinest 
* dynesty 
* JAXNS 
* ultranest (experimental) 
* blackjax-ns (experimental)
* polychord (experimental)

As well as two NUTS MCMC samplers:

* numpyro 
* blackjax 

Opacity data
------------

Some input correlated-k tables, opacity sampled tables and CIA tables can be found at the following:
`Exo Skryer opacity collection <https://drive.google.com/drive/folders/1qmTAwizPOZATYvrOeXSDHTKKhxpi-LKA?usp=drive_link>`__

Exo Skryer can also use the TauREX (opacity sampling mode) and petitRADTRANS (correlated-k mode) tables available from the `ExoMol website <https://www.exomol.com/>`__

Your first forward model (transmission)
---------------------------------------

Before running a full retrieval, it is useful to generate a single forward model spectrum to check that the configuration is sensible and produces a reasonable output.
The ``experiments/forward_model_trans`` directory contains a ready-to-run transmission forward model example.

The key difference from a retrieval configuration is that **all parameters use** ``dist: delta`` with explicit values, so no sampling is performed.
The ``sampling`` section is omitted entirely.

From the repository root, run::

    cd experiments/forward_model_trans
    python run_forward_model.py --config forward_config.yaml

.. note::
    Make sure that the forward_config.yaml file points to the correct (relative) path to the line opacity, cia and wavelength data in the opac: section, for example.

.. code-block:: yaml

   wl_master: ../../opac_data/ck/wl_ck_R250.txt

   line:
     - {species: H2O, path: ../../opac_data/ck/H2O_ck_R250.npz}
     - {species: Na, path: ../../opac_data/ck/Na_ck_R250.npz}
     - {species: K, path: ../../opac_data/ck/K_ck_R250.npz}

   cia:
     - {species: H2-H2, path: ../../opac_data/cia/H2-H2_2011.npz}
     - {species: H2-He, path: ../../opac_data/cia/H2-He_2011.npz}

This will produce a high-resolution spectrum file ``forward_spectrum_highres.txt`` containing two columns: wavelength (um) and transit depth.

If you also want the spectrum convolved to an observational bandpass, pass the ``--obs`` flag with a path to an observational data file::

    python run_forward_model.py --config forward_config.yaml --obs ../../obs_data/WASP-107b_JWST.txt

This additionally produces ``forward_spectrum_binned.txt`` with three columns: wavelength (um), half bin width (um), and transit depth.

To plot the output spectrum::

    python plot_spectrum.py --spectrum forward_spectrum_highres.txt

Or with an observational data overlay::

    python plot_spectrum.py --spectrum forward_spectrum_binned.txt --obs ../../obs_data/WASP-107b_JWST.txt

The example ``forward_config.yaml`` uses a Guillot temperature-pressure profile, correlated-k opacities at R=250, constant VMR chemistry with H2O, CO2, CO and CH4, and an exponential decay cloud with F18 opacity.
To adjust the model, simply edit the parameter values in the ``params`` section of the YAML file and re-run.

.. note::
   When ``data.obs`` is set to ``None`` in the config (the default), only the high-resolution spectrum is produced.
   The ``--obs`` CLI flag overrides this setting.


Your first forward model (emission)
------------------------------------

The ``experiments/forward_model_em`` directory provides an emission forward model example configured for a brown dwarf, outputting the absolute planetary flux.

From the repository root, run::

    cd experiments/forward_model_em
    python run_forward_model.py --config forward_config.yaml

This produces ``forward_spectrum_highres.txt`` with two columns: wavelength (um) and flux (erg s\ :sup:`-1` cm\ :sup:`-2` cm\ :sup:`-1`).

To plot the emission spectrum::

    python plot_spectrum.py --spectrum forward_spectrum_highres.txt

The example ``forward_config.yaml`` uses:

* ``rt_scheme: emission_1d`` with ``emission_mode: brown_dwarf`` and the ``eaa`` emission solver
* A modified Milne temperature-pressure profile (``vert_Tp: Milne_modified``) with parameters ``T_int``, ``T_ratio``, ``log_10_k_ir``, ``log_10_p_t`` and ``beta``
* Gravity-consistent altitude grid (``vert_alt: hypsometric_variable_g_pref``) using the mass parameter ``M_p``
* Correlated-k opacities at R=250 with H2O, CO, CH4, NH3, Na and K
* Distance parameter ``D`` (parsecs) for absolute flux scaling: :math:`F = (R_0 / D)^2 \times I_{\rm TOA}`

.. note::
   In brown dwarf mode, ``R_s`` is set to 0.0 (unused) and the output is absolute flux rather than a planet-to-star flux ratio.


Your first retrieval model
--------------------------

The "experiments/HD209_Barstow_2020_trans_setup" provides a first taste of how to use Exo Skryer, performing a first retrieval model, as well as postprocessing, testing individual functions and other things.
From the repository root, run the example HD 209458b retrieval model::

    cd experiments/HD209_Barstow_2020_trans_setup
    python -u -m exo_skryer.run_retrieval --config retrieval_config.yaml

Where the model will read the YAML file after the --config flag, which contains the full information to run the retrieval model.

.. note::
    Make sure that the retrieval_config.yaml file points to the correct (relative) path to the line opacity, cia and wavelength data in the opac: section, for example.

.. code-block:: yaml

   wl_master: ../../opac_data/ck/wl_ck_R250.txt

   line:
     - {species: H2O, path: ../../opac_data/ck/H2O_ck_R250.npz}
     - {species: Na, path: ../../opac_data/ck/Na_ck_R250.npz}
     - {species: K, path: ../../opac_data/ck/K_ck_R250.npz}

   cia:
     - {species: H2-H2, path: ../../opac_data/cia/H2-H2_2011.npz}
     - {species: H2-He, path: ../../opac_data/cia/H2-He_2011.npz}


After completion (a few minutes), the code will output both the dynesty.pkl file and posterior.nc (ArviZ format) files which can then be used to post-process the retrieval results. 


The traditional corner plot can be plotted through the script::

  python posterior_corner.py --config retrieval_config.yaml

With additional options, such mapping LaTeX label formatting and positioning of the variables through corner_config.yaml, where non-present parameters are not plotted::

  python posterior_corner.py --config retrieval_config.yaml --label-map corner_config.yaml

Or presenting kernel density estimates instead of histograms::

  python posterior_corner.py --config retrieval_config.yaml --label-map corner_config.yaml --kde-diag


Best-fit median models for transmission spectra models can be produced using::

  python bestfit_plot.py --config retrieval_config.yaml

or for emission spectra models::

  python bestfit_em_plot.py --config retrieval_config.yaml

To plot full resolution spectra, set ``full_grid: true`` in the YAML file.

The temperature-pressure (T-p) profile can be plotted using::

  python plot_Tp.py --config retrieval_config.yaml
