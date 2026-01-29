***************
Getting started
***************

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

These demands have pushed the computational burden of retrieval models
Exo Skryer attempts to solve these computational issues without the using excessive amounts of high performance computing (HPC) power.
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

Your first model
----------------

The "experiments/HD209_Barstow_2020_trans_setup" provides a first taste of how to use Exo Skryer, performing a first retrieval model, as well as postprocessing, testing individual functions and other things.
We can try running in the command line, the example HD 209459b retrieval model::

    cd experiments/HD209_Barstow_2020_trans_setup
    python -u -m exo_skryer.run_retrieval --config retrieval_config.yaml

Where the model will read the YAML file after the --config flag, which contains the full information to run the retrieval model.

.. note::
    Make sure that the retrieval_config.yaml points to the correct (relative) path to the line opacity, cia and wavelength data in the opac: section, for example.

.. code-block:: yaml

   wl_master: ../../opac_data/ck/wl_ck_R250.txt

   line:
     - {species: H2O, path: ../../opac_data/ck/H2O_ck_R250.npz}
     - {species: Na, path: ../../opac_data/ck/Na_ck_R250.npz}
     - {species: K, path: ../../opac_data/ck/K_ck_R250.npz}

   cia:
     - {species: H2-H2, path: ../../opac_data/cia/H2-H2_2011.npz}
     - {species: H2-He, path: ../../opac_data/cia/H2-He_2011.npz}


After completion (a few minutes), the code will output both the dynesty.pkl file and posterior.nc (ArViZ format) files which can then be used to post-process the retrieval results. 


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

Where to plot full resolution spectra, the option "full_grid" in the YAML file must be : True.

The temperature-pressure (T-p) profile can be plotted using::

  python plot_Tp.py --config retrieval_config.yaml

References
----------
