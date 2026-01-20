.. _install:

************
Installation
************


Install from source
-------------------

Clone the repository, change directories into it, and build from source::

    git clone https://github.com/ELeeAstro/Exo_Skryer.git
    cd Exo_Skryer

    python -m pip install -e .


.. note::

    ``Exo_Skryer`` requires ``jax`` as a dependency. Depending on your hardware, 
    you may want to install ``jax`` for your CPU or GPU. For details on jax 
    installation, see `JAX Installation <https://docs.jax.dev/en/latest/installation.html>`_.

Install via pip
---------------

To install the most recent release of ``exo_skryer``, run::

    python -m pip install exo_skryer


Running the code
----------------

We can try running in the command line, the example HD 189733b retrieval model::

    cd experiments/HD189_Barstow_2020_trans_setup
    python -u -m exo_skryer.run_retrieval --config retrieval_config.yaml

Where the model will read the YAML file after the --config flag, which contains the full information to run the retrieval model.