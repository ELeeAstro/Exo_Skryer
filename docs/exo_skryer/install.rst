.. _install:

************
Installation
************


Recommended: install in a clean environment
-------------------------------------------

We strongly recommend installing Exo_Skryer into a dedicated environment (not ``base``),
to avoid dependency conflicts (JAX, NumPyro, plotting, etc.).

Conda / Mambaforge (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create and activate a fresh environment:

.. code-block:: bash

   conda create -n exo_skryer python=3.12 -y
   conda activate exo_skryer

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

Verify:

.. code-block:: bash

   python -c "import exo_skryer; print('exo_skryer import ok')"

Python venv (alternative)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip
   python -m pip install -e .


GPU note (JAX)
^^^^^^^^^^^^^^

JAX GPU/Metal support requires a platform-specific install (CUDA on Linux/Windows,
Metal on macOS). Follow the official JAX installation instructions first, then install
Exo_Skryer (``pip install -e .``) inside the same environment.

JAX install instructions:

.. code-block:: text

   https://jax.readthedocs.io/en/latest/installation.html


Install via pip (eventually)
----------------------------

To install the most recent release of ``exo_skryer``, run::

    python -m pip install exo_skryer

Building the docs
-----------------

For local use and browsing of the documentation, the easiest method is via tox
from the repository root:

.. code-block:: bash

   python -m pip install tox
   tox -e build-docs

The built HTML documentation will be in:

- ``docs/_build/html/index.html``

Running the web app (config generator)
--------------------------------------

The web interface is a Streamlit app in ``web_interface/``. It generates
retrieval YAML configuration files.

1) Install app dependencies:

.. code-block:: bash

   cd /path/to/Exo_Skryer/web_interface
   python -m pip install -r requirements.txt

2) Start the app:

.. code-block:: bash

   streamlit run app.py

Streamlit will print a local URL (typically ``http://localhost:8501``).


Next steps
----------

See :doc:`gettingstarted` for next steps, and running a test retrieval model!

