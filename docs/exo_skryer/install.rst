.. _install:

************
Installation
************


First: Get the code
-------------------

Clone the repository, change directories into it::

    git clone https://github.com/ELeeAstro/Exo_Skryer.git
    cd Exo_Skryer

.. note::

    **JAX and GPU support:** ``Exo_Skryer`` installs ``jax`` automatically, but
    GPU/Metal acceleration requires a platform-specific JAX install (CUDA on
    Linux/Windows, Metal on macOS). For GPU support, install JAX with the
    appropriate backend *before* installing Exo_Skryer, following the
    `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.

Recommended: install in a clean environment
-------------------------------------------

We strongly recommend installing Exo_Skryer into a dedicated environment (not ``base``),
to avoid dependency conflicts (JAX, NumPyro, plotting, etc.).

Conda / Mambaforge (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create and activate a fresh environment, then install:

.. code-block:: bash

   conda create -n exo_skryer python=3.12 -y
   conda activate exo_skryer
   python -m pip install -e .


Python venv (alternative)
^^^^^^^^^^^^^^^^^^^^^^^^^

Create and activate a venv inside the repo, then install:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   python -m pip install -e .

Install via pip (coming soon)
-----------------------------

Once ``exo_skryer`` is available on PyPI, you will be able to install it with::

    python -m pip install exo_skryer


Verify installation
-------------------

.. code-block:: bash

   python -c "import exo_skryer; print('exo_skryer import ok')"

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

1) Install app dependencies (from the repo root):

.. code-block:: bash

   cd web_interface
   python -m pip install -r requirements.txt

2) Start the app:

.. code-block:: bash

   streamlit run app.py

Streamlit will print a local URL (typically ``http://localhost:8501``).

Next steps
----------

See :doc:`gettingstarted` for next steps, and running a test retrieval model!

