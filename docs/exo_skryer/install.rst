.. _install:

************
Installation
************


First: Get the code
-------------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/ELeeAstro/Exo_Skryer.git

Change into the repository directory:

.. code-block:: bash

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

**Step 1** — create a fresh environment:

.. code-block:: bash

   conda create -n exo_skryer python=3.12 -y

**Step 2** — activate the environment:

.. code-block:: bash

   conda activate exo_skryer

**Step 2b** — install pip:

.. code-block:: bash

   conda install pip

**Step 3** — install Exo Skryer and its dependencies:

.. code-block:: bash

   python -m pip install -e .


Python venv (alternative)
^^^^^^^^^^^^^^^^^^^^^^^^^

**Step 1** — create a venv inside the repo:

.. code-block:: bash

   python -m venv .venv

**Step 2** — activate the venv:

.. code-block:: bash

   source .venv/bin/activate

.. note::

   On Windows use: ``.venv\Scripts\activate``

**Step 3** — install Exo Skryer and its dependencies:

.. code-block:: bash

   python -m pip install -e .

GPU Installation (NVIDIA/CUDA)
------------------------------

GPU support requires Linux with an NVIDIA GPU (SM ≥ 7.5, i.e. Turing or newer)
and NVIDIA driver ≥ 580. GPU JAX wheels are not available for macOS or Windows.

**Option 1 — pip (GPU extra)**

Install the ``gpu`` optional dependency group, which pulls in ``jax[cuda13]``
including the CUDA runtime:

.. code-block:: bash

   python -m pip install -e ".[gpu]"

Verify that JAX sees your GPU:

.. code-block:: bash

   python -c "import jax; print(jax.devices())"

You should see a ``CudaDevice`` listed alongside the CPU device.

**Option 2 — conda (recommended for HPC clusters)**

A ready-made conda environment file is provided that installs the CUDA-enabled
``jaxlib`` from conda-forge and then installs Exo_Skryer via pip:

.. code-block:: bash

   conda env create -f environment_gpu.yml
   conda activate exo_skryer_gpu

.. note::

   The JAX team recommends ``jax[cuda13]`` (CUDA 13, cuDNN ≥ 9.8) and plans
   to drop CUDA 12 support in a future release. See the
   `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
   for the latest requirements.

Install via pip (coming soon)
-----------------------------

Once ``exo_skryer`` is available on PyPI, you will be able to install it with:

.. code-block:: bash

   python -m pip install exo_skryer


Verify installation
-------------------

Check that the package imports correctly:

.. code-block:: bash

   python -c "import exo_skryer; print('exo_skryer import ok')"

Building the docs
-----------------

For local use and browsing of the documentation, the easiest method is via tox
from the repository root.

Install tox:

.. code-block:: bash

   python -m pip install tox

Build the docs:

.. code-block:: bash

   tox -e build-docs

The built HTML documentation will be in:

- ``docs/_build/html/index.html``

Running the web app (config generator)
--------------------------------------

The web interface is a Streamlit app in ``web_interface/``. It generates
retrieval YAML configuration files.

Navigate to the web interface directory:

.. code-block:: bash

   cd web_interface

Install the app dependencies:

.. code-block:: bash

   python -m pip install -r requirements.txt

Start the app:

.. code-block:: bash

   streamlit run app.py

Streamlit will print a local URL (typically ``http://localhost:8501``).

Next steps
----------

See :doc:`gettingstarted` for next steps, and running a test retrieval model!
