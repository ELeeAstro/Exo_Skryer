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
