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
    GPU/Metal acceleration requires a platform-specific JAX install (CUDA or
    ROCm on Linux, Metal on macOS). For GPU support, install JAX with the
    appropriate backend *before* installing Exo_Skryer, following the
    `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.

Recommended: install in a clean environment
-------------------------------------------

We strongly recommend installing Exo_Skryer into a dedicated environment (not ``base``),
to avoid dependency conflicts (JAX, NumPyro, plotting, etc.). Exo_Skryer supports
Python 3.12–3.14.

Pixi
^^^^

`Pixi <https://pixi.sh>`_ builds the environment from ``pyproject.toml`` and the
committed ``pixi.lock``, so every machine gets the same package versions. Compiled
libraries (NumPy, SciPy, HDF5, netCDF, …) come from conda-forge; JAX and the
JAX-based packages come from PyPI so the latest JAX release is used. No conda
installation or ``base`` environment is needed.

**Step 1** — install pixi (once per machine):

.. code-block:: bash

   curl -fsSL https://pixi.sh/install.sh | sh

**Step 2** — create the environment from the repository root:

.. code-block:: bash

   pixi install

**Step 3** — run commands inside the environment, either one at a time or from
an activated shell:

.. code-block:: bash

   pixi run exo-skryer --config experiments/<experiment>/retrieval_config.yaml
   pixi shell                  # activated shell; leave with `exit`

The environments defined in ``pyproject.toml`` are:

=============  ==========================================================
Environment    Contents
=============  ==========================================================
``default``    CPU JAX and atmodeller (macOS arm64 and Linux x86-64)
``cuda``       NVIDIA GPU JAX (``jax[cuda13]``), Linux x86-64 only
``rocm``       AMD GPU JAX using the system ROCm, Linux x86-64 only
``test``       pytest plus all optional samplers
``nessai``     ``default`` plus nessai and PyTorch
``docs``       Sphinx documentation build
``py312`` …    ``default`` pinned to Python 3.12, 3.13 or 3.14
=============  ==========================================================

Select one with ``-e``, e.g. ``pixi run -e cuda exo-skryer --config …``.
``pixi run devices`` prints the JAX version and the devices JAX can see.

.. note::

   On a cluster, set ``PIXI_CACHE_DIR`` to a scratch or project directory
   before ``pixi install`` to keep the package cache out of your home quota.

Conda / Mambaforge
^^^^^^^^^^^^^^^^^^

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

**Option 2 — pixi (recommended for HPC clusters)**

The ``cuda`` environment installs ``jax[cuda13]``, including the CUDA runtime,
so only the NVIDIA driver is needed on the system:

.. code-block:: bash

   pixi install -e cuda
   pixi run -e cuda devices

**Option 3 — conda**

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

AMD APU Installation (ROCm)
----------------------------

Exo Skryer supports AMD GPUs and APUs through JAX's ROCm plugin on Linux. ROCm
does not support every Ryzen APU, so first check the Ryzen APU and operating
system tables in AMD's current `ROCm compatibility matrix
<https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>`_.
The graphics model or LLVM target reported by ``rocminfo`` must be supported;
having an AMD integrated GPU alone is not sufficient.

**Option 1 — AMD's JAX container (recommended)**

AMD publishes a preconfigured ``rocm/jax`` image. From the repository root:

.. code-block:: bash

   docker pull rocm/jax:latest
   docker run --rm -it \
       --device=/dev/kfd --device=/dev/dri \
       --group-add video --ipc=host --shm-size 16G \
       -v "$PWD":/workspace -w /workspace \
       rocm/jax:latest bash
   python -m pip install -e .

For reproducible work, replace ``latest`` with a versioned tag whose Python and
JAX versions satisfy this project. AMD's `JAX on ROCm installation guide
<https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html>`_
lists current validated images and matching package versions.

**Option 2 — bare metal**

Install ROCm for the exact APU and Linux release, then install Exo Skryer with
the ``rocm`` extra. This uses JAX's ``rocm7-local`` dependency group, which
selects compatible JAX, jaxlib, ROCm plugin, and PJRT packages while using the
system ROCm installation:

.. code-block:: bash

   rocminfo | grep -m1 gfx
   export LLVM_PATH=/opt/rocm/llvm
   python -m pip install -e ".[rocm]"
   python -c "import jax; print(jax.devices())"

The JAX check should show a ``RocmDevice``. If device access is denied, ensure
the user belongs to the system's ``render`` and ``video`` groups, then log out
and back in.

**Option 3 — pixi (bare metal)**

The ``rocm`` environment installs the same ``rocm7-local`` packages against the
system ROCm, and sets ``LLVM_PATH=/opt/rocm/llvm`` when it is activated:

.. code-block:: bash

   pixi install -e rocm
   pixi run -e rocm devices

Finally select ROCm in the retrieval configuration. On a single-APU machine,
``rocm_visible_devices`` may be omitted:

.. code-block:: yaml

   runtime:
     platform: rocm
     rocm_visible_devices: "0"
     preallocate: false
     mem_fraction: 0.75
     threads: 4

On CPU runs, ``threads`` limits the cores JAX uses by restricting the process's
CPU affinity. This is Linux-only; on macOS JAX uses every core and a warning is
printed. Processes sharing a node under MPI or Slurm take consecutive blocks of
``threads`` cores by local rank.

The ``gpu`` platform remains a CUDA alias for compatibility and will not select
an AMD device. APU memory is shared with the CPU; reduce ``mem_fraction`` if the
desktop or retrieval is being starved of memory.

If JAX fails with ``assert _default_backend is not None``, first check the
configuration for ``platform: gpu``. That value forces CUDA; change it to
``platform: rocm`` for an AMD APU. If ROCm then fails during startup, compare
the installed JAX packages and confirm device access:

.. code-block:: bash

   python -m pip list | grep -E '^jax|rocm'
   rocminfo | grep -m1 gfx
   ls -l /dev/kfd /dev/dri/render*

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

With pixi, build the docs from the repository root with:

.. code-block:: bash

   pixi run -e docs sphinx-build -b html docs docs/_build/html

Otherwise, the easiest method is via tox from the repository root.

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
