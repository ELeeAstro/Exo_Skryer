******************************
Preparing Observational Data
******************************

.. note::
   This page is a placeholder and will be expanded in a future update.

Exo_Skryer reads observational data via :py:func:`exo_skryer.read_obs.read_obs_data`.
The data file is a whitespace-delimited text file whose path is set in the YAML
configuration under ``data.obs``.

File Format
-----------

Each row represents one spectral bin. Columns are:

.. list-table::
   :header-rows: 1
   :widths: 10 20 50

   * - Column
     - Name
     - Description
   * - 1
     - ``wavelength``
     - Central wavelength of the bin [micron]
   * - 2
     - ``delta_wavelength``
     - Half-width of the wavelength bin [micron]
   * - 3
     - ``y``
     - Observed quantity (e.g. :math:`(R_p/R_s)^2` for transmission, :math:`F_p/F_s` for emission)
   * - 4
     - ``dy``
     - Symmetric uncertainty on ``y`` (1-sigma). If a 5th numeric column is present it is treated as the negative error bar, and the larger of the two is used.
   * - 5 (optional)
     - ``response_mode``
     - Instrument response function type for the bin (default: ``boxcar``)
   * - 6 (optional)
     - ``offset_group``
     - Label grouping bins that share an instrument-offset parameter (e.g. ``MIRI``)

Lines beginning with ``#`` are treated as comments.

Example
-------

A minimal four-column file:

.. code-block:: text

   # wavelength  dwl       depth      uncertainty
   1.0           0.01      0.0207     0.00008
   1.1           0.01      0.0205     0.00007

A six-column file with response mode and offset groups:

.. code-block:: text

   # wl    dwl     depth     dy       response  offset_group
   5.064   0.075   0.020065  0.000084 boxcar    MIRI
   5.214   0.075   0.020083  0.000091 boxcar    MIRI

YAML Configuration
------------------

Point to the data file with:

.. code-block:: yaml

   data:
     obs: obs_data/my_planet.txt
