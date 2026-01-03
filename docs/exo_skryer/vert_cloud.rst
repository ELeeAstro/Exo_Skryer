***************************
Cloud Vertical Profiles
***************************

The ``vert_cloud`` module provides functions to compute the vertical distribution of cloud
mass mixing ratio (``q_c_lay``) as a function of pressure and atmospheric conditions.

These functions are called after the atmospheric structure (temperature, chemistry, altitude)
has been computed, and return a cloud mass mixing ratio profile that is then used by cloud
opacity functions to compute optical properties.

Available Cloud Profiles
=========================

No Cloud
--------

Returns zero cloud mass mixing ratio everywhere (cloud-free atmosphere).

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_cloud: none

**Parameters:**

None required.


Exponential Decay Profile
--------------------------

Exponential decay profile with smooth tanh base cutoff.

The cloud mass mixing ratio follows:

.. math::

   q_c(P) = q_{c,0} \left(\frac{P}{P_{\rm base}}\right)^\alpha S_{\rm base}(P)

where :math:`S_{\rm base}` is a smooth gate function that transitions from 1 (aloft)
to 0 (deep atmosphere) around :math:`P_{\rm base}`.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_cloud: exponential

   params:
     - { name: log_10_q_c_0, dist: uniform, lower: -12, upper: -2 }
     - { name: log_10_H_cld, dist: uniform, lower: -2, upper: 2 }
     - { name: log_10_p_base, dist: uniform, lower: -6, upper: 3 }
     - { name: width_base_dex, dist: delta, value: 0.25 }  # optional

**Parameters:**

- ``log_10_q_c_0``: Log₁₀ cloud mass mixing ratio at the base pressure
- ``log_10_H_cld``: Log₁₀ cloud pressure scale height parameter (controls α = 1/H_cld)
- ``log_10_p_base``: Log₁₀ base pressure in bar
- ``width_base_dex``: Width of the base cutoff in dex (default: 0.25)


Slab Profile
------------

Uniform cloud slab with hard pressure cutoffs.

The cloud has constant :math:`q_c` between :math:`P_{\rm top}` and :math:`P_{\rm bot}`,
and zero outside.

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_cloud: slab

   params:
     - { name: log_10_q_c, dist: uniform, lower: -12, upper: -2 }
     - { name: log_10_p_top_slab, dist: uniform, lower: -6, upper: 2 }
     - { name: log_10_dp_slab, dist: uniform, lower: 0.5, upper: 4 }

**Parameters:**

- ``log_10_q_c``: Log₁₀ cloud mass mixing ratio inside the slab
- ``log_10_p_top_slab``: Log₁₀ pressure at the top of the slab in bar
- ``log_10_dp_slab``: Log₁₀ pressure extent of the slab (P_bot = P_top × 10^Δlog_P)


Integration with Cloud Opacity
===============================

The cloud vertical profile functions compute ``q_c_lay``, which is then added to the
atmospheric state dictionary. Cloud opacity functions (e.g., ``direct_nk``, ``grey_cloud``)
access this value to compute optical properties.

**Workflow:**

1. Compute atmospheric structure (T, P, chemistry, altitude, density)
2. Call ``vert_cloud`` kernel → produces ``q_c_lay``
3. Add ``q_c_lay`` to ``state`` dictionary
4. Cloud opacity function uses ``state["q_c_lay"]`` to compute extinction


API Reference
=========

.. automodapi:: exo_skryer.vert_cloud
   :no-heading:
   :no-main-docstr:


See Also
========

- :doc:`opacity_cloud` for cloud optical property calculations
- :doc:`vert_Tp` for temperature-pressure profiles
- :doc:`vert_alt` for altitude structure
- :doc:`api` for complete API reference
