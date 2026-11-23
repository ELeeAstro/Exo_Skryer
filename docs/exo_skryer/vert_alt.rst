***********************
Altitude Profiles
***********************


Exo Skryer provides several vertical altitude calculation functions in the `~exo_skryer.vert_alt` module.

Hypsometric - Constant Gravity
===============================

Uses hypsometic equation to calculate altitude assuming a constant gravity throughout the atmosphere.
Reference pressure is assumed to be at the highest pressure of the atmosphere.

.. math::

   \Delta z = \frac{k_{\rm b}T}{\overline{\mu} \cdot {\rm amu} g_{\rm ref}} \ln\left(\frac{p_{\rm lower}}{p_{\rm upper}}\right)


**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_alt: hypsometric

   params:
     - { name: log_10_g, dist: uniform, low: 2.7, high: 3.3, transform: logit, init: 3.0 }

Hypsometric - Variable Gravity
==============================

Uses hypsometic equation to calculate altitude with variable gravity throughout the atmosphere, from a reference gravity, \ :math:`log_{10} g`, and radius, \ :math:`R_{\rm p}`.
Reference pressure is assumed to be at the highest pressure of the atmosphere.

.. math::

   g(z) = g_{\rm ref} \left(\frac{R_{\rm ref}}{R_{\rm ref} + z}\right)^{2}

With constant temperature and mean molecular weight within each layer, hydrostatic
balance integrates exactly in :math:`1/r` (with :math:`r = R_{\rm ref} + z`):

.. math::

   \frac{1}{r_{i+1}} = \frac{1}{r_i} - \frac{k_{\rm b} T_i}{\overline{\mu}_i \, {\rm amu} \, g_{\rm ref} R_{\rm ref}^{2}} \ln\left(\frac{p_i}{p_{i+1}}\right)

so the whole profile is a single cumulative sum,
:math:`1/r_j = 1/R_{\rm ref} - C_j` with :math:`C_j` the running sum of the per-layer
terms (no layer-by-layer loop). Altitudes and layer thicknesses are formed as
:math:`z_j = R_{\rm ref} C_j / (1/r_j)` and
:math:`\Delta z_i = (C_{i+1} - C_i) / \left[(1/r_i)(1/r_{i+1})\right]`, which avoids
subtracting nearly equal radii and keeps them accurate to machine precision.

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_alt: hypsometric_variable_g

   params:
     - { name: log_10_g, dist: uniform, low: 2.0, high: 5.0, transform: logit, init: 3.5 }
     - { name: R_p, dist: uniform, low: 0.8, high: 1.5, transform: logit, init: 1.0 }


Hypsometric - Variable Gravity with Reference Pressure
======================================================

Uses hypsometic equation to calculate altitude with variable gravity throughout the atmosphere, from a reference gravity and radius.
Reference pressure is given in the YAML parameters.

The same per-layer :math:`1/r` solution as above is used, but anchored so that
:math:`r = R_{\rm ref}` (:math:`z = 0`) at :math:`p_{\rm ref}`:
:math:`1/r_j = 1/R_{\rm ref} - (C_j - C_{\rm ref})`, where :math:`C_{\rm ref}` adds the
partial step from the level below :math:`p_{\rm ref}` using that layer's temperature
and mean molecular weight. Levels at higher pressure than :math:`p_{\rm ref}` have
negative altitude. :math:`p_{\rm ref}` is clipped to the pressure grid.

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_alt: hypsometric_variable_g_pref

   params:
     - { name: log_10_g, dist: uniform, low: 2.0, high: 5.0, transform: logit, init: 3.5 }
     - { name: R_p, dist: uniform, low: 0.8, high: 1.5, transform: logit, init: 1.0 }
     - { name: log_10_p_ref, dist: delta,   value: 0.0, transform: identity, init: 0.0}
