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
=============================

Uses hypsometic equation to calculate altitude with variable gravity throughout the atmosphere, from a reference gravity, \ :math:`log_{10} g`, and radius, \ :math:`R_{\rm p}`.
Reference pressure is assumed to be at the highest pressure of the atmosphere.

.. math::

   g(z) = g_{\rm ref} \left(\frac{R_{\rm ref}}{R_{\rm ref} + z}\right)^{2}

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

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_alt: hypsometric_variable_g_pref

   params:
     - { name: log_10_g, dist: uniform, low: 2.0, high: 5.0, transform: logit, init: 3.5 }
     - { name: R_p, dist: uniform, low: 0.8, high: 1.5, transform: logit, init: 1.0 }
     - { name: log_10_p_ref, dist: delta,   value: 0.0, transform: identity, init: 0.0}
