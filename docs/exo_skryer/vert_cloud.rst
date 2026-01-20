***************************
Cloud Vertical Profiles
***************************

Should cloud opacities be required in the retrieval model, first a vertical profile must be defined.
Exo Skryer uses the mass mixing ratio, :math:`q_{\rm c}` [g g\ :sup:`-1`],  of cloud materials as the basic unit of the vertical cloud profile

.. math::

  q_{\rm c} = \frac{\rho_{\rm c}}{\rho_{\rm a}}

where :math:`\rho_{\rm c}` [g cm\ :sup:`-3`] is the condensed cloud mass density and :math:`\rho_{\rm a}` [g cm\ :sup:`-3`] the background atmospheric mass density.
Exo Skryer provides several vertical cloud profile functions in the `~exo_skryer.vert_cloud` module.

No Cloud
--------

A zero cloud profile can be defined, which is useful should custom methods be required that don't need the cloud mass mixing ratio

.. math::

  q_{\rm c}(p) = 0

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_cloud: no_cloud


Constant Uniform Slab
---------------------

A constant slab profile across the entire pressure domain can be given.

.. math::

  q_{\rm c}(p) = q_{\rm c, slab}

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_cloud: const_profile


Slab Profile
------------

A slab profile is defined with a constant :math:`q_{\rm c}` between a top pressure and a given :math:`\Delta` pressure.
This follows a similar profile to ` <>`_.

**Example YAML Configuration:**

.. math::

   q_{\rm c}(p) = \begin{cases}
      q_{\rm c, slab}
       &  p \le p_{\rm c, top} + \Delta p \\
      0 & p > p_{\rm c, top} + \Delta p\\
   \end{cases}

.. code-block:: yaml

   physics:
     vert_cloud: slab_profile

   params:
     - { name: log_10_q_c, dist: uniform, low: -12, high: -2, transform: logit, init: -6 }
     - { name: log_10_p_top_slab, dist: uniform, low: -6, high: 2, transform: logit, init: -2 }
     - { name: log_10_dp_slab, dist: uniform, low: -2, high: 2, transform: logit, init: 0.5 }


Exponential Decay Profile
-------------------------

The exponential decay profile reduces the :math:`q_{\rm c}` with pressure from a given base value at a base pressure exponentially, given by a decay rate :math:`\alpha`.
Below the base pressure, there is zero cloud.

**Example YAML Configuration:**

.. math::

   q_{\rm c}(p) = \begin{cases}
      q_{\rm c, base} \left(\frac{p}{p_{\rm c, base}}\right)^{\alpha} &  p \le p_{\rm c, base} \\
      0 & p > p_{\rm c, base}\\
   \end{cases}

.. code-block:: yaml

   physics:
     vert_cloud: exponential_decay_profile

   params:
     - { name: log_10_q_c, dist: uniform, low: -12, high: -2, transform: logit, init: -6 }
     - { name: log_10_alpha_cld, dist: uniform, low: -2, high: 2, transform: logit, init: 0.0 }
     - { name: log_10_p_base, dist: uniform, low: -6, high: 2, transform: logit, init: -1 }
