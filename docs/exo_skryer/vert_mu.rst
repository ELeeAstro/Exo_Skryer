********************************************
Mean Molecular Weight :math:`\overline{\mu}`
********************************************

The mean molecular weight module, `~exo_skryer.vert_mu`, provides functions to compute the mean molecular weight of the atmosphere. 

Exo Skryer provides several mean molecular weight calculation functions in the `~exo_skryer.vert_mu` module.


Constant :math:`\overline{\mu}`
-------------------------------

Assume a single constant value for mean molecular weight throughout the atmosphere.

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_mu: constant_mu

   params:
     - { name: mu, dist: uniform, low: 2.0, high: 3.0, transform: logit, init: 2.3 }


Dynamic :math:`\overline{\mu}`
------------------------------

Computes the mean molecular weight, :math:`\overline{\mu}` [g mol\ :sup:`-1`], at each layer.
This is the sum of each species, :math:`i`, volume mixing ratio (VMR), :math:`x` with it's respective molecular weight :math:`\mu` [g mol\ :sup:`-1`].

.. math::

   \overline{\mu} = \sum_{i}x_{i}\mu_{i}


**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_mu: dynamic
