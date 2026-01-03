***********************************
Temperature-pressure (T-p) Profiles
***********************************

The T-p profile describes the thermal structure of an atmosphere as a function of pressure.
For retreival models, typically a quick to calculate, physically plausible parameterised T-p profile that forms the basis for the forward radiative-transfer model.
Exo Skryer provides several analytical T-p profile functions in the `~exo_skryer.vert_Tp` module.

Isothermal
----------

The simplest T-p profile is assuming that the entire atmosphere is a constant value:

.. math::

   T(p) = T_{\rm iso}

This is particularly useful for transmission spectroscopy, where
For emission, do not use an isothermal atmosphere unless under very specific test conditions, as no spectral features would be produced.

Required parameters: :math:`T_{\rm iso}` [K]

**Example Plot:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from exo_skryer.vert_Tp import isothermal
   from exo_skryer.data_constants import bar

   # Create pressure grid
   nlev = 100
   p_bot = np.log10(100.0)
   p_top = np.log10(1e-6)
   p_lev = np.logspace(p_bot, p_top, nlev) * bar  # 1e-6 to 100 bar in dyne/cm²

   # Example parameters
   params = {"T_iso": 400.0}
   T_lev, T_lay = isothermal(p_lev, params)

   # Plot
   fig, ax = plt.subplots(figsize=(10, 5))
   ax.semilogy(T_lev, p_lev/bar, c='orchid')
   ax.set_xlabel('Temperature [K]', fontsize=16)
   ax.set_ylabel('pressure [bar]', fontsize=16)
   ax.set_title('Isothermal T-p Profile', fontsize=14)
   ax.tick_params(labelsize=14)
   ax.invert_yaxis()
   plt.tight_layout()

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_Tp: isothermal

   params:
     - { name: T_iso, dist: uniform, low: 500.0, high: 2000.0, transform: logit, init: 1000.0 }


Barstow (2020)
--------------

`Barstow (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4183B/abstract>`_ used a simple T-p stucture with the following form:

.. math::

   T(p) = \begin{cases}
      T_{\rm strat} &  p \le p_1 \\
      T_{\rm strat} \left(\frac{p}{p_1}\right)^{\kappa} & p_1 < p < p_2 \\
      T_{\rm strat} \left(\frac{p_2}{p_1}\right)^{\kappa}  & p \ge p_2
   \end{cases}

where :math:`p_1` = 0.1 bar, :math:`p_2` = 1.0 bar and :math:`\kappa` is the adiabatic coefficient (default = 2/7).
This forms an initial upper atmosphere isotherm at :math:`T_{\rm strat}` at :math:`p \le p_1`, an adiabatic gradient for :math:`p_1 < p < p_2`, then a deep isothermal region at :math:`p \ge p_2`.

Required parameters: :math:`T_{\rm strat}` [K]

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from exo_skryer.vert_Tp import Barstow
   from exo_skryer.data_constants import bar

   # Create pressure grid
   nlev = 100
   p_bot = np.log10(100.0)
   p_top = np.log10(1e-6)
   p_lev = np.logspace(p_bot, p_top, nlev) * bar

   # Example parameters
   params = {"T_strat": 300.0}
   T_lev, T_lay = Barstow(p_lev, params)

   # Plot
   fig, ax = plt.subplots(figsize=(10, 5))
   ax.semilogy(T_lev, p_lev/bar, c='crimson')
   ax.set_xlabel('Temperature [K]', fontsize=16)
   ax.set_ylabel('pressure [bar]', fontsize=16)
   ax.set_title('Barstow (2020) T-p Profile', fontsize=14)
   ax.tick_params(labelsize=14)
   ax.invert_yaxis()
   plt.tight_layout()

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_Tp: Barstow

   params:
     - { name: T_strat, dist: uniform, low: 100.0, high: 1000.0, transform: logit, init: 300.0 }


Guillot (2010)
--------------

The `Guillot (2010) <https://www.aanda.org/articles/aa/abs/2010/10/aa13396-09/aa13396-09.html>`_ profile combines internal heating and external irradiation, commonly used for hot Jupiters and other irradiated planets.

Required parameters: :math:`T_{\rm int}` [K], :math:`T_{\rm eq}` [K], :math:`\log_{10} \kappa_{\rm ir}` [cm\ :sup:`2` g\ :sup:`-1`], :math:`\log_{10} \gamma_v` (visible-to-IR opacity ratio), :math:`\log_{10}g` [cm s\ :sup:`-2`], :math:`f_{\rm hem}` (hemispheric redistribution factor)

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from exo_skryer.vert_Tp import Guillot
   from exo_skryer.data_constants import bar

   # Create pressure grid
   nlev = 100
   p_bot = np.log10(100.0)
   p_top = np.log10(1e-6)
   p_lev = np.logspace(p_bot, p_top, nlev) * bar

   # Example parameters for a hot Jupiter
   params = {
       "T_int": 100.0,
       "T_eq": 1000.0,
       "log_10_k_ir": -2.0,
       "log_10_gam_v": -2,
       "log_10_g": 3.0,
       "f_hem": 0.25
   }
   T_lev, T_lay = Guillot(p_lev, params)

   # Plot
   fig, ax = plt.subplots(figsize=(10, 5))
   ax.semilogy(T_lev, p_lev/bar, c='darkorange')
   ax.set_xlabel('Temperature [K]', fontsize=16)
   ax.set_ylabel('pressure [bar]', fontsize=16)
   ax.set_title('Guillot (2010) T-p Profile', fontsize=14)
   ax.tick_params(labelsize=14)
   ax.invert_yaxis()
   plt.tight_layout()

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_Tp: Guillot

   params:
     - { name: T_int, dist: uniform, low: 100.0, high: 1000.0, transform: logit, init: 500.0 }
     - { name: T_eq, dist: uniform, low: 1000.0, high: 3000.0, transform: logit, init: 1500.0 }
     - { name: log_10_k_ir, dist: uniform, low: -6, high: 6, transform: logit, init: -2 }
     - { name: log_10_gam_v, dist: uniform, low: -3, high: 3, transform: logit, init: 0.0 }
     - { name: log_10_g, dist: uniform, low: 2.0, high: 4.0, transform: logit, init: 3.0 }
     - { name: f_hem, dist: delta, value: 0.25, transform: identity, init: 0.25}


Madhusudhan & Seager (2009)
----------------------------

The `Madhusudhan & Seager (2009) <https://ui.adsabs.harvard.edu/abs/2009ApJ…707…24M/abstract>`_ profile uses a piecewise polynomial parameterization with multiple pressure nodes, allowing for flexible T-p profiles with thermal inversions.

This scheme divides the atmosphere into three regions defined by pressure boundaries :math:`P_1`, :math:`P_2`, and :math:`P_3`, with each region having its own temperature gradient controlled by slope parameters :math:`a_1` and :math:`a_2`. This flexibility allows the model to capture both standard profiles and strong thermal inversions often seen in hot Jupiters.

Required parameters: :math:`a_1`, :math:`a_2` (polynomial slope coefficients), :math:`\log_{10} P_1`, :math:`\log_{10} P_2`, :math:`\log_{10} P_3` [bar] (pressure nodes), :math:`T_{\rm ref}` [K] (reference temperature at top of atmosphere)

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from exo_skryer.vert_Tp import MandS
   from exo_skryer.data_constants import bar

   # Create pressure grid
   nlev = 100
   p_bot = np.log10(100.0)
   p_top = np.log10(1e-6)
   p_lev = np.logspace(p_bot, p_top, nlev) * bar

   # Example parameters
   params = {
       "a1": 0.51,
       "a2": 0.51,
       "log_10_P1": -4.0,
       "log_10_P2": -2.0,
       "log_10_P3": 1.0,
       "T_ref": 1000.0
   }
   T_lev, T_lay = MandS(p_lev, params)

   # Plot
   fig, ax = plt.subplots(figsize=(10, 5))
   ax.semilogy(T_lev, p_lev/bar, c='mediumvioletred')
   ax.set_xlabel('Temperature [K]', fontsize=16)
   ax.set_ylabel('pressure [bar]', fontsize=16)
   ax.set_title('Madhusudhan & Seager (2009) T-p Profile', fontsize=14)
   ax.tick_params(labelsize=14)
   ax.invert_yaxis()
   plt.tight_layout()

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_Tp: MandS

   params:
     - { name: a1, dist: uniform, low: -2.0, high: 2.0, transform: logit, init: 0.5 }
     - { name: a2, dist: uniform, low: -2.0, high: 2.0, transform: logit, init: -0.4 }
     - { name: log_10_P1, dist: uniform, low: -4, high: 1, transform: logit, init: -2 }
     - { name: log_10_P2, dist: uniform, low: -4, high: 1, transform: logit, init: -0.5 }
     - { name: log_10_P3, dist: uniform, low: -4, high: 2, transform: logit, init: 1 }
     - { name: T_ref, dist: uniform, low: 500.0, high: 3000.0, transform: logit, init: 800.0 }


Milne
-----

A Milne T-p profile is a classic analytical radiative-equilibrium (RE) solution for a grey opacity atmosphere with only an internal heat flux source.

.. math::

   T^{4}(p) = \frac{3}{4}T_{\rm int}^{4}\left(q(\tau) + \tau\right)

where :math:`T_{\rm int}` [K] is the internal (effective) temperature, :math:`\tau` the vertical optical depth, :math:`q(\tau)` the classic Hopf function solution,
that varies between :math:`q_{\infty} \approx 0.71` at :math:`\tau \rightarrow \infty` and  :math:`q_{0} \approx 1/\sqrt{3}` at :math:`\tau \rightarrow 0`.

Required parameters: :math:`T_{\rm int}` [K], :math:`\log_{10}g` [cm s\ :sup:`-2`], :math:`\log_{10} \kappa_{\rm ir}` [cm\ :sup:`2` g\ :sup:`-1`]

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from exo_skryer.vert_Tp import Milne
   from exo_skryer.data_constants import bar

   # Create pressure grid
   nlev = 100
   p_bot = np.log10(1000.0)
   p_top = np.log10(1e-5)
   p_lev = np.logspace(p_bot, p_top, nlev) * bar

   # Example parameters for a brown dwarf
   params = {
       "T_int": 1000.0,
       "log_10_g": 4.5,
       "log_10_k_ir": -2.0
   }
   T_lev, T_lay = Milne(p_lev, params)

   # Plot
   fig, ax = plt.subplots(figsize=(10, 5))
   ax.semilogy(T_lev, p_lev/bar, c='forestgreen')
   ax.set_xlabel('Temperature [K]', fontsize=16)
   ax.set_ylabel('pressure [bar]', fontsize=16)
   ax.set_title('Milne T-p Profile', fontsize=14)
   ax.tick_params(labelsize=14)
   ax.invert_yaxis()
   plt.tight_layout()

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_Tp: Milne

   params:
     - { name: log_10_g, dist: uniform, low: 4.0, high: 5.5, transform: logit, init: 5.0 }
     - { name: T_int, dist: uniform, low: 500.0, high: 1500.0, transform: logit, init: 1000.0 }
     - { name: log_10_k_ir, dist: uniform, low: -6, high: 6, transform: logit, init: -2 }


Modified Milne
--------------

The "Modified Milne" profile extends the standard Milne solution to one that smoothly varies from a skin temperature at zero optical depth to the classic Milne solution at high optical depth.
It does this by creating a modified Hopf function with a stretched exponential transition following:

.. math::

   T^{4}(p) = \frac{3}{4}T_{\rm int}^{4}\left(q(p) + \tau\right)

   q(p) = q_{\infty} + (q_{0} - q_{\infty}) \exp\left[-\left(\frac{p}{p_{\rm t}}\right)^{\beta}\right]

   q_{0} = \frac{4}{3}\left(\frac{T_{\rm skin}}{T_{\rm int}}\right)^{4}

where :math:`T_{\rm int}` [K] is the internal (effective) temperature, :math:`\tau` the vertical optical depth, :math:`q_{\infty} \approx 0.71`, :math:`\beta` is a stretching exponential parameter, 
which is shallower than exponential for :math:`\beta < 1` and steeper when :math:`\beta > 1`, 
:math:`p_{\rm t}` [bar] the transition region between :math:`q_{\infty}` and :math:`q_{0}` and  :math:`T_{\rm skin}` [K] the skin temperature (temperature at zero optical depth).
This allows much greater flexibility than the classic Milne solution, and the ability to closely mimic the stucture of (cloud free) non-grey radiative-convective-equilibrium (RCE) models.

Required parameters: :math:`T_{\rm int}` [K], :math:`T_{\rm skin}` [K], :math:`\log_{10}g` [cm s\ :sup:`-2`], :math:`\log_{10} \kappa_{\rm ir}` [cm\ :sup:`2` g\ :sup:`-1`], :math:`\log_{10} p_{\rm t}` [bar], :math:`\beta`

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from exo_skryer.vert_Tp import Milne_modified
   from exo_skryer.data_constants import bar

   # Create pressure grid
   nlev = 100
   p_bot = np.log10(1000.0)
   p_top = np.log10(1e-5)
   p_lev = np.logspace(p_bot, p_top, nlev) * bar

   # Example parameters
   params = {
       "T_int": 1000.0,
       "T_skin": 300.0,
       "log_10_g": 4.5,
       "log_10_k_ir": -2.0,
       "log_10_p_t": 0.0,
       "beta": 0.55
   }
   T_lev, T_lay = Milne_modified(p_lev, params)

   # Plot
   fig, ax = plt.subplots(figsize=(10, 5))
   ax.semilogy(T_lev, p_lev/bar, c='dodgerblue')
   ax.set_xlabel('Temperature [K]', fontsize=16)
   ax.set_ylabel('pressure [bar]', fontsize=16)
   ax.set_title('Modified Milne T-p Profile', fontsize=14)
   ax.tick_params(labelsize=14)
   ax.invert_yaxis()
   plt.tight_layout()

**YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_Tp: Milne_modified

   params:
     - { name: log_10_g, dist: uniform, low: 4.0, high: 5.5, transform: logit, init: 5.0 }
     - { name: T_int, dist: uniform, low: 500.0, high: 1500.0, transform: logit, init: 1000.0 }
     - { name: T_skin, dist: uniform, low: 100.0, high: 500.0, transform: logit, init: 300.0 }
     - { name: log_10_k_ir, dist: uniform, low: -6, high: 6, transform: logit, init: -2 }
     - { name: log_10_p_t, dist: uniform, low: -3, high: 2, transform: logit, init: 0.0 }
     - { name: beta, dist: uniform, low: 0.3, high: 1.0, transform: logit, init: 0.55 }



