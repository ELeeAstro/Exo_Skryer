***********************
Chemistry Schemes
***********************

The vertical chemistry routines calculate the volume mixing ratio (VMR) of each species, as defined as the fraction of the local total number density of the atmosphere.

Exo Skryer provides several vertical chemistry calculation functions in the `~exo_skryer.vert_chem` module.


.. math::

  x_{i} = \frac{n_{i}}{n_{\rm tot}}

Several schemes are included in Exo Skryer, from simple constant profiles to chemical equilibrium to quenching timescale approximation.
Throughout, we assume an H\ :sub:`2`-He dominaated atmosphere, forcing the total VMR to satisfy

.. math::

  \sum_{i}x_{i} = 1

Constant VMR
------------

The constant VMR profile assumes a constant value for each species as given by the sampled (or delta) parameter, log_10_f_x.

.. math::

  x_{i} = x_{\rm const} 

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from exo_skryer.vert_Tp import Milne_modified
   from exo_skryer.vert_chem import constant_vmr
   from exo_skryer.data_constants import bar

   nlev = 100
   p_bot = np.log10(100.0)
   p_top = np.log10(1e-4)
   p_lev = np.logspace(p_bot, p_top, nlev) * bar

   params_tp = {
       "T_int": 1200.0,
       "T_skin": 400.0,
       "log_10_g": 4.5,
       "log_10_k_ir": -2.0,
       "log_10_p_t": 0.0,
       "beta": 0.55
   }
   T_lev, T_lay = Milne_modified(p_lev, params_tp)
   p_lay = (p_lev[1:] - p_lev[:-1]) / np.log(p_lev[1:] / p_lev[:-1])

   species_order = ("H2O", "CO", "CO2", "CH4")
   chem_kernel = constant_vmr(species_order)
   params = {
       "log_10_f_H2O": -3.0,
       "log_10_f_CO": -4.0,
       "log_10_f_CO2": -8.0,
       "log_10_f_CH4": -6.0,
   }
   vmr_lay = chem_kernel(p_lay, T_lay, params, nlev - 1)

   fig, ax = plt.subplots(figsize=(10, 5))
   for key in ("H2O", "CO", "CO2", "CH4"):
       if key in vmr_lay:
           ax.semilogy(vmr_lay[key], p_lay / bar, label=key)
   ax.set_xlabel("VMR", fontsize=16)
   ax.set_ylabel("pressure [bar]", fontsize=16)
   ax.set_title("Constant VMR Chemistry", fontsize=14)
   ax.legend(fontsize=10)
   ax.set_xscale('log')
   ax.invert_yaxis()
   plt.tight_layout()

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_chem: constant_vmr

   params:
     - { name: log_10_f_H2O, dist: uniform, low: -9, high: -1, transform: logit, init: -3 }
     - { name: log_10_f_CO, dist: uniform, low: -9, high: -1, transform: logit, init: -4 }
     - { name: log_10_f_CO2, dist: uniform, low: -9, high: -1, transform: logit, init: -8 }
     - { name: log_10_f_CH4, dist: uniform, low: -9, high: -1, transform: logit, init: -6 }


Chemical Equilibrium with rate JAX
----------------------------------

We can also use the semi-analytical chemical equilibrium scheme, Reliable Analytic Thermochemical Equilibrium (rate), from `Cubillos et al (2019) <https://ui.adsabs.harvard.edu/abs/2019ApJ...872..111C/abstract>`_.
This was converted into JAX compabitile python from the origional python code found on `GitHib <https://github.com/pcubillos/rate>`_.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from pathlib import Path
   from exo_skryer.rate_jax import load_gibbs_cache
   from exo_skryer.vert_Tp import Milne_modified
   from exo_skryer.vert_chem import CE_rate_jax
   from exo_skryer.data_constants import bar

   try:
       root = Path(__file__).resolve().parent
   except NameError:
       root = Path.cwd().resolve()
   for _ in range(5):
       if (root / "JANAF_data").is_dir():
           break
       root = root.parent
   load_gibbs_cache(str(root / "JANAF_data"))

   nlev = 100
   p_bot = np.log10(100.0)
   p_top = np.log10(1e-4)
   p_lev = np.logspace(p_bot, p_top, nlev) * bar

   params_tp = {
       "T_int": 1200.0,
       "T_skin": 400.0,
       "log_10_g": 4.5,
       "log_10_k_ir": -2.0,
       "log_10_p_t": 0.0,
       "beta": 0.55
   }
   T_lev, T_lay = Milne_modified(p_lev, params_tp)
   p_lay = (p_lev[1:] - p_lev[:-1]) / np.log(p_lev[1:] / p_lev[:-1])

   params = {"M/H": 0.0, "C/O": 0.55}
   vmr_lay = CE_rate_jax(p_lay, T_lay, params, nlev - 1)

   fig, ax = plt.subplots(figsize=(10, 5))
   for key in ("H2O", "CO", "CH4", "NH3", "HCN", "CO2"):
       if key in vmr_lay:
           ax.semilogy(vmr_lay[key], p_lay / bar, label=key)
   ax.set_xlabel("VMR", fontsize=16)
   ax.set_ylabel("pressure [bar]", fontsize=16)
   ax.set_title("CE Chemistry (RateJAX)", fontsize=14)
   ax.legend(fontsize=10)
   ax.set_xscale('log')
   ax.invert_yaxis()
   plt.tight_layout()

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_chem: CE_rate_jax

   params:
     - { name: M/H, dist: uniform, low: -1.0, high: 2.0, transform: logit, init: 0.0 }
     - { name: C/O, dist: uniform, low: 0.1, high: 1.5, transform: logit, init: 0.55 }


Chemical Equilibrium with FastChem interpolation
------------------------------------------------

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_chem: CE_fastchem_jax


Quenching Timescale Approximation 
-----------------------------------

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from pathlib import Path
   from exo_skryer.rate_jax import load_gibbs_cache
   from exo_skryer.vert_Tp import Milne_modified
   from exo_skryer.vert_chem import quench_approx
   from exo_skryer.data_constants import bar

   try:
       root = Path(__file__).resolve().parent
   except NameError:
       root = Path.cwd().resolve()
   for _ in range(5):
       if (root / "JANAF_data").is_dir():
           break
       root = root.parent
   load_gibbs_cache(str(root / "JANAF_data"))

   nlev = 100
   p_bot = np.log10(100.0)
   p_top = np.log10(1e-4)
   p_lev = np.logspace(p_bot, p_top, nlev) * bar

   params_tp = {
       "T_int": 1200.0,
       "T_skin": 400.0,
       "log_10_g": 4.5,
       "log_10_k_ir": -2.0,
       "log_10_p_t": 0.0,
       "beta": 0.55
   }
   T_lev, T_lay = Milne_modified(p_lev, params_tp)
   p_lay = (p_lev[1:] - p_lev[:-1]) / np.log(p_lev[1:] / p_lev[:-1])

   params = {"M/H": 0.0, "C/O": 0.55, "Kzz": 1e8, "log_10_g": 4.5}
   vmr_lay = quench_approx(p_lay, T_lay, params, nlev - 1)

   fig, ax = plt.subplots(figsize=(10, 5))
   for key in ("H2O", "CO", "CH4", "NH3", "HCN", "CO2"):
       if key in vmr_lay:
           ax.semilogy(vmr_lay[key], p_lay / bar, label=key)
   ax.set_xlabel("VMR", fontsize=16)
   ax.set_ylabel("pressure [bar]", fontsize=16)
   ax.set_title("Quench Approx Chemistry", fontsize=14)
   ax.legend(fontsize=10)
   ax.set_xscale('log')
   ax.invert_yaxis()
   plt.tight_layout()

**Example YAML Configuration:**

.. code-block:: yaml

   physics:
     vert_chem: quench_approx

   params:
     - { name: M/H, dist: uniform, low: -1.0, high: 2.0, transform: logit, init: 0.0 }
     - { name: C/O, dist: uniform, low: 0.1, high: 1.5, transform: logit, init: 0.55 }
     - { name: Kzz, dist: uniform, low: 1e6, high: 1e10, transform: logit, init: 1e8 }
     - { name: log_10_g, dist: uniform, low: 2.0, high: 4.0, transform: logit, init: 3.0 }
  
