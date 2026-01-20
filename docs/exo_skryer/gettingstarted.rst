***************
Getting started
***************

Why Exo Skryer?
---------------

Contemporary retrieval frameworks for sub-stellar atmospheres have numerous requirements to fulfill to be useful tools for analysis and scientific discovery:

* Robust parameter estimation
* Statistically consistent Bayesian evidence comparison
* Fast completion times
* Flexibility and scalability
* Physically interpretable results

In the era of JWST where medium resolution, large wavelength range coverage observational data is available, 
retrieval results have now become a standard methodology to publish alongside new observational data.
Retrieval models for high spectral resolution data are also being developed, which will become boosted by the near-future operation of the ELT.

These demands have pushed the computational burden of retrieval models
Exo Skryer attempts to solve these computational issues without the using excessive amounts of high performance computing (HPC) power.
Exo Skryer uses the JAX extension to Python to accelerate both the sampling and forward model evaluations, enabling efficient, scalable operation on CPUs and GPUs.
This enables complex retrieval modelling to be performed in good time on desktop computers with a GPU.


The current version of Exo Skryer offers the following nested sampling options:

* JAXNS -  
* pymultinest - 
* dynesty - 

(experimental) 

* ultranest - 
* blackjax-ns -

As well as two NUTS MCMC samplers:

* numpyro - 
* blackjax - 

Your first model
----------------

The experiments/HD189_Barstow_2020_trans_setup provides a first taste of how to use Exo Skryer, performing a first retrieval model, as well as postprocessing, testing individual functions and other things.


References
----------