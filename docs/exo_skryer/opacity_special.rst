***********************
Special Opacities
***********************

Special opacities are continuum sources that are not handled by the line, CIA,
or Rayleigh registries. In the current codebase this is the H⁻ (H-minus)
continuum, split into:

* bound-free: :py:func:`exo_skryer.opacity_special.compute_hminus_bf_opacity`
* free-free: :py:func:`exo_skryer.opacity_special.compute_hminus_ff_opacity`

The forward model uses :py:func:`exo_skryer.opacity_special.compute_special_opacity`
to sum all enabled special-opacity contributions.

Enabling H⁻ in YAML
-------------------

Two separate switches are involved:

1) **Enable special opacity in the forward model** (kernel runs):

.. code-block:: yaml

   physics:
     opac_special: ck   # or lbl, or on

2) **Enable which H⁻ tables are built/loaded** (registry contents):

.. code-block:: yaml

   opac:
     special:
       - {species: H-, bf: true, ff: true}

Requirements
------------

H⁻ bound-free (bf)
  Requires the VMR key ``"H-"`` in ``state["vmr_lay"]``. In constant-VMR chemistry
  this typically means including ``H-`` as a retrieved trace species
  (e.g. ``log_10_f_H-`` / ``clr_H-`` depending on chemistry mode).

H⁻ free-free (ff)
  Requires:

  * atomic hydrogen VMR key ``"H"`` in ``state["vmr_lay"]`` (for constant-VMR modes,
    add parameter ``log_10_H_over_H2`` so ``H`` can be derived from the H2+He filler)
  * parameter ``log_10_ne_over_ntot`` in the retrieval parameter list (proxy for
    electron fraction).
