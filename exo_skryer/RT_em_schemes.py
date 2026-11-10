"""
RT_em_schemes.py
================

Emission radiative transfer solvers, selected with ``physics.em_scheme``:

- ``eaa`` (:func:`~exo_skryer.RT_em_eaa.solve_alpha_eaa`): extended absorption
  approximation, linear in tau (Exo-FMS lw_AA_L). Fastest; approximate scattering.
- ``toon89`` (:func:`~exo_skryer.RT_em_toon89.solve_toon89`): Toon et al. (1989)
  two-stream source-function method (Exo-FMS lw_Toon89).
- ``dda`` (:func:`~exo_skryer.RT_em_dda.solve_dda`): four-stream discrete
  ordinates adding method (Zhang et al. 2013; Exo-FMS lw_DDA). Most accurate
  for multiple scattering.
- ``ad`` (:func:`~exo_skryer.RT_em_ad.solve_ad`): adding-doubling matrix operator
  method, 8 streams (Exo-FMS lw_AD). Accurate multiple scattering for any w and g;
  the most expensive.
- ``sc_ric`` (:func:`~exo_skryer.RT_em_sc_ric.solve_sc_ric`): linear short
  characteristics with multiple scattering, Riccati solution, 8 streams, with
  source function technique along the Hogan (2024) angles (Exo-FMS lw_sc_scat_ric).
  Under-resolves optically thick, near-conservative layers.

All solvers share the signature
``(be_levels, dtau_layers, ssa, g_phase, be_internal, return_layer_contrib) -> (olr, layer_contrib)``
with bottom-first inputs, and return only the upward TOA flux (OLR) needed for
emission spectra.
"""

from __future__ import annotations

from .RT_em_eaa import solve_alpha_eaa
from .RT_em_toon89 import solve_toon89
from .RT_em_dda import solve_dda
from .RT_em_ad import solve_ad
from .RT_em_sc_ric import solve_sc_ric

__all__ = ["solve_alpha_eaa", "solve_toon89", "solve_dda", "solve_ad", "solve_sc_ric", "get_emission_solver"]

solve_toon89_picaso = solve_toon89  # former name


def get_emission_solver(name: str):
    """Get emission RT solver function by name.

    Parameters
    ----------
    name : str
        Scheme name. Supported values:
        - "eaa", "alpha_eaa": Extended absorption approximation (linear in tau)
        - "toon89", "toon89_picaso": Toon et al. (1989) two-stream source-function method
        - "dda", "4dda": Four-stream discrete ordinates adding method
        - "ad", "lw_ad", "adding_doubling": Adding-doubling method, 8 streams
        - "sc_ric", "sc_scat_ric", "lw_sc_scat_ric": Riccati short characteristics, 8 streams

    Returns
    -------
    solver : callable
        Emission solver function with signature:
        (be_levels, dtau_layers, ssa, g_phase, be_internal, return_layer_contrib)
        -> (olr, layer_contrib_flux)

    Raises
    ------
    NotImplementedError
        If the scheme name is not recognized.
    """
    name = str(name).lower().strip()
    if name in ("eaa", "alpha_eaa"):
        return solve_alpha_eaa
    elif name in ("toon89", "toon89_picaso"):
        return solve_toon89
    elif name in ("dda", "4dda"):
        return solve_dda
    elif name in ("ad", "lw_ad", "adding_doubling"):
        return solve_ad
    elif name in ("sc_ric", "sc_scat_ric", "lw_sc_scat_ric"):
        return solve_sc_ric
    raise NotImplementedError(f"Unknown emission scheme '{name}'")
