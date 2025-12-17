# rate_jax.py
from __future__ import annotations
from typing import Dict, Mapping, Tuple, Union, Optional
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optimistix as optx

# Gas constant [J/(mol·K)]
R_GAS = 8.314462618

# ============================================================================
# Global cache for Gibbs free energy tables
# ============================================================================
_GIBBS_CACHE: Optional["GibbsTableJAX"] = None

Species = Tuple[str, ...]

__all__ = [
    "GibbsTableJAX",
    "load_gibbs_cache",
    "get_gibbs_cache",
    "clear_gibbs_cache",
    "is_gibbs_cache_loaded",
]


class GibbsTableJAX:
    """
    JAX-friendly Gibbs free energy interpolator.

    Expect data as:
      data[spec] = {
        "T": jnp.array([...]),          # temperature grid [K]
        "g_over_R": jnp.array([...]),   # G/R on that grid
        "heat_over_R298": float,        # (1000 * H_298.15 / R)
      }

    This avoids SciPy splines and uses jax.numpy.interp instead.
    """
    def __init__(self, data: Mapping[str, Mapping[str, jnp.ndarray]]):
        self.data = data

    def g_rt(self, spec: str, T: jnp.ndarray) -> jnp.ndarray:
        """
        Dimensionless Gibbs combination similar to your original gRT.eval:
        g_RT = -G/R + (H_298/R) / T
        """
        d = self.data[spec]
        T_grid      = d["T"]
        g_over_R    = d["g_over_R"]
        heat_over_R = d["heat_over_R298"]   # already in units of (H/R)

        g_over_R_T = jnp.interp(T, T_grid, g_over_R)
        return -g_over_R_T + heat_over_R / T


# ============================================================================
# Global cache management functions
# ============================================================================

def load_gibbs_cache(janaf_dir: str) -> GibbsTableJAX:
    """
    Load JANAF thermodynamic tables into global cache.

    This function should be called once during initialization (e.g., in
    run_retrieval.py) to load and cache the Gibbs free energy data before
    running forward models or retrievals.

    Parameters
    ----------
    janaf_dir : str
        Directory containing JANAF table files

    Returns
    -------
    gibbs : GibbsTableJAX
        The loaded Gibbs table (also cached globally)

    Notes
    -----
    Expected file format:
      - Each file: 3 columns -> T, G, H
      - Filename: "<MOLNAME>_*.txt" (e.g., "H2O_janaf.txt")
      - Units: T [K], G [J/mol], H [kJ/mol]

    Examples
    --------
    >>> # In run_retrieval.py or similar initialization:
    >>> from rate_jax import load_gibbs_cache
    >>> gibbs = load_gibbs_cache("JANAF_data/")
    """
    global _GIBBS_CACHE

    data: Dict[str, Dict[str, jnp.ndarray]] = {}

    for fname in os.listdir(janaf_dir):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(janaf_dir, fname)
        molname = fname.split("_")[0]  # "H2O_foo.txt" -> "H2O"

        T, G, H = np.loadtxt(path, unpack=True)

        # Sort by temperature
        order = np.argsort(T)
        T = T[order]
        G = G[order]
        H = H[order]

        # Find value closest to 298.15 K
        idx_298 = np.argmin(np.abs(T - 298.15))
        heat_over_R298 = 1000.0 * H[idx_298] / R_GAS  # H in kJ/mol → J/mol, /R

        data[molname] = {
            "T": jnp.asarray(T),
            "g_over_R": jnp.asarray(G / R_GAS),
            "heat_over_R298": float(heat_over_R298),
        }

    _GIBBS_CACHE = GibbsTableJAX(data)
    return _GIBBS_CACHE


def get_gibbs_cache() -> Optional[GibbsTableJAX]:
    """
    Get the cached Gibbs free energy table.

    Returns
    -------
    gibbs : GibbsTableJAX or None
        The cached Gibbs table, or None if not yet loaded

    Raises
    ------
    RuntimeError
        If cache has not been initialized with load_gibbs_cache()
    """
    if _GIBBS_CACHE is None:
        raise RuntimeError(
            "Gibbs cache not initialized. Call load_gibbs_cache() first."
        )
    return _GIBBS_CACHE


def clear_gibbs_cache() -> None:
    """
    Clear the cached Gibbs free energy table.

    Useful for freeing memory or reloading with different data.
    """
    global _GIBBS_CACHE
    _GIBBS_CACHE = None


def is_gibbs_cache_loaded() -> bool:
    """
    Check if Gibbs cache is loaded.

    Returns
    -------
    loaded : bool
        True if cache is loaded, False otherwise
    """
    return _GIBBS_CACHE is not None


class RateJAX:
    """
    JAX-friendly version of RATE:
    - Works on vector T, p.
    - Returns VMR dictionary.
    """

    def __init__(
        self,
        gibbs: GibbsTableJAX,
        C: float = 2.5e-4,
        N: float = 1.0e-4,
        O: float = 5.0e-4,
        fHe: float = 0.0,
    ):
        self.gibbs = gibbs
        # Keep as JAX arrays for JIT compatibility (don't convert to Python float)
        self.C = C
        self.N = N
        self.O = O
        self.fHe = fHe

        self.species: Species = (
            "H2O", "CH4", "CO", "CO2", "NH3",
            "C2H2", "C2H4", "HCN", "N2",
            "H2", "H", "He",
        )

    # ---------- Gibbs wrappers ----------

    def grt(self, spec: str, T: jnp.ndarray) -> jnp.ndarray:
        return self.gibbs.g_rt(spec, T)

    # ---------- Equilibrium constants k' ----------

    def kprime0(self, T: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """
        Equilibrium constant for hydrogen dissociation: H2 ↔ 2H

        K'₀ = exp(-ΔG/RT) / p
        where ΔG = 2·G(H) - G(H₂)

        Parameters
        ----------
        T : array
            Temperature [K]
        p : array
            Pressure [bar]

        Returns
        -------
        K'₀ : array
            Modified equilibrium constant [bar⁻¹]
        """
        return jnp.exp(-(2.0 * self.grt("H", T) - self.grt("H2", T))) / p

    def kprime1(self, T: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """
        Equilibrium constant for methane-water reaction: CH₄ + H₂O ↔ CO + 3H₂

        K'₁ = exp(-ΔG/RT) / p²
        where ΔG = G(CO) + 3·G(H₂) - G(CH₄) - G(H₂O)

        This is the key reaction controlling the C/O ratio in hot atmospheres.

        Parameters
        ----------
        T : array
            Temperature [K]
        p : array
            Pressure [bar]

        Returns
        -------
        K'₁ : array
            Modified equilibrium constant [bar⁻²]
        """
        return jnp.exp(
            -(
                self.grt("CO", T) + 3.0 * self.grt("H2", T)
                - self.grt("CH4", T) - self.grt("H2O", T)
            )
        ) / p**2

    def kprime2(self, T: jnp.ndarray) -> jnp.ndarray:
        """
        Equilibrium constant for carbon dioxide reduction: CO₂ + H₂ ↔ CO + H₂O

        K'₂ = exp(-ΔG/RT)
        where ΔG = G(CO) + G(H₂O) - G(CO₂) - G(H₂)

        Parameters
        ----------
        T : array
            Temperature [K]

        Returns
        -------
        K'₂ : array
            Equilibrium constant [dimensionless]
        """
        return jnp.exp(
            -(
                self.grt("CO", T) + self.grt("H2O", T)
                - self.grt("CO2", T) - self.grt("H2", T)
            )
        )

    def kprime3(self, T: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """
        Equilibrium constant for acetylene formation: 2CH₄ ↔ C₂H₂ + 3H₂

        K'₃ = exp(-ΔG/RT) / p²
        where ΔG = G(C₂H₂) + 3·G(H₂) - 2·G(CH₄)

        Important for high-C/O and high-temperature atmospheres.

        Parameters
        ----------
        T : array
            Temperature [K]
        p : array
            Pressure [bar]

        Returns
        -------
        K'₃ : array
            Modified equilibrium constant [bar⁻²]
        """
        return jnp.exp(
            -(
                self.grt("C2H2", T) + 3.0 * self.grt("H2", T)
                - 2.0 * self.grt("CH4", T)
            )
        ) / p**2

    def kprime4(self, T: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """
        Equilibrium constant for ethylene-acetylene: C₂H₄ ↔ C₂H₂ + H₂

        K'₄ = exp(-ΔG/RT) / p
        where ΔG = G(C₂H₂) + G(H₂) - G(C₂H₄)

        Parameters
        ----------
        T : array
            Temperature [K]
        p : array
            Pressure [bar]

        Returns
        -------
        K'₄ : array
            Modified equilibrium constant [bar⁻¹]
        """
        return jnp.exp(
            -(
                self.grt("C2H2", T) + self.grt("H2", T)
                - self.grt("C2H4", T)
            )
        ) / p

    def kprime5(self, T: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """
        Equilibrium constant for ammonia dissociation: 2NH₃ ↔ N₂ + 3H₂

        K'₅ = exp(-ΔG/RT) / p²
        where ΔG = G(N₂) + 3·G(H₂) - 2·G(NH₃)

        Dominant nitrogen chemistry reaction in hot atmospheres.

        Parameters
        ----------
        T : array
            Temperature [K]
        p : array
            Pressure [bar]

        Returns
        -------
        K'₅ : array
            Modified equilibrium constant [bar⁻²]
        """
        return jnp.exp(
            -(
                self.grt("N2", T) + 3.0 * self.grt("H2", T)
                - 2.0 * self.grt("NH3", T)
            )
        ) / p**2

    def kprime6(self, T: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """
        Equilibrium constant for HCN formation: NH₃ + CH₄ ↔ HCN + 3H₂

        K'₆ = exp(-ΔG/RT) / p²
        where ΔG = G(HCN) + 3·G(H₂) - G(NH₃) - G(CH₄)

        Important when both N and C are abundant at high temperatures.

        Parameters
        ----------
        T : array
            Temperature [K]
        p : array
            Pressure [bar]

        Returns
        -------
        K'₆ : array
            Modified equilibrium constant [bar⁻²]
        """
        return jnp.exp(
            -(
                self.grt("HCN", T) + 3.0 * self.grt("H2", T)
                - self.grt("NH3", T) - self.grt("CH4", T)
            )
        ) / p**2

    # ---------- Turnover pressure (CO vs H2O dominated) ----------

    @staticmethod
    def top(T: jnp.ndarray, C: float, N: float, O: float) -> jnp.ndarray:
        """
        Turnover pressure: transition between CO-dominated and H2O-dominated chemistry.

        Computes the pressure where CO and H2O abundances become comparable,
        based on a polynomial fit to thermochemical equilibrium calculations
        (Lodders & Fegley 2002).

        Parameters
        ----------
        T : array
            Temperature [K]
        C : float
            Carbon elemental abundance (number ratio relative to H2)
        N : float
            Nitrogen elemental abundance (number ratio relative to H2)
        O : float
            Oxygen elemental abundance (number ratio relative to H2)

        Returns
        -------
        P_turnover : array
            Turnover pressure [bar], where CO/H2O ~ 1
        """
        # Polynomial coefficients for log10(P_turnover)
        # Fit to (log T, log C, log N, log O) → log10(P_turnover)
        # Each variable has powers 1-4, plus constant term
        coeffs = jnp.array([
            -1.07028658e+03,  # constant
            # Temperature terms (powers 1-4)
             1.20815018e+03, -5.21868655e+02,  1.02459233e+02, -7.68350388e+00,
            # Carbon terms (powers 1-4)
             1.30787500e+00,  3.18619604e-01,  5.32918135e-02,  3.12269845e-03,
            # Nitrogen terms (powers 1-4)
             2.81238906e-02,  1.26015039e-02,  2.07616221e-03,  1.16038224e-04,
            # Oxygen terms (powers 1-4)
            -1.69589064e-01, -5.21662503e-02, -7.33669631e-03, -3.74492912e-04
        ])

        # Compute log10 of input variables
        logT = jnp.log10(T)
        logC = jnp.log10(C)
        logN = jnp.log10(N)
        logO = jnp.log10(O)

        # Build polynomial: constant + sum over (var, power) pairs
        log10_P_turn = coeffs[0]  # constant term

        # Temperature terms (powers 1-4)
        idx = 1
        for power in range(1, 5):
            log10_P_turn += coeffs[idx] * logT**power
            idx += 1

        # Carbon terms (powers 1-4)
        for power in range(1, 5):
            log10_P_turn += coeffs[idx] * logC**power
            idx += 1

        # Nitrogen terms (powers 1-4)
        for power in range(1, 5):
            log10_P_turn += coeffs[idx] * logN**power
            idx += 1

        # Oxygen terms (powers 1-4)
        for power in range(1, 5):
            log10_P_turn += coeffs[idx] * logO**power
            idx += 1

        # Clip to valid pressure range: 10^-8 to 10^3 bar
        log10_P_turn = jnp.clip(log10_P_turn, -8.0001, 3.0001)

        return 10.0 ** log10_P_turn

    # ---------- Polynomial builders (example + pattern) ----------

    def HCO_poly6_CO(self, f, k1, k2, k3, k4):
        """
        HCO chemistry, polynomial in CO.
        Now returns 7 coefficients (last one is 0.0) for JAX compatibility.
        """
        C, O = self.C, self.O
        A0 = -C * O**2 * f**3 * k1**2 * k2**3 * k4
        A1 = (
            -C * O**2 * f**3 * k1**2 * k2**2 * k4
            + 2 * C * O * f**2 * k1**2 * k2**3 * k4
            + O**3 * f**3 * k1**2 * k2**2 * k4
            + O**2 * f**2 * k1**2 * k2**3 * k4
            + O * f * k1 * k2**3 * k4
        )
        A2 = (
            2 * C * O * f**2 * k1**2 * k2**2 * k4
            - C * f * k1**2 * k2**3 * k4
            - 2 * O**2 * f**2 * k1**2 * k2**2 * k4
            - 2 * O * f * k1**2 * k2**3 * k4
            + 2 * O * f * k1 * k2**2 * k4
            - k1 * k2**3 * k4
            + 2 * k2**3 * k3 * k4
            + 2 * k2**3 * k3
        )
        A3 = (
            -C * f * k1**2 * k2**2 * k4
            + O * f * k1**2 * k2**2 * k4
            + O * f * k1 * k2 * k4
            + k1**2 * k2**3 * k4
            - 2 * k1 * k2**2 * k4
            + 6 * k2**2 * k3 * k4
            + 6 * k2**2 * k3
        )
        A4 = -k1 * k2 * k4 + 6 * k2 * k3 * k4 + 6 * k2 * k3
        A5 = 2 * k3 * k4 + 2 * k3
        A6 = 0.0  # pad to degree-6 polynomial

        return jnp.array([A0, A1, A2, A3, A4, A5, A6])

    def HCO_poly6_H2O(self, f, k1, k2, k3, k4):
        """
        HCO chemistry, polynomial in H2O.
        Now returns 7 coefficients (last one is 0.0) for JAX compatibility.
        """
        C, O = self.C, self.O
        A0 = 2 * O**2 * f**2 * k2**2 * k3 * k4 + 2 * O**2 * f**2 * k2**2 * k3
        A1 = O * f * k1 * k2**2 * k4 - 4 * O * f * k2**2 * k3 * k4 - 4 * O * f * k2**2 * k3
        A2 = (
            -C * f * k1**2 * k2**2 * k4
            + O * f * k1**2 * k2**2 * k4
            + O * f * k1 * k2 * k4
            - k1 * k2**2 * k4
            + 2 * k2**2 * k3 * k4
            + 2 * k2**2 * k3
        )
        A3 = (
            -2 * C * f * k1**2 * k2 * k4
            + 2 * O * f * k1**2 * k2 * k4
            - k1**2 * k2**2 * k4
            - k1 * k2 * k4
        )
        A4 = -C * f * k1**2 * k4 + O * f * k1**2 * k4 - 2 * k1**2 * k2 * k4
        A5 = -k1**2 * k4
        A6 = 0.0  # pad to degree-6 polynomial

        return jnp.array([A0, A1, A2, A3, A4, A5, A6])


    # ---------- HCNO, polynomial in CO ----------

    def HCNO_poly8_CO(self, f, k1, k2, k3, k4, k5, k6):
        """
        JAX version of original HCNO_poly8_CO (CO is the root variable).
        """
        C, N, O = self.C, self.N, self.O

        A0 = 2 * C**2 * O**4 * f**6 * k1**4 * k4**2 * k5

        A1 = -C * O**3 * f**4 * k1**3 * k4**2 * (
            8 * C * f * k1 * k5 + 4 * O * f * k1 * k5 + 4 * k5 - k6
        )

        A2 = (
            O**2 * f**2 * k1**2 * k4 * (
                12 * C**2 * f**2 * k1**2 * k4 * k5
                + 16 * C * O * f**2 * k1**2 * k4 * k5
                + 12 * C * f * k1 * k4 * k5
                - 3 * C * f * k1 * k4 * k6
                - 8 * C * f * k3 * k4 * k5
                - 8 * C * f * k3 * k5
                + C * f * k4 * k6**2
                - N * f * k4 * k6**2
                + 2 * O**2 * f**2 * k1**2 * k4 * k5
                + 4 * O * f * k1 * k4 * k5
                - O * f * k1 * k4 * k6
                + 2 * k4 * k5
                - k4 * k6
            )
        )

        A3 = -O * f * k1 * k4 * (
            8 * C**2 * f**2 * k1**3 * k4 * k5
            + 24 * C * O * f**2 * k1**3 * k4 * k5
            + 12 * C * f * k1**2 * k4 * k5
            - 3 * C * f * k1**2 * k4 * k6
            - 16 * C * f * k1 * k3 * k4 * k5
            - 16 * C * f * k1 * k3 * k5
            + 2 * C * f * k1 * k4 * k6**2
            - 2 * N * f * k1 * k4 * k6**2
            + 8 * O**2 * f**2 * k1**3 * k4 * k5
            + 12 * O * f * k1**2 * k4 * k5
            - 3 * O * f * k1**2 * k4 * k6
            - 8 * O * f * k1 * k3 * k4 * k5
            - 8 * O * f * k1 * k3 * k5
            + O * f * k1 * k4 * k6**2
            + 4 * k1 * k4 * k5
            - 2 * k1 * k4 * k6
            - 8 * k3 * k4 * k5
            + 2 * k3 * k4 * k6
            - 8 * k3 * k5
            + 2 * k3 * k6
            + k4 * k6**2
        )

        A4 = (
            2 * C**2 * f**2 * k1**4 * k4**2 * k5
            + 16 * C * O * f**2 * k1**4 * k4**2 * k5
            + 4 * C * f * k1**3 * k4**2 * k5
            - C * f * k1**3 * k4**2 * k6
            - 8 * C * f * k1**2 * k3 * k4**2 * k5
            - 8 * C * f * k1**2 * k3 * k4 * k5
            + C * f * k1**2 * k4**2 * k6**2
            - N * f * k1**2 * k4**2 * k6**2
            + 12 * O**2 * f**2 * k1**4 * k4**2 * k5
            + 12 * O * f * k1**3 * k4**2 * k5
            - 3 * O * f * k1**3 * k4**2 * k6
            - 16 * O * f * k1**2 * k3 * k4**2 * k5
            - 16 * O * f * k1**2 * k3 * k4 * k5
            + 2 * O * f * k1**2 * k4**2 * k6**2
            + 2 * k1**2 * k4**2 * k5
            - k1**2 * k4**2 * k6
            - 8 * k1 * k3 * k4**2 * k5
            + 2 * k1 * k3 * k4**2 * k6
            - 8 * k1 * k3 * k4 * k5
            + 2 * k1 * k3 * k4 * k6
            + k1 * k4**2 * k6**2
            + 8 * k3**2 * k4**2 * k5
            + 16 * k3**2 * k4 * k5
            + 8 * k3**2 * k5
            - 2 * k3 * k4**2 * k6**2
            - 2 * k3 * k4 * k6**2
        )

        A5 = -k1**2 * k4 * (
            4 * C * f * k1**2 * k4 * k5
            + 8 * O * f * k1**2 * k4 * k5
            + 4 * k1 * k4 * k5
            - k1 * k4 * k6
            - 8 * k3 * k4 * k5
            - 8 * k3 * k5
            + k4 * k6**2
        )

        A6 = 2 * k1**4 * k4**2 * k5

        return jnp.array([A0, A1, A2, A3, A4, A5, A6])

    # ---------- HCNO, polynomial in H2O ----------

    def HCNO_poly8_H2O(self, f, k1, k2, k3, k4, k5, k6):
        """
        JAX version of original HCNO_poly8_H2O (H2O is the root variable).
        """
        C, N, O = self.C, self.N, self.O

        A0 = 2 * O**4 * f**4 * k3 * (k4 + 1.0) * (4 * k3 * k4 * k5 + 4 * k3 * k5 - k4 * k6**2)

        A1 = O**3 * f**3 * (
            8 * k1 * k3 * k4**2 * k5
            - 2 * k1 * k3 * k4**2 * k6
            + 8 * k1 * k3 * k4 * k5
            - 2 * k1 * k3 * k4 * k6
            - k1 * k4**2 * k6**2
            - 32 * k3**2 * k4**2 * k5
            - 64 * k3**2 * k4 * k5
            - 32 * k3**2 * k5
            + 8 * k3 * k4**2 * k6**2
            + 8 * k3 * k4 * k6**2
        )

        A2 = -O**2 * f**2 * (
            8 * C * f * k1**2 * k3 * k4**2 * k5
            + 8 * C * f * k1**2 * k3 * k4 * k5
            - C * f * k1**2 * k4**2 * k6**2
            + N * f * k1**2 * k4**2 * k6**2
            - 8 * O * f * k1**2 * k3 * k4**2 * k5
            - 8 * O * f * k1**2 * k3 * k4 * k5
            + O * f * k1**2 * k4**2 * k6**2
            - 2 * k1**2 * k4**2 * k5
            + k1**2 * k4**2 * k6
            + 24 * k1 * k3 * k4**2 * k5
            - 6 * k1 * k3 * k4**2 * k6
            + 24 * k1 * k3 * k4 * k5
            - 6 * k1 * k3 * k4 * k6
            - 3 * k1 * k4**2 * k6**2
            - 48 * k3**2 * k4**2 * k5
            - 96 * k3**2 * k4 * k5
            - 48 * k3**2 * k5
            + 12 * k3 * k4**2 * k6**2
            + 12 * k3 * k4 * k6**2
        )

        A3 = -O * f * (
            4 * C * f * k1**3 * k4**2 * k5
            - C * f * k1**3 * k4**2 * k6
            - 16 * C * f * k1**2 * k3 * k4**2 * k5
            - 16 * C * f * k1**2 * k3 * k4 * k5
            + 2 * C * f * k1**2 * k4**2 * k6**2
            - 2 * N * f * k1**2 * k4**2 * k6**2
            - 4 * O * f * k1**3 * k4**2 * k5
            + O * f * k1**3 * k4**2 * k6
            + 24 * O * f * k1**2 * k3 * k4**2 * k5
            + 24 * O * f * k1**2 * k3 * k4 * k5
            - 3 * O * f * k1**2 * k4**2 * k6**2
            + 4 * k1**2 * k4**2 * k5
            - 2 * k1**2 * k4**2 * k6
            - 24 * k1 * k3 * k4**2 * k5
            + 6 * k1 * k3 * k4**2 * k6
            - 24 * k1 * k3 * k4 * k5
            + 6 * k1 * k3 * k4 * k6
            + 3 * k1 * k4**2 * k6**2
            + 32 * k3**2 * k4**2 * k5
            + 64 * k3**2 * k4 * k5
            + 32 * k3**2 * k5
            - 8 * k3 * k4**2 * k6**2
            - 8 * k3 * k4 * k6**2
        )

        A4 = (
            2 * C**2 * f**2 * k1**4 * k4**2 * k5
            - 4 * C * O * f**2 * k1**4 * k4**2 * k5
            + 4 * C * f * k1**3 * k4**2 * k5
            - C * f * k1**3 * k4**2 * k6
            - 8 * C * f * k1**2 * k3 * k4**2 * k5
            - 8 * C * f * k1**2 * k3 * k4 * k5
            + C * f * k1**2 * k4**2 * k6**2
            - N * f * k1**2 * k4**2 * k6**2
            + 2 * O**2 * f**2 * k1**4 * k4**2 * k5
            - 8 * O * f * k1**3 * k4**2 * k5
            + 2 * O * f * k1**3 * k4**2 * k6
            + 24 * O * f * k1**2 * k3 * k4**2 * k5
            + 24 * O * f * k1**2 * k3 * k4 * k5
            - 3 * O * f * k1**2 * k4**2 * k6**2
            + 2 * k1**2 * k4**2 * k5
            - k1**2 * k4**2 * k6
            - 8 * k1 * k3 * k4**2 * k5
            + 2 * k1 * k3 * k4**2 * k6
            - 8 * k1 * k3 * k4 * k5
            + 2 * k1 * k3 * k4 * k6
            + k1 * k4**2 * k6**2
            + 8 * k3**2 * k4**2 * k5
            + 16 * k3**2 * k4 * k5
            + 8 * k3**2 * k5
            - 2 * k3 * k4**2 * k6**2
            - 2 * k3 * k4 * k6**2
        )

        A5 = k1**2 * k4 * (
            4 * C * f * k1**2 * k4 * k5
            - 4 * O * f * k1**2 * k4 * k5
            + 4 * k1 * k4 * k5
            - k1 * k4 * k6
            - 8 * k3 * k4 * k5
            - 8 * k3 * k5
            + k4 * k6**2
        )

        A6 = 2 * k1**4 * k4**2 * k5

        return jnp.array([A0, A1, A2, A3, A4, A5, A6])

    # ---------- Newton–Raphson (bounded) using Optimistix ----------

    @staticmethod
    def _eval_poly(A: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate polynomial at x.

        Parameters
        ----------
        A : array, shape (n,)
            Polynomial coefficients, A[0] = constant, A[-1] = highest degree
        x : scalar array
            Point at which to evaluate polynomial

        Returns
        -------
        p(x) : scalar array
            Polynomial value at x
        """
        return jnp.polyval(A[::-1], x)  # polyval expects highest degree first

    @classmethod
    def newton_raphson_bounded(
        cls,
        A: jnp.ndarray,
        guess: float,
        vmax: float,
        xtol: float = 1e-8,
        imax: int = 50,
        kmax: int = 10,
    ) -> jnp.ndarray:
        """
        Robust polynomial root finding with bounded domain using Optimistix.

        Uses Newton's method with automatic differentiation. Tries multiple
        initial guesses with decreasing scales if needed, then clamps result
        to [0, vmax].

        Parameters
        ----------
        A : array, shape (n,)
            Polynomial coefficients (constant to highest degree)
        guess : float
            Initial guess for root
        vmax : float
            Maximum valid value for root
        xtol : float
            Relative/absolute tolerance for convergence
        imax : int
            Maximum iterations per attempt
        kmax : int
            Maximum number of retry attempts with scaled guesses

        Returns
        -------
        root : scalar array
            Root of polynomial, clamped to [0, vmax]
        """

        def poly_fn(x, _args):
            """Function to find root of: polynomial(x) = 0"""
            return cls._eval_poly(A, x)

        def try_solve(guess_scaled: float) -> jnp.ndarray:
            """Try to solve from a given initial guess."""
            solver = optx.Newton(rtol=xtol, atol=xtol)
            sol = optx.root_find(
                poly_fn,
                solver,
                y0=guess_scaled,
                args=None,
                max_steps=imax,
                throw=False,  # Don't raise on convergence failure
            )
            return sol.value

        # Try multiple initial guesses with decreasing scales
        def attempt_body(carry):
            root, attempt = carry
            scale = 10.0 ** (-attempt)
            guess_scaled = guess * scale
            root_candidate = try_solve(guess_scaled)
            return (root_candidate, attempt + 1)

        def attempt_cond(carry):
            root, attempt = carry
            finite = jnp.isfinite(root)
            in_range = jnp.logical_and(root >= 0.0, root <= vmax)
            good = jnp.logical_and(finite, in_range)
            more_attempts = attempt < kmax
            # keep trying while root is bad and we still have attempts left
            return jnp.logical_and(~good, more_attempts)

        # Start with bogus root so we always do at least one attempt
        carry0 = (jnp.array(-1.0), jnp.int32(0))
        root_final, _ = lax.while_loop(attempt_cond, attempt_body, carry0)

        # Final clamp as a safety net
        root_final = jnp.clip(root_final, 0.0, vmax)
        return root_final



    # ---------- Rest of species from H2O & CO ----------

    def solve_rest(
        self,
        H2O: float,
        CO: float,
        f: float,
        k1: float,
        k2: float,
        k3: float,
        k4: float,
        k5: float,
        k6: float,
    ) -> jnp.ndarray:
        """
        JAX version of solve_rest for a single layer.
        Returns [H2O, CH4, CO, CO2, NH3, C2H2, C2H4, HCN, N2]
        """
        eps = 1e-300

        k1_safe = k1 + eps
        k2_safe = k2 + eps
        k4_safe = k4 + eps
        k5_safe = k5 + eps

        H2O_safe = H2O + eps

        CH4  = CO / (k1_safe * H2O_safe)
        CO2  = CO * H2O_safe / k2_safe
        C2H2 = k3 * CH4**2
        C2H4 = C2H2 / k4_safe

        # Quadratic for NH3:
        b = 1.0 + k6 * CH4
        disc = b**2 + 8.0 * f * k5_safe * self.N
        NH3 = (jnp.sqrt(disc) - b) / (4.0 * k5_safe)

        # Use approximation when 8 f k5 N / b^2 << 1:
        small_param = 8.0 * f * k5_safe * self.N / (b**2 + 1e-30)
        NH3_approx  = f * self.N
        NH3 = jnp.where(small_param < 1e-6, NH3_approx, NH3)

        HCN = k6 * NH3 * CH4
        N2  = k5_safe * NH3**2

        return jnp.array([H2O, CH4, CO, CO2, NH3, C2H2, C2H4, HCN, N2])


    # ---------- Choose polynomial index (0..3) per layer ----------

    def _choose_poly_index(self, T_i: float, p_i: float):
        """
        Encodes your logic:

          0 -> HCO_poly6_CO
          1 -> HCO_poly6_H2O
          2 -> HCNO_poly8_CO
          3 -> HCNO_poly8_H2O

        Implemented with lax.cond so it works under jit.
        Returns a JAX int32 scalar.
        """
        C, N, O = self.C, self.N, self.O
        C_over_O = C / O
        N_over_C = N / C

        def branch_CO_lt1(_):
            # C/O < 1
            cond_N_hot = jnp.logical_and(N_over_C > 10.0, T_i > 2200.0)

            def when_N_hot(_):
                def when_C_over_O_mid(_):
                    return jnp.int32(3)   # HCNO H2O
                def when_C_over_O_low(_):
                    return jnp.int32(2)   # HCNO CO
                return lax.cond(C_over_O > 0.1, when_C_over_O_mid, when_C_over_O_low, None)

            def when_else(_):
                return jnp.int32(0)      # HCO CO

            return lax.cond(cond_N_hot, when_N_hot, when_else, None)

        def branch_CO_ge1(_):
            # C/O >= 1
            turn = RateJAX.top(T_i, C, N, O)
            cond_lower = p_i > turn

            def when_lower(_):
                return jnp.int32(0)      # HCO CO

            def when_upper(_):
                cond_N_hot2 = jnp.logical_and(N_over_C > 0.1, T_i > 900.0)

                def when_N_hot2(_):
                    return jnp.int32(3)  # HCNO H2O

                def when_else2(_):
                    return jnp.int32(1)  # HCO H2O

                return lax.cond(cond_N_hot2, when_N_hot2, when_else2, None)

            return lax.cond(cond_lower, when_lower, when_upper, None)

        cond_CO_lt1 = C_over_O < 1.0

        # IMPORTANT: DO NOT wrap this in Python int()
        return lax.cond(cond_CO_lt1, branch_CO_lt1, branch_CO_ge1, None)


    # ---------- Solve one layer ----------

    def _solve_one_layer(
        self,
        T_i: float,
        p_i: float,
        f_i: float,
        k1_i: float,
        k2_i: float,
        k3_i: float,
        k4_i: float,
        k5_i: float,
        k6_i: float,
    ) -> jnp.ndarray:
        C, O = self.C, self.O

        # 0 -> HCO_poly6_CO
        # 1 -> HCO_poly6_H2O
        # 2 -> HCNO_poly8_CO
        # 3 -> HCNO_poly8_H2O
        poly_idx = self._choose_poly_index(T_i, p_i)   # JAX int scalar
        is_H2O_var = (poly_idx % 2 == 1)

        # Build all four coefficient sets (each length 7 now)
        A_HCO_CO   = self.HCO_poly6_CO(f_i, k1_i, k2_i, k3_i, k4_i)
        A_HCO_H2O  = self.HCO_poly6_H2O(f_i, k1_i, k2_i, k3_i, k4_i)
        A_HCNO_CO  = self.HCNO_poly8_CO(f_i, k1_i, k2_i, k3_i, k4_i, k5_i, k6_i)
        A_HCNO_H2O = self.HCNO_poly8_H2O(f_i, k1_i, k2_i, k3_i, k4_i, k5_i, k6_i)

        A_all = jnp.stack([A_HCO_CO, A_HCO_H2O, A_HCNO_CO, A_HCNO_H2O], axis=0)
        A = A_all[poly_idx]   # shape (7,)

        # Bounds for the root
        vmax_H2O = f_i * O
        vmax_CO  = f_i * jnp.minimum(C, O)
        vmax = jnp.where(is_H2O_var, vmax_H2O, vmax_CO)
        guess = 0.99 * vmax

        # More stable multi-guess NR:
        root = self.newton_raphson_bounded(A, guess, vmax)

        # Recover H2O and CO
        H2O_from_CO = (f_i * O - root) / (1.0 + 2.0 * root / k2_i)
        CO_from_H2O = (f_i * O - root) / (1.0 + 2.0 * root / k2_i)

        H2O = jnp.where(is_H2O_var, root,        H2O_from_CO)
        CO  = jnp.where(is_H2O_var, CO_from_H2O, root)

        # Remaining species (normalized to H2):
        return self.solve_rest(H2O, CO, f_i, k1_i, k2_i, k3_i, k4_i, k5_i, k6_i)


    # ---------- Main public API: solve profile & return VMR dict ----------

    def solve_profile(
        self,
        T: jnp.ndarray,
        p: jnp.ndarray,
        return_diagnostics: bool = False,
    ) -> Union[Dict[str, jnp.ndarray], Tuple[Dict[str, jnp.ndarray], Dict]]:
        """
        Solve thermochemical equilibrium across a 1D T-p profile.

        Parameters
        ----------
        T : 1D array [K]
            Temperature profile
        p : 1D array [bar]
            Pressure profile
        return_diagnostics : bool, optional
            If True, return (vmr_dict, diagnostics) with convergence info

        Returns
        -------
        vmr : dict[str, jnp.ndarray]
            Keys: self.species, each value shape = (nlayers,)
        diagnostics : dict, optional
            Only returned if return_diagnostics=True. Contains:
            - 'n_layers': number of layers
            - 'T_range': (min, max) temperature
            - 'p_range': (min, max) pressure

        Raises
        ------
        ValueError
            If inputs have incompatible shapes or invalid values
        """
        # Convert to arrays
        T = jnp.asarray(T)
        p = jnp.asarray(p)

        # Shape validation (JIT-compatible using assertions on static shapes)
        # Note: Value validation (T > 0, p > 0) should be done by caller when using JIT
        if hasattr(T, 'shape') and hasattr(p, 'shape'):
            # These checks work with concrete arrays (non-JIT)
            if T.ndim != 1:
                raise ValueError(f"Temperature must be 1D array, got {T.ndim}D")
            if p.ndim != 1:
                raise ValueError(f"Pressure must be 1D array, got {p.ndim}D")
            if T.shape != p.shape:
                raise ValueError(
                    f"Temperature and pressure must have same shape, "
                    f"got T.shape={T.shape} and p.shape={p.shape}"
                )

        nlayers = T.shape[0]

        # Equilibrium constants (vectorized):
         # Equilibrium constants (vectorized):
        k0 = self.kprime0(T, p)
        k1 = self.kprime1(T, p)
        k2 = self.kprime2(T)
        k3 = self.kprime3(T, p)
        k4 = self.kprime4(T, p)
        k5 = self.kprime5(T, p)
        k6 = self.kprime6(T, p)

        # Avoid exact 0/inf constants (which break algebra downstream)
        k_min, k_max = 1e-300, 1e300
        k0 = jnp.clip(k0, k_min, k_max)
        k1 = jnp.clip(k1, k_min, k_max)
        k2 = jnp.clip(k2, k_min, k_max)
        k3 = jnp.clip(k3, k_min, k_max)
        k4 = jnp.clip(k4, k_min, k_max)
        k5 = jnp.clip(k5, k_min, k_max)
        k6 = jnp.clip(k6, k_min, k_max)

        # Hydrogen chemistry:
        # Hatom and H2 from quadratic as in original code:
        Hatom = (-1.0 + jnp.sqrt(1.0 + 8.0 / k0)) / (4.0 / k0)
        Hmol  = Hatom**2 / k0     # n(H2)
        f     = (Hatom + 2.0 * Hmol) / Hmol

        # Solve heavy species per layer with vmap:
        solve_layer_vmapped = jax.vmap(
            self._solve_one_layer,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        # shape: (nlayers, 9)
        heavy_norm = solve_layer_vmapped(T, p, f, k1, k2, k3, k4, k5, k6)
        # Transpose to (9, nlayers)
        heavy_norm = heavy_norm.T

        # De-normalize by H2 to get absolute number ratios:
        heavy = heavy_norm * Hmol

        # H2, H, He
        H2 = Hmol
        H  = Hatom
        He = self.fHe * (2.0 * H2 + H)

        # Stack all species in the same order as self.species:
        all_species = jnp.vstack([heavy, H2[None, :], H[None, :], He[None, :]])

        # Convert to VMR: normalize by total:
        total = jnp.sum(all_species, axis=0, keepdims=True)
        vmr_all = all_species / total

        # Build dictionary: each species -> (nlayers,)
        vmr_dict: Dict[str, jnp.ndarray] = {}
        for i, name in enumerate(self.species):
            vmr_dict[name] = vmr_all[i, :]

        # Optionally return diagnostics
        if return_diagnostics:
            diagnostics = {
                'n_layers': nlayers,
                'T_range': (float(jnp.min(T)), float(jnp.max(T))),
                'p_range': (float(jnp.min(p)), float(jnp.max(p))),
                'T_mean': float(jnp.mean(T)),
                'p_mean_log': float(jnp.mean(jnp.log10(p))),
            }
            return vmr_dict, diagnostics

        return vmr_dict
