"""
rate_jax.py
===========
"""

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
# Global cache for NASA-9 thermo data
# ============================================================================
_NASA9_CACHE: Optional["NASA9ThermoJAX"] = None

Species = Tuple[str, ...]

__all__ = [
    "NASA9ThermoJAX",
    "load_nasa9_cache",
    "get_nasa9_cache",
    "clear_nasa9_cache",
    "is_nasa9_cache_loaded",
    "RateJAX",
]


class NASA9ThermoJAX:
    """JAX-friendly NASA-9 thermo evaluator.

    Stores per-species NASA-9 polynomial coefficients and evaluates the
    dimensionless Gibbs free energy `G/(R T)` on-the-fly in JAX (no pre-tabulation).

    Notes
    -----
    Expected per-species dictionary structure:

    - `coeffs_low` : `~jax.numpy.ndarray`, shape (10,)
        NASA-9 coefficients for the low-temperature range.
    - `coeffs_high` : `~jax.numpy.ndarray`, shape (10,)
        NASA-9 coefficients for the high-temperature range.
    - `t_switch` : float
        Temperature [K] where the polynomial range switches.
    - `t_min`, `t_max` : float
        Valid temperature bounds for the data (inputs are clipped).
    """
    def __init__(self, data: Mapping[str, Mapping[str, jnp.ndarray]]):
        self.data = data

    def g_over_RT(self, spec: str, T: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the dimensionless Gibbs free energy `G/(R T)` from NASA-9 polynomials.

        Parameters
        ----------
        spec : str
            Species key in the Gibbs table (e.g., `"H2O"`).
        T : `~jax.numpy.ndarray`
            Temperature in Kelvin.

        Returns
        -------
        g_over_RT : `~jax.numpy.ndarray`
            Dimensionless Gibbs free energy `G/(R T)` evaluated at `T`.
        """
        d = self.data[spec]
        coeffs_low = d["coeffs_low"]
        coeffs_high = d["coeffs_high"]
        t_switch = d["t_switch"]
        t_min = d["t_min"]
        t_max = d["t_max"]

        T = jnp.asarray(T)
        T = jnp.clip(T, t_min, t_max)

        def _h_over_RT(coeffs: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
            a1, a2, a3, a4, a5, a6, a7, a8, a9, _a10 = coeffs
            t2 = t * t
            t3 = t2 * t
            t4 = t3 * t
            t5 = t4 * t
            return (
                -a1 / (t * t)
                + (a2 * jnp.log(t)) / t
                + a3
                + a4 * t / 2.0
                + a5 * t2 / 3.0
                + a6 * t3 / 4.0
                + a7 * t4 / 5.0
                + a8 * t5 / 6.0
                + a9 / t
            )

        def _s_over_R(coeffs: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
            a1, a2, a3, a4, a5, a6, a7, a8, _a9, a10 = coeffs
            t2 = t * t
            t3 = t2 * t
            t4 = t3 * t
            t5 = t4 * t
            return (
                -0.5 * a1 / (t * t)
                - a2 / t
                + a3 * jnp.log(t)
                + a4 * t
                + a5 * t2 / 2.0
                + a6 * t3 / 3.0
                + a7 * t4 / 4.0
                + a8 * t5 / 5.0
                + a10
            )

        h_low = _h_over_RT(coeffs_low, T)
        s_low = _s_over_R(coeffs_low, T)
        h_high = _h_over_RT(coeffs_high, T)
        s_high = _s_over_R(coeffs_high, T)

        use_low = T < t_switch
        h = jnp.where(use_low, h_low, h_high)
        s = jnp.where(use_low, s_low, s_high)
        return h - s


# ============================================================================
# Global cache management functions
# ============================================================================

def load_nasa9_cache(nasa9_dir: str) -> NASA9ThermoJAX:
    """Load NASA-9 polynomial coefficient files into the global NASA-9 cache.

    This function should be called once during initialization (e.g., in
    run_retrieval.py) to load and cache the Gibbs free energy data before
    running forward models or retrievals.

    Parameters
    ----------
    nasa9_dir : str
        Directory containing NASA-9 coefficient files

    Returns
    -------
    nasa9 : NASA9ThermoJAX
        The loaded NASA-9 thermo table (also cached globally)

    Notes
    -----
    Expected file format:
      - Each file contains 20 floating point values (10 low-T + 10 high-T)
      - This repo stores them as 4 lines with 5 values each (still 20 total)
      - Filename: "<MOLNAME>.txt" (e.g., "H2O.txt")

    Examples
    --------
    >>> # In run_retrieval.py or similar initialization:
    >>> from rate_jax import load_nasa9_cache
    >>> thermo = load_nasa9_cache("NASA9/")
    """
    global _NASA9_CACHE

    data: Dict[str, Dict[str, jnp.ndarray]] = {}

    t_min = 200.0
    t_max = 6000.0
    t_switch = 1000.0

    def _read_nasa9_coeffs(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """Read NASA-9 coefficients from a file.

        Supported formats:
        - 20 numbers: low10 + high10
        - 23 numbers: Tmin, Tmid, Tmax + low10 + high10

        Lines may contain comments starting with '#' or '!'.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

        cleaned: list[str] = []
        for line in raw_lines:
            line = line.split("#", 1)[0]
            line = line.split("!", 1)[0]
            cleaned.append(line)

        raw = "\n".join(cleaned).replace("D", "E")
        coeffs = np.fromstring(raw, sep=" ")

        t_switch_local = t_switch
        if coeffs.size == 23:
            _tmin, tmid, _tmax = coeffs[:3]
            t_switch_local = float(tmid)
            coeffs = coeffs[3:]

        # Some sources include an additional high-temperature range, providing
        # three blocks of 10 coefficients (30 total). For Exo_Skryer we keep the
        # first two ranges (low, high) and ignore the third.
        if coeffs.size == 30:
            coeffs = coeffs[:20]

        if coeffs.size != 20:
            raise ValueError(
                f"Expected 20 NASA-9 coefficients in {path} (or 23 including Tmin/Tmid/Tmax), got {coeffs.size}."
            )
        return coeffs[:10], coeffs[10:], t_switch_local

    for fname in os.listdir(nasa9_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(nasa9_dir, fname)
        molname = os.path.splitext(fname)[0]

        coeffs_low_np, coeffs_high_np, t_switch_local = _read_nasa9_coeffs(path)
        data[molname] = {
            "coeffs_low": jnp.asarray(coeffs_low_np),
            "coeffs_high": jnp.asarray(coeffs_high_np),
            "t_switch": jnp.asarray(t_switch_local),
            "t_min": jnp.asarray(t_min),
            "t_max": jnp.asarray(t_max),
        }

    _NASA9_CACHE = NASA9ThermoJAX(data)
    return _NASA9_CACHE


def get_nasa9_cache() -> NASA9ThermoJAX:
    """Return the cached NASA-9 thermo table.

    Returns
    -------
    nasa9 : NASA9ThermoJAX
        The cached NASA-9 thermo table.

    Raises
    ------
    RuntimeError
        If cache has not been initialized with load_nasa9_cache()
    """
    if _NASA9_CACHE is None:
        raise RuntimeError(
            "NASA-9 cache not initialized. Call load_nasa9_cache() first."
        )
    return _NASA9_CACHE


def clear_nasa9_cache() -> None:
    """Clear the cached NASA-9 thermo table.

    Useful for freeing memory or reloading with different data.
    """
    global _NASA9_CACHE
    _NASA9_CACHE = None


def is_nasa9_cache_loaded() -> bool:
    """Return `True` if the NASA-9 cache is loaded.

    Returns
    -------
    loaded : bool
        True if cache is loaded, False otherwise.
    """
    return _NASA9_CACHE is not None


class RateJAX:
    """RATE-style thermochemical equilibrium solver implemented in JAX.

    This class computes equilibrium abundances for a reduced H/C/N/O chemistry
    network over a 1D (T, p) profile. It is designed to be usable inside
    JIT-compiled forward models:

    - Uses `jax.vmap` to solve each layer independently
    - Avoids SciPy; uses `optimistix` for root finding where needed
    - Returns a VMR dictionary keyed by species name

    Parameters
    ----------
    thermo : `~exo_skryer.rate_jax.NASA9ThermoJAX`
        NASA-9 thermo evaluator created by `load_nasa9_cache`.
    C, N, O : float
        Elemental abundances (number ratios relative to H₂, following the original
        RATE conventions).
    fHe : float
        Helium fraction factor used to compute He from H-bearing species.
    """

    def __init__(
        self,
        thermo: NASA9ThermoJAX,
        C: float = 2.5e-4,
        N: float = 1.0e-4,
        O: float = 5.0e-4,
        fHe: float = 0.0,
    ):
        self.thermo = thermo
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

    # ---------- Thermo wrappers ----------

    def g_over_RT(self, spec: str, T: jnp.ndarray) -> jnp.ndarray:
        return self.thermo.g_over_RT(spec, T)

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
        return jnp.exp(-(2.0 * self.g_over_RT("H", T) - self.g_over_RT("H2", T))) / p

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
                self.g_over_RT("CO", T) + 3.0 * self.g_over_RT("H2", T)
                - self.g_over_RT("CH4", T) - self.g_over_RT("H2O", T)
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
                self.g_over_RT("CO", T) + self.g_over_RT("H2O", T)
                - self.g_over_RT("CO2", T) - self.g_over_RT("H2", T)
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
                self.g_over_RT("C2H2", T) + 3.0 * self.g_over_RT("H2", T)
                - 2.0 * self.g_over_RT("CH4", T)
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
                self.g_over_RT("C2H2", T) + self.g_over_RT("H2", T)
                - self.g_over_RT("C2H4", T)
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
                self.g_over_RT("N2", T) + 3.0 * self.g_over_RT("H2", T)
                - 2.0 * self.g_over_RT("NH3", T)
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
                self.g_over_RT("HCN", T) + 3.0 * self.g_over_RT("H2", T)
                - self.g_over_RT("NH3", T) - self.g_over_RT("CH4", T)
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
        # Polynomial coefficients organized by variable
        # Structure: constant + T^1..4 + C^1..4 + N^1..4 + O^1..4
        const = -1.07028658e+03
        coeff_T = jnp.array([1.20815018e+03, -5.21868655e+02, 1.02459233e+02, -7.68350388e+00])
        coeff_C = jnp.array([1.30787500e+00, 3.18619604e-01, 5.32918135e-02, 3.12269845e-03])
        coeff_N = jnp.array([2.81238906e-02, 1.26015039e-02, 2.07616221e-03, 1.16038224e-04])
        coeff_O = jnp.array([-1.69589064e-01, -5.21662503e-02, -7.33669631e-03, -3.74492912e-04])

        # Compute log10 of input variables
        logT = jnp.log10(T)
        logC = jnp.log10(C)
        logN = jnp.log10(N)
        logO = jnp.log10(O)

        # Powers array [1, 2, 3, 4] for vectorized exponentiation
        powers = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Vectorized polynomial evaluation using broadcasting
        # For T: compute [logT^1, logT^2, logT^3, logT^4] then dot with coefficients
        # Note: logT can be array (batched), so we use outer-like broadcasting
        T_contrib = jnp.sum(coeff_T * logT[..., None] ** powers, axis=-1)
        C_contrib = jnp.sum(coeff_C * logC ** powers)
        N_contrib = jnp.sum(coeff_N * logN ** powers)
        O_contrib = jnp.sum(coeff_O * logO ** powers)

        log10_P_turn = const + T_contrib + C_contrib + N_contrib + O_contrib

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
        xtol: float = 1e-10,
        imax: int = 80,
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
        # Optimistix tolerances should be set in line with dtype. If x64 is disabled,
        # JAX will use float32, and tighter tolerances than ~1e-6 are ineffective.
        xtol_eff = float(xtol)
        if A.dtype == jnp.float32:
            xtol_eff = max(xtol_eff, 1e-6)

        # Success sentinel for Optimistix root-finding.
        _RESULT_SUCCESS = optx.RESULTS._name_to_item["successful"]

        def poly_fn(x, _args):
            """Function to find root of: polynomial(x) = 0"""
            return cls._eval_poly(A, x)

        def try_solve(guess_scaled: float) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Try to solve from a given initial guess."""
            solver = optx.Newton(rtol=xtol_eff, atol=xtol_eff)
            sol = optx.root_find(
                poly_fn,
                solver,
                y0=guess_scaled,
                args=None,
                max_steps=imax,
                throw=False,  # Don't raise on convergence failure
            )
            # Accept either explicit success or sufficiently small residual.
            fval = sol.state.f
            ok = (sol.result == _RESULT_SUCCESS) | (jnp.abs(fval) <= 10.0 * xtol_eff)
            return sol.value, ok

        # Try multiple initial guesses with decreasing scales
        def attempt_body(carry):
            root, ok, attempt = carry
            scale = 10.0 ** (-attempt)
            guess_scaled = guess * scale
            root_candidate, ok_candidate = try_solve(guess_scaled)
            return (root_candidate, ok_candidate, attempt + 1)

        def attempt_cond(carry):
            root, ok, attempt = carry
            finite = jnp.isfinite(root)
            in_range = jnp.logical_and(root >= 0.0, root <= vmax)
            good = jnp.logical_and(ok, jnp.logical_and(finite, in_range))
            more_attempts = attempt < kmax
            # keep trying while root is bad and we still have attempts left
            return jnp.logical_and(~good, more_attempts)

        # Start with bogus root so we always do at least one attempt
        carry0 = (jnp.array(-1.0), jnp.array(False), jnp.int32(0))
        root_final, ok_final, _ = lax.while_loop(attempt_cond, attempt_body, carry0)

        def _bisect_root() -> jnp.ndarray:
            """Bisection fallback on [0, vmax] if a sign change exists."""
            lo = jnp.array(0.0, dtype=A.dtype)
            hi = jnp.asarray(vmax, dtype=A.dtype)
            flo = cls._eval_poly(A, lo)
            fhi = cls._eval_poly(A, hi)

            n_iter = 60 if A.dtype == jnp.float64 else 40

            def body(_, state):
                lo, hi, flo, fhi = state
                mid = 0.5 * (lo + hi)
                fmid = cls._eval_poly(A, mid)
                # If fmid is NaN/Inf, shrink interval conservatively.
                bad = ~jnp.isfinite(fmid)
                go_left = jnp.where(bad, True, flo * fmid <= 0.0)
                lo2 = jnp.where(go_left, lo, mid)
                hi2 = jnp.where(go_left, mid, hi)
                flo2 = jnp.where(go_left, flo, fmid)
                fhi2 = jnp.where(go_left, fmid, fhi)
                return lo2, hi2, flo2, fhi2

            lo, hi, _, _ = lax.fori_loop(0, n_iter, body, (lo, hi, flo, fhi))
            return 0.5 * (lo + hi)

        def _fallback_root() -> jnp.ndarray:
            # If Newton fails entirely (NaN or no acceptable candidate), return a safe interior point.
            return jnp.asarray(0.5, dtype=A.dtype) * jnp.asarray(vmax, dtype=A.dtype)

        # If Newton succeeded and produced a finite in-range root, use it. Otherwise try bisection if possible.
        root_final = jnp.clip(root_final, 0.0, vmax)
        newton_good = ok_final & jnp.isfinite(root_final)
        f0 = cls._eval_poly(A, jnp.array(0.0, dtype=A.dtype))
        f1 = cls._eval_poly(A, jnp.asarray(vmax, dtype=A.dtype))
        sign_change = jnp.isfinite(f0) & jnp.isfinite(f1) & (f0 * f1 <= 0.0)
        return lax.cond(
            newton_good,
            lambda: root_final,
            lambda: lax.cond(sign_change, _bisect_root, _fallback_root),
        )



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
        Encodes the logic:

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

        # Avoid exact 0/inf constants (which break algebra downstream).
        # Use dtype-aware bounds to avoid float32 overflow/underflow when x64 is disabled.
        if k0.dtype == jnp.float64:
            k_min, k_max = 1e-300, 1e300
        else:
            finfo = jnp.finfo(jnp.float32)
            k_min, k_max = finfo.tiny, finfo.max
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
