"""
chem_equilibrium_jax.py
=======================

JAX-friendly thermochemical equilibrium solver for an ideal-gas mixture using
the method of element potentials (Lagrange multipliers on element budgets).

This is intended as a GPU-ready building block:
- `jax.jit` compatible (no Python-side species loops inside the solve).
- `jax.vmap` compatible (solve many independent (T,P) points in parallel).
- Uses NASA-9 polynomials from the repo `NASA9/` directory (via Exo_Skryer's loader).

The formulation matches the "element potentials" approach used in the `stash/`
experiments, but packaged as a small, reusable module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
import optimistix as optx
import lineax as lx


# ----------------------------
# Numerical constants
# ----------------------------

EPSILON_B = 1e-300
EPSILON_CLIP_MIN = 1e-300
EPSILON_CLIP_MAX = 1e300
EPSILON_DELTA = 1e-30

Y_FLOOR_DEFAULT = 1e-30
REL_FLOOR_DEFAULT = 1e-30
RELAX_FRAC_DEFAULT = 0.5
FALLBACK_ELEM_TOL_DEFAULT = 1e-10
RIDGE_DEFAULT = 1e-12


# ----------------------------
# NASA9 utilities
# ----------------------------

def list_nasa9_species(nasa9_dir: str | Path) -> Tuple[str, ...]:
    nasa9_dir = Path(nasa9_dir)
    species = sorted(p.stem for p in nasa9_dir.glob("*.txt"))
    return tuple(species)


_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a simple chemical formula like 'C2H4', 'SO3', 'He'.

    Assumptions:
    - Only element symbols (Capital + optional lowercase) and integer counts.
    - No charges, parentheses, dots, or phase suffixes.
    """
    counts: Dict[str, int] = {}
    for el, n_str in _FORMULA_RE.findall(formula):
        n = int(n_str) if n_str else 1
        counts[el] = counts.get(el, 0) + n
    if not counts:
        raise ValueError(f"Could not parse chemical formula: {formula!r}")
    return counts


def infer_elements(
    species: Sequence[str],
    *,
    preferred_order: Sequence[str] = ("H", "He", "C", "N", "O", "S"),
) -> Tuple[str, ...]:
    present = set()
    for sp in species:
        present.update(parse_formula(sp).keys())
    ordered = [e for e in preferred_order if e in present]
    ordered += sorted(e for e in present if e not in set(preferred_order))
    return tuple(ordered)


def build_stoich_matrix(
    species: Sequence[str],
    elements: Sequence[str],
) -> jnp.ndarray:
    """
    Build A[e,s] = number of atoms of element e in species s.
    Shape (Ne, Ns).
    """
    Ne, Ns = len(elements), len(species)
    e_idx = {e: i for i, e in enumerate(elements)}
    A = np.zeros((Ne, Ns), dtype=np.float64)
    for j, sp in enumerate(species):
        for e, v in parse_formula(sp).items():
            if e in e_idx:
                A[e_idx[e], j] = float(v)
    return jnp.asarray(A)


def load_nasa9_coeff_arrays(
    nasa9_dir: str | Path,
    species: Sequence[str],
    *,
    default_t_switch: float = 1000.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return (coeff_low, coeff_high, T_mid) for `species`.

    Uses Exo_Skryer's NASA-9 loader so the parsing format stays consistent.
    """
    from exo_skryer.rate_jax import load_nasa9_cache

    thermo = load_nasa9_cache(str(nasa9_dir))
    coeff_low = jnp.stack([thermo.data[sp]["coeffs_low"] for sp in species], axis=0)
    coeff_high = jnp.stack([thermo.data[sp]["coeffs_high"] for sp in species], axis=0)
    T_mid = jnp.stack(
        [jnp.asarray(thermo.data[sp].get("t_switch", default_t_switch)) for sp in species],
        axis=0,
    )
    return coeff_low, coeff_high, T_mid


def nasa9_g0_over_RT(
    T: jnp.ndarray,
    coeff_low: jnp.ndarray,   # (Ns, 10)
    coeff_high: jnp.ndarray,  # (Ns, 10)
    T_mid: jnp.ndarray,       # (Ns,)
) -> jnp.ndarray:
    """Compute psi_s(T) = g°_s(T)/(R*T) for all species, shape (Ns,)."""
    T = jnp.asarray(T)
    lnT = jnp.log(T)

    use_low = T <= T_mid  # (Ns,)
    a = jnp.where(use_low[:, None], coeff_low, coeff_high)  # (Ns,10)
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = [a[:, k] for k in range(10)]

    t = T
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    invt = 1.0 / t
    invt2 = invt * invt

    h_over_RT = (
        -a1 * invt2
        + (a2 * lnT) * invt
        + a3
        + a4 * t / 2.0
        + a5 * t2 / 3.0
        + a6 * t3 / 4.0
        + a7 * t4 / 5.0
        + a8 * t5 / 6.0
        + a9 * invt
    )

    s_over_R = (
        -0.5 * a1 * invt2
        - a2 * invt
        + a3 * lnT
        + a4 * t
        + a5 * t2 / 2.0
        + a6 * t3 / 3.0
        + a7 * t4 / 4.0
        + a8 * t5 / 5.0
        + a10
    )

    return h_over_RT - s_over_R


# ----------------------------
# Element potentials solve
# ----------------------------

@dataclass(frozen=True)
class EquilInputs:
    A: jnp.ndarray          # (Ne, Ns)
    logA: jnp.ndarray       # (Ne, Ns) log(A) with -inf where A==0
    b: jnp.ndarray          # (Ne,) element totals (same order as `elements`)
    coeff_low: jnp.ndarray  # (Ns, 10)
    coeff_high: jnp.ndarray # (Ns, 10)
    T_mid: jnp.ndarray      # (Ns,)
    P0: jnp.ndarray         # scalar, same units as P inputs
    e_ref: int              # reference element index (e.g. H)

    def tree_flatten(self):
        children = (self.A, self.logA, self.b, self.coeff_low, self.coeff_high, self.T_mid, self.P0)
        aux_data = (int(self.e_ref),)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (e_ref,) = aux_data
        A, logA, b, coeff_low, coeff_high, T_mid, P0 = children
        return cls(A=A, logA=logA, b=b, coeff_low=coeff_low, coeff_high=coeff_high, T_mid=T_mid, P0=P0, e_ref=e_ref)


jax.tree_util.register_pytree_node_class(EquilInputs)

def _optx_fn(Lambda: jnp.ndarray, args):
    T, P, inp = args
    return residual_lambda(Lambda, T, P, inp)


def residual_lambda(
    Lambda: jnp.ndarray,  # (Ne,)
    T: jnp.ndarray,
    P: jnp.ndarray,
    inp: EquilInputs,
) -> jnp.ndarray:
    """Residual F(Lambda)=0, length Ne."""
    A = inp.A
    logA = inp.logA
    b = inp.b
    e_ref = inp.e_ref

    psi = nasa9_g0_over_RT(T, inp.coeff_low, inp.coeff_high, inp.T_mid)  # (Ns,)
    q = -psi + (A.T @ Lambda)  # (Ns,)
    q = jnp.where(jnp.isfinite(q), q, jnp.array(-1e30, dtype=q.dtype))

    lnP = jnp.log(P / inp.P0)
    m = lax.stop_gradient(jnp.max(q))
    logZ = m + logsumexp(q - m)
    F0 = logZ - lnP

    q_shift = q - m
    log_bhat = logsumexp(logA + q_shift[None, :], axis=1)  # (Ne,)
    log_b = jnp.log(jnp.maximum(b, EPSILON_B))

    Fe = (log_bhat - log_bhat[e_ref]) - (log_b - log_b[e_ref])  # (Ne,)
    Fe = Fe.at[e_ref].set(F0)
    return Fe


def _reconstruct(
    T: jnp.ndarray,
    P: jnp.ndarray,
    inp: EquilInputs,
    Lambda: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Stable reconstruction of (y, n) from Lambda."""
    psi = nasa9_g0_over_RT(T, inp.coeff_low, inp.coeff_high, inp.T_mid)
    q = -psi + (inp.A.T @ Lambda)
    q = jnp.where(jnp.isfinite(q), q, jnp.array(-1e30, dtype=q.dtype))
    lnP = jnp.log(P / inp.P0)

    m = lax.stop_gradient(jnp.max(q))
    logZ = m + logsumexp(q - m)
    F0 = logZ - lnP

    # Always return a normalized composition; pressure consistency is enforced by F0 in the residual.
    y = jnp.exp(q - logZ)

    q_shift = q - m
    log_bhat = logsumexp(inp.logA + q_shift[None, :], axis=1)  # (Ne,)
    log_bhat_ref = log_bhat[inp.e_ref]

    # b_hat_ref = exp(m - lnP) * Σ_s A_ref,s * exp(q_shift_s)
    # => log(b_hat_ref) = (m - lnP) + log_bhat_ref
    log_b_hat_ref = (m - lnP) + log_bhat_ref
    log_b_ref = jnp.log(jnp.maximum(inp.b[inp.e_ref], EPSILON_B))
    log_n_tot = log_b_ref - log_b_hat_ref
    log_n_tot = jnp.clip(log_n_tot, -700.0, 700.0)
    n_tot = jnp.exp(log_n_tot)
    n = y * n_tot
    return y, n


def _element_budget_rel_inf_norm(inp: EquilInputs, n: jnp.ndarray) -> jnp.ndarray:
    b = inp.b
    r = inp.A @ n - b
    denom = jnp.maximum(jnp.abs(b), EPSILON_B)
    return jnp.max(jnp.abs(r) / denom)


def initial_lambda_guess(
    T: jnp.ndarray,
    P: jnp.ndarray,
    inp: EquilInputs,
    *,
    y_floor: float = Y_FLOOR_DEFAULT,
    ridge: float = RIDGE_DEFAULT,
) -> jnp.ndarray:
    """
    Build a cheap, JIT-safe initial guess by:
    1) seeding a simple mole-fraction vector y,
    2) solving a ridge-regularized normal equation for Lambda.
    """
    A = inp.A
    b = inp.b
    Ne, Ns = A.shape

    n_seed = jnp.full((Ns,), y_floor, dtype=jnp.float64)

    nonzero = jnp.sum(A > 0, axis=0)  # (Ns,)

    def pick_species_for_element(e_idx, n_accum):
        a_e = A[e_idx, :]  # (Ns,)
        mask = a_e > 0
        # Prefer "pure" species (only contains this element), and among them
        # prefer higher stoichiometric coefficient (e.g. H2 over H).
        score = jnp.where(
            mask,
            jnp.where(nonzero == 1, 0.0, 1e3) + 1.0 / jnp.maximum(a_e, 1.0),
            jnp.inf,
        )
        s_idx = jnp.argmin(score)
        add = b[e_idx] / jnp.maximum(a_e[s_idx], 1.0)
        return n_accum.at[s_idx].add(jnp.maximum(add, 0.0))

    n_seed = lax.fori_loop(0, Ne, pick_species_for_element, n_seed)
    y = n_seed / jnp.sum(n_seed)
    y = jnp.clip(y, y_floor, 1.0)
    y = y / jnp.sum(y)

    psi = nasa9_g0_over_RT(T, inp.coeff_low, inp.coeff_high, inp.T_mid)
    rhs = jnp.log(jnp.clip(y * (P / inp.P0), EPSILON_CLIP_MIN, EPSILON_CLIP_MAX)) + psi

    AA = inp.A @ inp.A.T
    AA = AA + ridge * jnp.eye(Ne, dtype=AA.dtype)
    return jnp.linalg.solve(AA, inp.A @ rhs)


def _log_ns_and_log_n_from_lambda(
    T: jnp.ndarray,
    P: jnp.ndarray,
    inp: EquilInputs,
    Lambda: jnp.ndarray,
    *,
    rel_floor: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute implied log(ns) and log(n) from Lambda (used for fallback damping).
    """
    A = inp.A
    logA = inp.logA
    e_ref = inp.e_ref
    b_ref = inp.b[e_ref]

    psi = nasa9_g0_over_RT(T, inp.coeff_low, inp.coeff_high, inp.T_mid)
    q = -psi + (A.T @ Lambda)

    m = lax.stop_gradient(jnp.max(q))
    q_shift = q - m

    log_bhat_ref = logsumexp(logA[e_ref, :] + q_shift, axis=0)
    log_b_ref = jnp.log(jnp.maximum(b_ref, EPSILON_B))

    log_ns = log_b_ref - log_bhat_ref + q_shift
    log_n = log_b_ref - log_bhat_ref + logsumexp(q_shift)

    log_ns = jnp.maximum(log_ns, log_n + jnp.log(rel_floor))
    return log_ns, log_n


def _damped_newton_fallback(
    T: jnp.ndarray,
    P: jnp.ndarray,
    inp: EquilInputs,
    Lambda0: jnp.ndarray,
    *,
    max_steps: int,
    tol: float,
    rel_floor: float = REL_FLOOR_DEFAULT,
    relax_frac: float = RELAX_FRAC_DEFAULT,
) -> jnp.ndarray:
    F = lambda L: residual_lambda(L, T, P, inp)
    J = jax.jacfwd(F)
    lin = lx.AutoLinearSolver(well_posed=False)

    def cond(state):
        k, L, fn = state
        return jnp.logical_and(k < max_steps, jnp.logical_or(~jnp.isfinite(fn), fn > tol))

    def body(state):
        k, L, _ = state
        f = F(L)
        Jm = J(L)

        finite = jnp.logical_and(jnp.all(jnp.isfinite(f)), jnp.all(jnp.isfinite(Jm)))

        def _solve_delta(_):
            op = lx.MatrixLinearOperator(Jm)
            sol = lx.linear_solve(op, -f, lin)
            return sol.value

        delta = lax.cond(finite, _solve_delta, lambda _: jnp.zeros_like(L), operand=None)
        delta = jnp.where(jnp.all(jnp.isfinite(delta)), delta, jnp.zeros_like(delta))

        log_ns, log_n = _log_ns_and_log_n_from_lambda(T, P, inp, L, rel_floor=rel_floor)
        log_ns_t, _ = _log_ns_and_log_n_from_lambda(T, P, inp, L + delta, rel_floor=rel_floor)
        dlog_ns = log_ns_t - log_ns
        max_abs = jnp.maximum(jnp.max(jnp.abs(dlog_ns)), EPSILON_DELTA)

        lam = jnp.minimum(1.0, (relax_frac * jnp.abs(log_n)) / max_abs)
        L_new = L + lam * delta

        fn_new = jnp.linalg.norm(F(L_new))
        fn_new = jnp.where(jnp.isfinite(fn_new), fn_new, jnp.array(jnp.inf, dtype=fn_new.dtype))
        return (k + 1, L_new, fn_new)

    f0 = F(Lambda0)
    fn0 = jnp.linalg.norm(f0)
    fn0 = jnp.where(jnp.isfinite(fn0), fn0, jnp.array(jnp.inf, dtype=fn0.dtype))
    _, Lf, _ = lax.while_loop(cond, body, (jnp.array(0, dtype=jnp.int32), Lambda0, fn0))
    return Lf


def solve_one_TP(
    T: jnp.ndarray,
    P: jnp.ndarray,
    inp: EquilInputs,
    Lambda0: jnp.ndarray,
    *,
    max_steps: int = 64,
    tol: float = 1e-11,
    throw: bool = False,
    prefer_chord: bool = True,
    fallback_rel_floor: float = REL_FLOOR_DEFAULT,
    fallback_relax_frac: float = RELAX_FRAC_DEFAULT,
    fallback_elem_tol: float = FALLBACK_ELEM_TOL_DEFAULT,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solve equilibrium at a single (T,P).

    Uses Optimistix Newton first, then (if needed) a damped-Newton fallback in Lambda-space.
    Returns (Lambda, y, n, result_code).
    """
    success_value = int(optx.RESULTS._name_to_item["successful"]._value)

    def run_newton(L0):
        solver = optx.Newton(rtol=tol, atol=tol, linear_solver=lx.AutoLinearSolver(well_posed=False))
        sol = optx.root_find(_optx_fn, solver, y0=L0, args=(T, P, inp), max_steps=max_steps, throw=throw)
        L = jnp.clip(sol.value, -1e3, 1e3)
        return L, jnp.asarray(sol.result._value, dtype=jnp.int32)

    def run_chord(L0):
        solver = optx.Chord(rtol=tol, atol=tol, linear_solver=lx.AutoLinearSolver(well_posed=False))
        sol = optx.root_find(_optx_fn, solver, y0=L0, args=(T, P, inp), max_steps=max_steps, throw=throw)
        L = jnp.clip(sol.value, -1e3, 1e3)
        return L, jnp.asarray(sol.result._value, dtype=jnp.int32)

    def run_lm(L0):
        solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)
        sol = optx.least_squares(_optx_fn, solver, y0=L0, args=(T, P, inp), max_steps=max_steps, throw=throw)
        L = jnp.clip(sol.value, -1e3, 1e3)
        return L, jnp.asarray(sol.result._value, dtype=jnp.int32)

    def _finite_or(candidate: jnp.ndarray, fallback: jnp.ndarray) -> jnp.ndarray:
        ok = jnp.all(jnp.isfinite(candidate))
        return lax.cond(ok, lambda _: candidate, lambda _: fallback, operand=None)

    def assess(L: jnp.ndarray):
        y, n = _reconstruct(T, P, inp, L)
        f = residual_lambda(L, T, P, inp)
        fn = jnp.linalg.norm(f)
        fn = jnp.where(jnp.isfinite(fn), fn, jnp.array(jnp.inf, dtype=fn.dtype))
        elem = _element_budget_rel_inf_norm(inp, n)
        elem = jnp.where(jnp.isfinite(elem), elem, jnp.array(jnp.inf, dtype=elem.dtype))
        return fn, elem, y, n

    def _run_primary(_):
        def _use_newton(__):
            Ln, rn = run_newton(Lambda0)
            def _fallback(___):
                Ln0 = _finite_or(Ln, Lambda0)
                return run_lm(Ln0)
            return lax.cond(rn == success_value, lambda ___: (Ln, rn), _fallback, operand=None)

        def _use_chord_with_fallback(__):
            Lc, rc = run_chord(Lambda0)
            def _fallback(___):
                Lc0 = _finite_or(Lc, Lambda0)
                Ln, rn = run_newton(Lc0)
                def _fallback2(____):
                    Ln0 = _finite_or(Ln, Lambda0)
                    return run_lm(Ln0)
                return lax.cond(rn == success_value, lambda ____: (Ln, rn), _fallback2, operand=None)
            return lax.cond(rc == success_value, lambda ___: (Lc, rc), _fallback, operand=None)

        return lax.cond(prefer_chord, _use_chord_with_fallback, _use_newton, operand=None)

    L_primary, r_primary = _run_primary(None)

    fn_primary, elem_err_primary, y_primary, n_primary = assess(L_primary)
    good_primary = jnp.logical_and(fn_primary < tol, elem_err_primary < fallback_elem_tol)
    good_primary = jnp.logical_and(good_primary, r_primary == success_value)

    def _accept(_):
        return L_primary, y_primary, n_primary, r_primary

    def _fallback(_):
        L_start = _finite_or(L_primary, Lambda0)
        failure_value = int(optx.RESULTS._name_to_item["max_steps_reached"]._value)

        # LM is typically the most robust recovery if Newton/Chord are unstable.
        L_lm, r_lm = run_lm(Lambda0)
        fn_lm, elem_err_lm, y_lm, n_lm = assess(L_lm)
        good_lm = jnp.logical_and(fn_lm < tol, elem_err_lm < fallback_elem_tol)
        good_lm = jnp.logical_and(good_lm, r_lm == success_value)

        def _accept_lm(__):
            return L_lm, y_lm, n_lm, r_lm

        def _fallback_newton(__):
            L_fb = _damped_newton_fallback(
                T,
                P,
                inp,
                _finite_or(L_lm, L_start),
                max_steps=max_steps,
                tol=tol,
                rel_floor=fallback_rel_floor,
                relax_frac=fallback_relax_frac,
            )
            fn_fb, elem_err_fb, y_fb, n_fb = assess(L_fb)
            ok = jnp.logical_and(fn_fb < tol, elem_err_fb < fallback_elem_tol)
            r_out = lax.cond(
                ok,
                lambda __: jnp.asarray(success_value, dtype=jnp.int32),
                lambda __: jnp.asarray(failure_value, dtype=jnp.int32),
                operand=None,
            )
            return L_fb, y_fb, n_fb, r_out

        return lax.cond(good_lm, _accept_lm, _fallback_newton, operand=None)

    return lax.cond(jnp.logical_not(good_primary), _fallback, _accept, operand=None)


def solve_profile_vmap(
    T_profile: jnp.ndarray,
    P_profile: jnp.ndarray,
    inp: EquilInputs,
    *,
    max_steps: int = 64,
    tol: float = 1e-11,
    throw: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve along a profile with cold starts (parallel over layers)."""

    def solve_single(T, P):
        L0 = initial_lambda_guess(T, P, inp)
        return solve_one_TP(T, P, inp, L0, max_steps=max_steps, tol=tol, throw=throw, prefer_chord=False)

    return jax.vmap(solve_single)(T_profile, P_profile)


def solve_profile_scan(
    T_profile: jnp.ndarray,
    P_profile: jnp.ndarray,
    inp: EquilInputs,
    *,
    Lambda_init: jnp.ndarray | None = None,
    max_steps: int = 64,
    tol: float = 1e-11,
    throw: bool = False,
    prefer_chord: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve along a profile with warm-starting (sequential scan)."""
    Ne = inp.A.shape[0]
    if Lambda_init is None:
        Lambda_init = initial_lambda_guess(T_profile[0], P_profile[0], inp)
    Lambda_init = jnp.asarray(Lambda_init).reshape((Ne,))

    def step(L_prev, tp):
        T, P, idx = tp
        prefer_chord_i = jnp.logical_and(prefer_chord, idx != 0)
        L, y, n, res = solve_one_TP(
            T,
            P,
            inp,
            L_prev,
            max_steps=max_steps,
            tol=tol,
            throw=throw,
            prefer_chord=prefer_chord_i,
        )
        return L, (L, y, n, res)

    idx = jnp.arange(T_profile.shape[0], dtype=jnp.int32)
    _, out = lax.scan(step, Lambda_init, (T_profile, P_profile, idx))
    return out


# ----------------------------
# Convenience builder
# ----------------------------

@dataclass(frozen=True)
class ElementPotentialsModel:
    """
    Small convenience wrapper that keeps species/elements metadata next to EquilInputs.
    """
    species: Tuple[str, ...]
    elements: Tuple[str, ...]
    inputs: EquilInputs

    @classmethod
    def from_nasa9_dir(
        cls,
        nasa9_dir: str | Path,
        *,
        species: Sequence[str] | None = None,
        elements: Sequence[str] | None = None,
        element_budgets: Mapping[str, float] | None = None,
        P0_bar: float = 1.0,
        e_ref: str = "H",
    ) -> "ElementPotentialsModel":
        nasa9_dir = Path(nasa9_dir)
        sp = tuple(species) if species is not None else list_nasa9_species(nasa9_dir)
        el = tuple(elements) if elements is not None else infer_elements(sp)

        if element_budgets is None:
            b = jnp.ones((len(el),), dtype=jnp.float64)
        else:
            b = jnp.asarray([float(element_budgets.get(e, 0.0)) for e in el], dtype=jnp.float64)

        A = build_stoich_matrix(sp, el)
        coeff_low, coeff_high, T_mid = load_nasa9_coeff_arrays(nasa9_dir, sp)

        e_ref_idx = el.index(e_ref) if e_ref in el else 0
        inp = EquilInputs(
            A=A,
            logA=jnp.where(A > 0, jnp.log(A), -jnp.inf),
            b=b,
            coeff_low=coeff_low,
            coeff_high=coeff_high,
            T_mid=T_mid,
            P0=jnp.asarray(P0_bar, dtype=jnp.float64),
            e_ref=int(e_ref_idx),
        )
        return cls(species=sp, elements=el, inputs=inp)
