"""
chem_easychem_short_jax.py
==========================

JAX implementation of the EasyChem Fortran ``SHORT`` gas-phase solver for the
restricted case of:

- gas phase only
- no condensates
- no ions/electrons

The public API mirrors the existing Exo_Skryer chemistry backends:

- ``EasyChemShortModel.from_nasa9_dir(...)``
- ``solve_one_TP(...)``
- ``solve_profile_scan(...)``
- ``solve_profile_vmap(...)``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


EPS = 1e-300
TRACE_FLOOR_FRAC = 1e-4
MAX_STEPS_DEFAULT = 128
TOL_DEFAULT = 5e-6
RELAX_LIMIT_DEFAULT = 0.75
SIZE_LIMIT = 18.420681
PI_TOL = 1e-3
ABUND_TOL = 1e-10
MASS_REL_TOL = 1e-2


def list_nasa9_species(nasa9_dir: str | Path) -> Tuple[str, ...]:
    nasa9_dir = Path(nasa9_dir)
    return tuple(sorted(p.stem for p in nasa9_dir.glob("*.txt")))


_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> Dict[str, int]:
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


def build_stoich_matrix(species: Sequence[str], elements: Sequence[str]) -> jnp.ndarray:
    e_idx = {e: i for i, e in enumerate(elements)}
    A = np.zeros((len(elements), len(species)), dtype=np.float64)
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
    from .rate_jax import load_nasa9_cache

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
    coeff_low: jnp.ndarray,
    coeff_high: jnp.ndarray,
    T_mid: jnp.ndarray,
) -> jnp.ndarray:
    T = jnp.asarray(T)
    lnT = jnp.log(T)

    use_low = T <= T_mid
    a = jnp.where(use_low[:, None], coeff_low, coeff_high)
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


@dataclass(frozen=True)
class EasyChemShortInputs:
    A: jnp.ndarray
    b: jnp.ndarray
    coeff_low: jnp.ndarray
    coeff_high: jnp.ndarray
    T_mid: jnp.ndarray
    P0: jnp.ndarray
    n_species: int
    n_elements: int

    def tree_flatten(self):
        children = (self.A, self.b, self.coeff_low, self.coeff_high, self.T_mid, self.P0)
        aux_data = (int(self.n_species), int(self.n_elements))
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        n_species, n_elements = aux_data
        A, b, coeff_low, coeff_high, T_mid, P0 = children
        return cls(
            A=A,
            b=b,
            coeff_low=coeff_low,
            coeff_high=coeff_high,
            T_mid=T_mid,
            P0=P0,
            n_species=n_species,
            n_elements=n_elements,
        )


jax.tree_util.register_pytree_node_class(EasyChemShortInputs)


@dataclass(frozen=True)
class EasyChemShortModel:
    species: Tuple[str, ...]
    elements: Tuple[str, ...]
    inputs: EasyChemShortInputs

    @classmethod
    def from_nasa9_dir(
        cls,
        nasa9_dir: str | Path,
        *,
        species: Sequence[str] | None = None,
        elements: Sequence[str] | None = None,
        element_budgets: Mapping[str, float] | None = None,
        P0_bar: float = 1.0,
    ) -> "EasyChemShortModel":
        nasa9_dir = Path(nasa9_dir)
        sp = tuple(species) if species is not None else list_nasa9_species(nasa9_dir)
        el = tuple(elements) if elements is not None else infer_elements(sp)

        if element_budgets is None:
            b = jnp.ones((len(el),), dtype=jnp.float64)
        else:
            b = jnp.asarray([float(element_budgets.get(e, 0.0)) for e in el], dtype=jnp.float64)

        A = build_stoich_matrix(sp, el)
        coeff_low, coeff_high, T_mid = load_nasa9_coeff_arrays(nasa9_dir, sp)
        inp = EasyChemShortInputs(
            A=A,
            b=b,
            coeff_low=coeff_low,
            coeff_high=coeff_high,
            T_mid=T_mid,
            P0=jnp.asarray(P0_bar, dtype=jnp.float64),
            n_species=len(sp),
            n_elements=len(el),
        )
        return cls(species=sp, elements=el, inputs=inp)


def _gibbs_terms(T: jnp.ndarray, inp: EasyChemShortInputs) -> jnp.ndarray:
    return nasa9_g0_over_RT(T, inp.coeff_low, inp.coeff_high, inp.T_mid)


def _mu_over_RT(
    p_bar: jnp.ndarray,
    log_ns: jnp.ndarray,
    log_n: jnp.ndarray,
    g0_rt: jnp.ndarray,
    inp: EasyChemShortInputs,
) -> jnp.ndarray:
    return g0_rt + log_ns - log_n + jnp.log(jnp.maximum(p_bar / inp.P0, EPS))


def initial_guess_from_budgets(
    inp: EasyChemShortInputs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    A = inp.A
    b = inp.b
    nelem, nspecies = A.shape
    n_seed = jnp.maximum(jnp.sum(jnp.maximum(b, 0.0)), 1.0)
    ns_floor = n_seed * TRACE_FLOOR_FRAC
    n_spec = jnp.full((nspecies,), ns_floor, dtype=jnp.float64)

    nonzero = jnp.sum(A > 0, axis=0)

    def add_element_seed(e_idx, ns_acc):
        a_e = A[e_idx, :]
        mask = a_e > 0
        score = jnp.where(
            mask,
            jnp.where(nonzero == 1, 0.0, 1e3) + 1.0 / jnp.maximum(a_e, 1.0),
            jnp.inf,
        )
        s_idx = jnp.argmin(score)
        add = b[e_idx] / jnp.maximum(a_e[s_idx], 1.0)
        return ns_acc.at[s_idx].add(jnp.maximum(add, 0.0))

    n_spec = lax.fori_loop(0, nelem, add_element_seed, n_spec)
    n_spec = jnp.maximum(n_spec, ns_floor)
    n = jnp.sum(n_spec) * 1.1
    pi_atom = jnp.zeros((nelem,), dtype=jnp.float64)
    return n_spec, jnp.log(jnp.maximum(n_spec, EPS)), n, pi_atom


def _assemble_matrix_and_vector(
    p_bar: jnp.ndarray,
    T: jnp.ndarray,
    n: jnp.ndarray,
    n_spec: jnp.ndarray,
    log_ns: jnp.ndarray,
    inp: EasyChemShortInputs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    A = inp.A.T
    g0_rt = _gibbs_terms(T, inp)
    log_n = jnp.log(jnp.maximum(n, EPS))
    mu_over_rt = _mu_over_RT(p_bar, log_ns, log_n, g0_rt, inp)

    weighted_A = A * n_spec[:, None]
    atom_block = weighted_A.T @ A
    atom_total = jnp.sum(weighted_A, axis=0)
    neq = inp.n_elements + 1

    matrix = jnp.zeros((neq, neq), dtype=jnp.float64)
    matrix = matrix.at[: inp.n_elements, : inp.n_elements].set(atom_block)
    matrix = matrix.at[: inp.n_elements, inp.n_elements].set(atom_total)
    matrix = matrix.at[inp.n_elements, : inp.n_elements].set(atom_total)
    matrix = matrix.at[inp.n_elements, inp.n_elements].set(jnp.sum(n_spec) - n)

    b = inp.A @ n_spec
    vector_atoms = inp.b - b + jnp.sum(weighted_A * mu_over_rt[:, None], axis=0)
    vector_total = n - jnp.sum(n_spec) + jnp.sum(n_spec * mu_over_rt)
    vector = jnp.concatenate((vector_atoms, jnp.asarray([vector_total], dtype=jnp.float64)))
    return matrix, vector, mu_over_rt, A


def _delta_log_n_gas(
    solution_vector: jnp.ndarray,
    mu_over_rt: jnp.ndarray,
    A_species: jnp.ndarray,
    inp: EasyChemShortInputs,
) -> jnp.ndarray:
    pi_atom = solution_vector[: inp.n_elements]
    dlogn = solution_vector[inp.n_elements]
    return A_species @ pi_atom + dlogn - mu_over_rt


def _compute_lambda(
    solution_vector: jnp.ndarray,
    delta_log_n_gas: jnp.ndarray,
    n_spec: jnp.ndarray,
    n: jnp.ndarray,
    inp: EasyChemShortInputs,
    relax_limit: float,
) -> jnp.ndarray:
    del relax_limit
    log_ratio = jnp.log(jnp.maximum(n_spec / jnp.maximum(n, EPS), EPS))
    dlogn = solution_vector[inp.n_elements]

    mask1 = log_ratio > -SIZE_LIMIT
    denom1 = jnp.maximum(5.0 * jnp.abs(dlogn), jnp.abs(delta_log_n_gas))
    lambda1_each = jnp.where(mask1, 2.0 / jnp.maximum(denom1, EPS), jnp.inf)
    lambda1 = jnp.min(lambda1_each)

    mask2 = jnp.logical_and(log_ratio <= -SIZE_LIMIT, delta_log_n_gas >= 0.0)
    numer2 = -log_ratio - 9.2103404
    denom2 = delta_log_n_gas - dlogn
    safe_ratio = numer2 / jnp.where(jnp.abs(denom2) > EPS, denom2, jnp.inf)
    lambda2_each = jnp.where(mask2, jnp.abs(safe_ratio), jnp.inf)
    lambda2 = jnp.min(lambda2_each)

    lambda_all = jnp.minimum(1.0, jnp.minimum(lambda1, lambda2))
    return jnp.where(jnp.isfinite(lambda_all), lambda_all, 1.0)


def _mass_balance_ok(n_spec: jnp.ndarray, inp: EasyChemShortInputs) -> jnp.ndarray:
    b = inp.A @ n_spec
    max_b0 = jnp.maximum(jnp.max(inp.b), EPS)
    tol = max_b0 * MASS_REL_TOL
    mask = inp.b > 1e-6
    errs = jnp.abs(inp.b - b)
    return jnp.all(jnp.logical_or(~mask, errs <= tol))


def _pi_stability_ok(pi_old: jnp.ndarray, pi_new: jnp.ndarray) -> jnp.ndarray:
    denom = jnp.where(jnp.abs(pi_new) > EPS, jnp.abs(pi_new), 1.0)
    rel = jnp.abs((pi_old - pi_new) / denom)
    rel = jnp.where(jnp.isfinite(rel), rel, jnp.inf)
    return jnp.all(rel <= PI_TOL)


def _fallback_abundance_stability_ok(n_spec: jnp.ndarray, n_spec_old: jnp.ndarray) -> jnp.ndarray:
    return jnp.all(jnp.abs(n_spec - n_spec_old) <= ABUND_TOL)


def compute_modified_residual(
    packed_state: jnp.ndarray,
    T: jnp.ndarray,
    P_bar: jnp.ndarray,
    inp: EasyChemShortInputs,
) -> jnp.ndarray:
    ne = inp.n_elements
    ns = inp.n_species
    pi_atom = packed_state[:ne]
    log_ns = packed_state[ne : ne + ns]
    log_n = packed_state[ne + ns]
    n_spec = jnp.exp(log_ns)
    n = jnp.exp(log_n)
    g0_rt = _gibbs_terms(T, inp)
    mu_over_rt = _mu_over_RT(P_bar, log_ns, log_n, g0_rt, inp)
    species_metric = n_spec * (mu_over_rt - inp.A.T @ pi_atom)
    mass_metric = inp.A @ n_spec - inp.b
    total_metric = jnp.asarray([n - jnp.sum(n_spec)], dtype=jnp.float64)
    return jnp.concatenate((species_metric, mass_metric, total_metric))


def _pack_state(
    pi_atom: jnp.ndarray,
    log_ns: jnp.ndarray,
    log_n: jnp.ndarray,
    solution_vector: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.concatenate((pi_atom, log_ns, jnp.asarray([log_n], dtype=jnp.float64), solution_vector))


def solve_one_TP(
    T: jnp.ndarray,
    P_bar: jnp.ndarray,
    inp: EasyChemShortInputs,
    state0: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    *,
    max_steps: int = MAX_STEPS_DEFAULT,
    tol: float = TOL_DEFAULT,
    relax_limit: float = RELAX_LIMIT_DEFAULT,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if state0 is None:
        state0 = initial_guess_from_budgets(inp)

    n_spec0, log_ns0, n0, pi0 = state0
    solution0 = jnp.zeros((inp.n_elements + 1,), dtype=jnp.float64)

    def step(carry, _):
        n_spec, log_ns, n, pi_atom, n_spec_old, converged, solution_prev = carry

        def _frozen(__):
            return carry, None

        def _iterate(__):
            matrix, vector, mu_over_rt, A_species = _assemble_matrix_and_vector(P_bar, T, n, n_spec, log_ns, inp)
            solution = jnp.linalg.solve(matrix, vector)
            good = jnp.all(jnp.isfinite(solution))

            def _bad(___):
                return (n_spec, log_ns, n, pi_atom, n_spec_old, False, solution_prev), None

            def _good(___):
                delta_gas = _delta_log_n_gas(solution, mu_over_rt, A_species, inp)
                lam = _compute_lambda(solution, delta_gas, n_spec, n, inp, relax_limit)
                n_spec_new = jnp.maximum(n_spec * jnp.exp(lam * delta_gas), EPS)
                log_ns_new = jnp.log(n_spec_new)
                pi_new = solution[: inp.n_elements]
                n_new = n * jnp.exp(lam * solution[inp.n_elements])

                gas_good = jnp.all((n_spec_new * jnp.abs(delta_gas) / jnp.maximum(jnp.sum(n_spec_new), EPS)) <= tol)
                total_good = (n_new * jnp.abs(solution[inp.n_elements]) / jnp.maximum(jnp.sum(n_spec_new), EPS)) <= tol
                mass_good_raw = _mass_balance_ok(n_spec_new, inp)
                pi_good_raw = _pi_stability_ok(pi_atom, pi_new)
                fallback_ok = _fallback_abundance_stability_ok(n_spec_new, n_spec_old)
                mass_good = jnp.where(jnp.logical_or(mass_good_raw, pi_good_raw), mass_good_raw, fallback_ok)
                pi_good = jnp.where(jnp.logical_or(mass_good_raw, pi_good_raw), pi_good_raw, fallback_ok)
                converged_new = gas_good & total_good & mass_good & pi_good
                return (n_spec_new, log_ns_new, n_new, pi_new, n_spec, converged_new, solution), None

            return lax.cond(good, _good, _bad, operand=None)

        return lax.cond(converged, _frozen, _iterate, operand=None)

    init = (n_spec0, log_ns0, n0, pi0, n_spec0, False, solution0)
    final_carry, _ = lax.scan(step, init, xs=None, length=max_steps)
    n_spec_f, log_ns_f, n_f, pi_f, _n_spec_prev, converged_f, solution_f = final_carry
    y_f = n_spec_f / jnp.maximum(jnp.sum(n_spec_f), EPS)
    packed = _pack_state(pi_f, log_ns_f, jnp.log(jnp.maximum(n_f, EPS)), solution_f)
    result_code = jnp.where(converged_f, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32))
    return packed, y_f, n_spec_f / jnp.maximum(n_f, EPS), result_code


def solve_profile_scan(
    T_profile: jnp.ndarray,
    P_profile: jnp.ndarray,
    inp: EasyChemShortInputs,
    *,
    state_init: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    max_steps: int = MAX_STEPS_DEFAULT,
    tol: float = TOL_DEFAULT,
    relax_limit: float = RELAX_LIMIT_DEFAULT,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if state_init is None:
        state_init = initial_guess_from_budgets(inp)

    def step(state_prev, tp):
        T, P = tp
        packed, y, ns_over_n, result = solve_one_TP(
            T,
            P,
            inp,
            state_prev,
            max_steps=max_steps,
            tol=tol,
            relax_limit=relax_limit,
        )
        ne = inp.n_elements
        ns = inp.n_species
        log_ns = packed[ne : ne + ns]
        n_spec = jnp.exp(log_ns)
        log_n = packed[ne + ns]
        n = jnp.exp(log_n)
        pi = packed[:ne]
        next_state = (n_spec, log_ns, n, pi)
        return next_state, (packed, y, ns_over_n, result)

    _, out = lax.scan(step, state_init, (T_profile, P_profile))
    return out


def solve_profile_vmap(
    T_profile: jnp.ndarray,
    P_profile: jnp.ndarray,
    inp: EasyChemShortInputs,
    *,
    max_steps: int = MAX_STEPS_DEFAULT,
    tol: float = TOL_DEFAULT,
    relax_limit: float = RELAX_LIMIT_DEFAULT,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def solve_single(T, P):
        state0 = initial_guess_from_budgets(inp)
        return solve_one_TP(T, P, inp, state0, max_steps=max_steps, tol=tol, relax_limit=relax_limit)

    return jax.vmap(solve_single)(T_profile, P_profile)
