"""
lxmie_mod.py
============

LX-MIE Mie code refactored into JAX (Kitzmann et al. 2018).

"""

from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_DEFAULT_CF_EPS = 1e-10
_RESCALE_THRESH = 1e150

__all__ = [
    "lxmie_jax",
    "lxmie_jax_vmap",
]


def _nb_from_x(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.maximum(x, 0.0)
    return (jnp.floor(x + 4.3 * jnp.cbrt(x)).astype(jnp.int32) + 2)


def _an(i: jnp.ndarray, nu: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    sign = jnp.where((i % 2) == 0, -1.0, 1.0)
    return (sign * 2.0 * (nu + (i.astype(jnp.float64) - 1.0))) / z


def _an_real(i: jnp.ndarray, nu: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    sign = jnp.where((i % 2) == 0, -1.0, 1.0)
    return (sign * 2.0 * (nu + (i.astype(jnp.float64) - 1.0))) / z


def _rescale_pair(num, den):
    s = jnp.maximum(jnp.abs(num), jnp.abs(den))
    do = s > _RESCALE_THRESH
    factor = jnp.where(do, s, 1.0)
    return num / factor, den / factor


def _starting_AN_cf(N: int, mx: jnp.ndarray, cf_max_terms: int, cf_eps: float) -> jnp.ndarray:
    nu = jnp.array(float(N) + 0.5, dtype=jnp.float64)

    f_num = jnp.array(1.0 + 1.0j, dtype=jnp.complex128)
    f_den = jnp.array(1.0 + 1.0j, dtype=jnp.complex128)

    # i = 1
    a_num = _an(jnp.int32(1), nu, mx)
    a_den = jnp.array(1.0 + 0.0j, dtype=jnp.complex128)
    f_num = f_num * a_num
    f_den = f_den * a_den
    f_num, f_den = _rescale_pair(f_num, f_den)

    # i = 2
    a2 = _an(jnp.int32(2), nu, mx)
    a_num = a2 + 1.0 / a_num
    a_den = a2
    f_num = f_num * a_num
    f_den = f_den * a_den
    f_num, f_den = _rescale_pair(f_num, f_den)

    def cond(state):
        i, a_num, a_den, f_num, f_den, con = state
        return jnp.logical_and(i <= cf_max_terms, con >= cf_eps)

    def body(state):
        i, a_num, a_den, f_num, f_den, con = state
        ai = _an(i, nu, mx)
        a_num_new = ai + 1.0 / a_num
        a_den_new = ai + 1.0 / a_den

        f_num_new = f_num * a_num_new
        f_den_new = f_den * a_den_new
        f_num_new, f_den_new = _rescale_pair(f_num_new, f_den_new)

        con_new = jnp.abs((jnp.abs(a_num_new) - jnp.abs(a_den_new)) / jnp.abs(a_num_new))
        return (i + 1, a_num_new, a_den_new, f_num_new, f_den_new, con_new)

    con0 = jnp.array(jnp.inf, dtype=jnp.float64)
    state0 = (jnp.int32(3), a_num, a_den, f_num, f_den, con0)
    _, _, _, f_num, f_den, _ = jax.lax.while_loop(cond, body, state0)

    return (f_num / f_den) - (jnp.array(float(N), dtype=jnp.float64) / mx)


def _starting_AN_cf_real(N: int, x: jnp.ndarray, cf_max_terms: int, cf_eps: float) -> jnp.ndarray:
    nu = jnp.array(float(N) + 0.5, dtype=jnp.float64)

    f_num = jnp.array(1.0, dtype=jnp.float64)
    f_den = jnp.array(1.0, dtype=jnp.float64)

    # i = 1
    a_num = _an_real(jnp.int32(1), nu, x)
    a_den = jnp.array(1.0, dtype=jnp.float64)
    f_num = f_num * a_num
    f_den = f_den * a_den
    f_num, f_den = _rescale_pair(f_num, f_den)

    # i = 2
    a2 = _an_real(jnp.int32(2), nu, x)
    a_num = a2 + 1.0 / a_num
    a_den = a2
    f_num = f_num * a_num
    f_den = f_den * a_den
    f_num, f_den = _rescale_pair(f_num, f_den)

    def cond(state):
        i, a_num, a_den, f_num, f_den, con = state
        return jnp.logical_and(i <= cf_max_terms, con >= cf_eps)

    def body(state):
        i, a_num, a_den, f_num, f_den, con = state
        ai = _an_real(i, nu, x)
        a_num_new = ai + 1.0 / a_num
        a_den_new = ai + 1.0 / a_den

        f_num_new = f_num * a_num_new
        f_den_new = f_den * a_den_new
        f_num_new, f_den_new = _rescale_pair(f_num_new, f_den_new)

        con_new = jnp.abs((a_num_new - a_den_new) / a_num_new)
        return (i + 1, a_num_new, a_den_new, f_num_new, f_den_new, con_new)

    con0 = jnp.array(jnp.inf, dtype=jnp.float64)
    state0 = (jnp.int32(3), a_num, a_den, f_num, f_den, con0)
    _, _, _, f_num, f_den, _ = jax.lax.while_loop(cond, body, state0)

    return (f_num / f_den) - (jnp.array(float(N), dtype=jnp.float64) / x)


def _compute_A_arrays(N: int, mx: jnp.ndarray, x: jnp.ndarray,
                      A_N_c: jnp.ndarray, A_N_r: jnp.ndarray):
    # Backward recursion from n=N..2, producing A_{n-1}
    ns = jnp.arange(N, 1, -1, dtype=jnp.int32)  # static because N is static

    def step(carry, n):
        A_c, A_r = carry
        dn = n.astype(jnp.float64)
        A_c_new = dn/mx - 1.0/(dn/mx + A_c)
        A_r_new = dn/x  - 1.0/(dn/x  + A_r)
        return (A_c_new, A_r_new), (A_c_new, A_r_new)

    (_, _), outs = jax.lax.scan(step, (A_N_c, A_N_r), ns)
    A_c_rev, A_r_rev = outs  # A_{N-1},...,A_1
    A_c = jnp.concatenate([A_c_rev[::-1], jnp.array([A_N_c], dtype=jnp.complex128)], axis=0)
    A_r = jnp.concatenate([A_r_rev[::-1], jnp.array([A_N_r], dtype=jnp.float64)], axis=0)
    return A_c, A_r  # length N


def _compute_mie_coeffs(N: int, m: jnp.ndarray, x: jnp.ndarray,
                        A_c: jnp.ndarray, A_r: jnp.ndarray):
    x = jnp.maximum(x, 1e-300)
    sinx = jnp.sin(x)
    cosx = jnp.cos(x)

    C = 1.0 + 1.0j * ((cosx + x*sinx) / (sinx - x*cosx))
    C = 1.0 / C
    D = -1.0j
    D = (-1.0/x) + 1.0/((1.0/x) - D)

    # n = 1
    A1 = A_c[0]
    A1r = A_r[0]
    a1 = C * ((A1/m) - A1r) / ((A1/m) - D)
    b1 = C * ((A1*m) - A1r) / ((A1*m) - D)

    ns = jnp.arange(2, N+1, dtype=jnp.int32)  # static

    def step(carry, n):
        C, D = carry
        dn = n.astype(jnp.float64)
        An = A_c[n-1]
        Anr = A_r[n-1]

        D = (-dn/x) + 1.0/((dn/x) - D)
        C = C * ((D + dn/x) / (Anr + dn/x))

        a = C * ((An/m) - Anr) / ((An/m) - D)
        b = C * ((An*m) - Anr) / ((An*m) - D)
        return (C, D), (a, b)

    (_, _), outs = jax.lax.scan(step, (C, D), ns)
    a_rest, b_rest = outs
    a = jnp.concatenate([jnp.array([a1], dtype=jnp.complex128), a_rest], axis=0)
    b = jnp.concatenate([jnp.array([b1], dtype=jnp.complex128), b_rest], axis=0)
    return a, b  # length N


@partial(jax.jit, static_argnames=("nmax", "cf_max_terms"))
def lxmie_jax(ri, x, *, nmax: int = 2000, cf_max_terms: int = 2000, cf_eps: float = _DEFAULT_CF_EPS):
    """JIT-safe LX-MIE  Mie solver.

    Computes Mie scattering efficiencies for homogeneous spheres using
    the full Lorenz-Mie solution with continued fractions for numerical stability (Kitzmann et al. 2018).
    For JIT compatibility, we meed to assume a constant nmax (Accurate up to around x = 1000)
    

    Parameters
    ----------
    ri : `~jax.numpy.ndarray`
        Complex refractive index (m = n + ik).
    x : `~jax.numpy.ndarray`
        Size parameter (x = 2πr/λ).
    nmax : int, optional
        Maximum number of Mie coefficients (default: 4096).
    cf_max_terms : int, optional
        Maximum continued fraction terms (default: 4096).
    cf_eps : float, optional
        Continued fraction convergence tolerance (default: 1e-10).

    Returns
    -------
    q_ext : `~jax.numpy.ndarray`
        Extinction efficiency.
    q_sca : `~jax.numpy.ndarray`
        Scattering efficiency.
    q_abs : `~jax.numpy.ndarray`
        Absorption efficiency.
    g : `~jax.numpy.ndarray`
        Asymmetry parameter.
    """
    m = ri.astype(jnp.complex128)
    x = x.astype(jnp.float64)
    mx = m * x

    # truncation order for sums only (dynamic is OK here because it's only used in comparisons)
    nb = jnp.minimum(_nb_from_x(x), jnp.int32(nmax))

    # Continued fraction at N = nmax (static)
    A_N_c = _starting_AN_cf(nmax, mx, cf_max_terms=cf_max_terms, cf_eps=cf_eps)
    A_N_r = _starting_AN_cf_real(nmax, x, cf_max_terms=cf_max_terms, cf_eps=cf_eps)

    # A_n arrays and Mie coefficients up to N=nmax
    A_c, A_r = _compute_A_arrays(nmax, mx, jnp.maximum(x, 1e-300), A_N_c, A_N_r)
    a, b = _compute_mie_coeffs(nmax, m, x, A_c, A_r)

    # Static n grid
    n = jnp.arange(1, nmax + 1, dtype=jnp.float64)
    mask = (n <= nb.astype(jnp.float64)).astype(jnp.float64)  # 1..nb

    w = (2.0 * n + 1.0) * mask
    q_sca = jnp.sum(w * ((jnp.abs(a) ** 2) + (jnp.abs(b) ** 2)))
    q_ext = jnp.sum(w * jnp.real(a + b))

    x2 = x * x
    q_sca = q_sca * (2.0 / x2)
    q_ext = q_ext * (2.0 / x2)
    q_abs = q_ext - q_sca

    # g sum over n=1..nb-1
    n_g = jnp.arange(1, nmax, dtype=jnp.float64)  # length nmax-1
    mask_g = (n_g < nb.astype(jnp.float64)).astype(jnp.float64)

    a_n = a[:-1]
    a_np1 = a[1:]
    b_n = b[:-1]
    b_np1 = b[1:]

    term1 = n_g * (n_g + 2.0) / (n_g + 1.0) * jnp.real(a_n * jnp.conj(a_np1) + b_n * jnp.conj(b_np1))
    term2 = (2.0 * n_g + 1.0) / (n_g * (n_g + 1.0)) * jnp.real(b_n * jnp.conj(b_n))

    g_num = jnp.sum((term1 + term2) * mask_g)
    g = jnp.where(q_sca > 0.0, g_num * (4.0 / (x2 * q_sca)), 0.0)

    return q_ext, q_sca, q_abs, g


def lxmie_jax_vmap(
    ri: jnp.ndarray,
    x: jnp.ndarray,
    *,
    nmax: int = 4096,
    cf_max_terms: int = 4096,
    cf_eps: float = _DEFAULT_CF_EPS,
):
    """Batched wrapper around lxmie_jax with static args bound.

    Parameters
    ----------
    ri : `~jax.numpy.ndarray`, shape (N,)
        Complex refractive indices.
    x : `~jax.numpy.ndarray`, shape (N,)
        Size parameters.
    nmax : int, optional
        Maximum number of Mie coefficients (default: 4096).
    cf_max_terms : int, optional
        Maximum continued fraction terms (default: 4096).
    cf_eps : float, optional
        Continued fraction convergence tolerance (default: 1e-10).

    Returns
    -------
    q_ext : `~jax.numpy.ndarray`, shape (N,)
        Extinction efficiencies.
    q_sca : `~jax.numpy.ndarray`, shape (N,)
        Scattering efficiencies.
    q_abs : `~jax.numpy.ndarray`, shape (N,)
        Absorption efficiencies.
    g : `~jax.numpy.ndarray`, shape (N,)
        Asymmetry parameters.
    """
    return jax.vmap(
        lambda ri_i, x_i: lxmie_jax(
            ri_i,
            x_i,
            nmax=nmax,
            cf_max_terms=cf_max_terms,
            cf_eps=cf_eps,
        ),
        in_axes=(0, 0),
        out_axes=(0, 0, 0, 0),
    )(ri, x)
