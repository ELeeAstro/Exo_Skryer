"""Transmission geometry and RT core on synthetic columns (no opacity data needed)."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from exo_skryer.RT_trans_geometry import (  # noqa: E402
    impact_parameters_1d, layer_radii, path_matrix_1d, path_tensor,
)
from exo_skryer.RT_trans_core_os import compute_transit_depth_1d_os  # noqa: E402
from exo_skryer.RT_trans_core_ck import compute_transit_depth_1d_ck  # noqa: E402

NLAY, NWL, NG = 40, 12, 8


def _state(seed=0, contri=False):
    rng = np.random.default_rng(seed)
    dz = rng.uniform(2e6, 4e6, NLAY)
    z_lev = np.concatenate([[-3e6], -3e6 + np.cumsum(dz)])
    return {
        "R0": jnp.asarray(7e9), "R_s": jnp.asarray(8e10),
        "z_lev": jnp.asarray(z_lev), "z_lay": jnp.asarray(0.5 * (z_lev[1:] + z_lev[:-1])),
        "dz": jnp.asarray(dz), "rho_lay": jnp.asarray(1e-4 * np.exp(-np.arange(NLAY) / 6.0)),
        "wl": jnp.asarray(np.linspace(1, 5, NWL)), "contri_func": contri,
    }, rng


def _opac(ng=NG):
    x, w = np.polynomial.legendre.leggauss(ng)
    return {"g_weights": jnp.asarray(w / 2), "g_points": jnp.asarray((x + 1) / 2)}


def _reference_path_matrix(state):
    """The 1D chord-length formula used before the unified geometry (kept as a reference)."""
    r_low, r_up = layer_radii(state)
    b = (state["R0"] + state["z_lay"])[:, None]
    dr = (r_up - r_low)[None, :]
    s = lambda r: jnp.sqrt(jnp.maximum(r[None, :] ** 2 - b**2, 0.0))
    tangent = (r_low[None, :] <= b) & (b < r_up[None, :])
    return jnp.where(r_up[None, :] <= b, 0.0, jnp.where(tangent, 2 * s(r_up) / dr, 2 * (s(r_up) - s(r_low)) / dr))


def test_path_matrix_matches_reference():
    st, _ = _state()
    b, _, _ = impact_parameters_1d(st)
    np.testing.assert_allclose(path_matrix_1d(st, b), _reference_path_matrix(st), rtol=1e-12, atol=0)


@pytest.mark.parametrize("edges", [
    [-np.pi / 2, 0.0, np.pi / 2],
    [-np.pi / 2, -0.3, 0.1, 0.5, np.pi / 2],
    list(np.linspace(-np.pi / 2, np.pi / 2, 9)),
])
def test_slices_of_identical_columns_sum_to_full_chord(edges):
    st, _ = _state()
    b, _, _ = impact_parameters_1d(st)
    r_low, r_up = layer_radii(st)
    k = len(edges) - 1
    P = path_tensor(b, jnp.tile(r_low, (k, 1)), jnp.tile(r_up, (k, 1)), jnp.asarray(edges))
    np.testing.assert_allclose(P.sum(axis=1), path_matrix_1d(st, b), rtol=1e-12, atol=1e-12)


def test_day_and_night_halves_are_symmetric():
    st, _ = _state()
    b, _, _ = impact_parameters_1d(st)
    r_low, r_up = layer_radii(st)
    P = path_tensor(b, jnp.stack([r_low, r_low]), jnp.stack([r_up, r_up]), jnp.asarray([-np.pi / 2, 0.0, np.pi / 2]))
    np.testing.assert_allclose(P[:, 0], P[:, 1], rtol=1e-12, atol=0)
    np.testing.assert_allclose(P[:, 0], 0.5 * path_matrix_1d(st, b), rtol=1e-12, atol=0)


def test_transparent_and_opaque_limits():
    st, _ = _state()
    r_low, r_up = layer_radii(st)
    zero = {"line": jnp.zeros((NLAY, NWL))}
    D0, _ = compute_transit_depth_1d_os(st, {}, zero, None)
    np.testing.assert_allclose(D0, (r_low[0] / st["R_s"]) ** 2, rtol=1e-12)
    opaque = {"line": jnp.full((NLAY, NWL), 1e30)}
    D1, _ = compute_transit_depth_1d_os(st, {}, opaque, None)
    # midpoint rays with 2 b dr annuli cover exactly r_up^2 - r_low^2 per layer
    np.testing.assert_allclose(D1, (r_up[-1] / st["R_s"]) ** 2, rtol=1e-12)


def test_ck_with_one_g_point_equals_os():
    st, rng = _state()
    k = jnp.asarray(10 ** rng.uniform(-4, 1, (NLAY, NWL)))
    D_os, _ = compute_transit_depth_1d_os(st, {}, {"line": k}, None)
    D_ck, _ = compute_transit_depth_1d_ck(st, {}, {"line": k[:, :, None]}, {"g_weights": jnp.ones(1)})
    np.testing.assert_allclose(D_ck, D_os, rtol=1e-13)


@pytest.mark.parametrize("mode", ["os", "ck"])
def test_contribution_function_sums_to_one(mode):
    st, rng = _state(contri=True)
    if mode == "os":
        comps = {"line": jnp.asarray(10 ** rng.uniform(-4, 1, (NLAY, NWL)))}
        _, cf = compute_transit_depth_1d_os(st, {}, comps, None)
    else:
        comps = {"line": jnp.asarray(np.sort(10 ** rng.uniform(-4, 1, (NLAY, NWL, NG)), axis=-1))}
        _, cf = compute_transit_depth_1d_ck(st, {}, comps, _opac())
    np.testing.assert_allclose(cf.sum(axis=0), 1.0, rtol=1e-10)


def test_partial_cloud_mixing_limits():
    st, rng = _state()
    comps = {"line": jnp.asarray(10 ** rng.uniform(-4, 1, (NLAY, NWL))),
             "cloud": jnp.asarray(10 ** rng.uniform(-3, 0, (NLAY, NWL)))}
    D_cloudy, _ = compute_transit_depth_1d_os(st, {}, comps, None)
    D_clear, _ = compute_transit_depth_1d_os(st, {}, {"line": comps["line"]}, None)
    D_f1, _ = compute_transit_depth_1d_os(st, {"f_cloud": jnp.asarray(1.0)}, comps, None)
    D_f0, _ = compute_transit_depth_1d_os(st, {"f_cloud": jnp.asarray(0.0)}, comps, None)
    np.testing.assert_allclose(D_f1, D_cloudy, rtol=1e-13)
    np.testing.assert_allclose(D_f0, D_clear, rtol=1e-13)


def test_gradients_finite():
    st, rng = _state()
    comps = {"line": jnp.asarray(np.sort(10 ** rng.uniform(-4, 1, (NLAY, NWL, NG)), axis=-1))}

    def total(R0, rho):
        return jnp.sum(compute_transit_depth_1d_ck({**st, "R0": R0, "rho_lay": rho}, {}, comps, _opac())[0])

    g_R0, g_rho = jax.grad(total, argnums=(0, 1))(st["R0"], st["rho_lay"])
    assert bool(jnp.isfinite(g_R0)) and bool(jnp.all(jnp.isfinite(g_rho)))
    # path tensor also differentiable w.r.t. the radii when the rays are fixed
    b, _, _ = impact_parameters_1d(st)
    g = jax.grad(lambda R0: jnp.sum(path_matrix_1d({**st, "R0": R0}, b)))(st["R0"])
    assert bool(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Multi-column (sectors x zenith slices)
# ---------------------------------------------------------------------------

from exo_skryer.RT_trans_geometry import (  # noqa: E402
    shared_ray_grid, uniform_sector_fractions, uniform_theta_edges,
)
from exo_skryer.RT_trans_core_os import transit_depth_columns, vertical_tau  # noqa: E402
from exo_skryer.RT_trans_core_ck import (  # noqa: E402
    compute_transit_depth_1d_ck_trans, transit_depth_columns_ro,
)


def _tile_columns(st, n_sector, n_slice):
    r_low, r_up = layer_radii(st)
    shape = (n_sector, n_slice, NLAY)
    return jnp.broadcast_to(r_low, shape), jnp.broadcast_to(r_up, shape)


def test_shared_ray_grid_reduces_to_1d():
    st, _ = _state()
    b1, a1, base1 = impact_parameters_1d(st)
    b, a, base = shared_ray_grid(*layer_radii(st))
    np.testing.assert_allclose(b, b1, rtol=1e-13)
    np.testing.assert_allclose(a, a1, rtol=1e-9)
    np.testing.assert_allclose(base, base1, rtol=1e-15)


@pytest.mark.parametrize("n_sector,n_slice", [(1, 1), (2, 1), (1, 6), (4, 6)])
@pytest.mark.parametrize("mode", ["os", "ck"])
def test_identical_columns_reproduce_1d(n_sector, n_slice, mode):
    st, rng = _state()
    if mode == "os":
        k = jnp.asarray(10 ** rng.uniform(-4, 1, (NLAY, NWL)))
        D1, _ = compute_transit_depth_1d_os(st, {}, {"line": k}, None)
        gw = None
    else:
        k = jnp.asarray(np.sort(10 ** rng.uniform(-4, 1, (NLAY, NWL, NG)), axis=-1))
        D1, _ = compute_transit_depth_1d_ck(st, {}, {"line": k}, _opac())
        gw = _opac()["g_weights"]
    dtau = vertical_tau(k, st["rho_lay"], st["dz"])
    r_low, r_up = _tile_columns(st, n_sector, n_slice)
    D, _ = transit_depth_columns(r_low, r_up, uniform_theta_edges(n_slice), uniform_sector_fractions(n_sector),
                                 lambda j: jnp.broadcast_to(dtau, (n_slice,) + dtau.shape), st["R_s"], gw)
    np.testing.assert_allclose(D, D1, rtol=1e-9)


@pytest.mark.parametrize("n_sector,n_slice", [(1, 1), (3, 4)])
def test_identical_columns_reproduce_1d_ck_trans(n_sector, n_slice):
    st, rng = _state()
    nspec = 3
    sigma = jnp.asarray(10 ** rng.uniform(-2, 3, (nspec, NLAY, NWL, NG)))
    vmr = jnp.asarray(10 ** rng.uniform(-6, -2, (nspec, NLAY)))
    cont = jnp.asarray(10 ** rng.uniform(-4, 0, (NLAY, NWL)))
    D1, _ = compute_transit_depth_1d_ck_trans(st, {}, {"line_perspecies": sigma, "vmr_perspecies": vmr,
                                                       "cia": cont}, _opac())
    scale = st["rho_lay"] * st["dz"]
    dcont = cont * scale[:, None]
    dspec = sigma * vmr[:, :, None, None] * scale[None, :, None, None]
    r_low, r_up = _tile_columns(st, n_sector, n_slice)
    fn = lambda j: (jnp.broadcast_to(dcont, (n_slice,) + dcont.shape),
                    jnp.broadcast_to(dspec[:, None], (nspec, n_slice) + dspec.shape[1:]))
    D, _ = transit_depth_columns_ro(r_low, r_up, uniform_theta_edges(n_slice), uniform_sector_fractions(n_sector),
                                    fn, st["R_s"], _opac()["g_weights"])
    np.testing.assert_allclose(D, D1, rtol=1e-9)


def test_rays_below_a_raised_column_are_opaque():
    """Transparent atmosphere, two sectors; sector 1's column sits 3 layers higher. Rays below
    its bottom are blocked in that sector only."""
    st, _ = _state()
    r_low, r_up = layer_radii(st)
    shift = r_up[2] - r_low[0]
    R_low = jnp.stack([r_low, r_low + shift])[:, None, :]
    R_up = jnp.stack([r_up, r_up + shift])[:, None, :]
    zero = jnp.zeros((1, NLAY, NWL))
    D, _ = transit_depth_columns(R_low, R_up, uniform_theta_edges(1), uniform_sector_fractions(2),
                                 lambda j: zero, st["R_s"])
    b, area, base = shared_ray_grid(R_low, R_up)
    blocked = 0.5 * jnp.sum(jnp.where(b < r_low[0] + shift, area, 0.0))
    np.testing.assert_allclose(D * st["R_s"] ** 2, base**2 + blocked, rtol=1e-12)
    assert 0 < int(jnp.sum(b < r_low[0] + shift)) < b.size


def test_columns_gradients_finite():
    st, rng = _state()
    k = jnp.asarray(10 ** rng.uniform(-4, 1, (NLAY, NWL)))
    dtau = vertical_tau(k, st["rho_lay"], st["dz"])
    r_low, r_up = _tile_columns(st, 2, 3)

    def total(scale_r, scale_tau):
        D, _ = transit_depth_columns(r_low * scale_r, r_up * scale_r, uniform_theta_edges(3),
                                     uniform_sector_fractions(2),
                                     lambda j: scale_tau * jnp.broadcast_to(dtau, (3,) + dtau.shape), st["R_s"])
        return jnp.sum(D)

    g = jax.grad(total, argnums=(0, 1))(1.0, 1.0)
    assert all(bool(jnp.isfinite(x)) for x in g)


# ---------------------------------------------------------------------------
# 2D day-night layout and tagging
# ---------------------------------------------------------------------------

from exo_skryer.RT_trans_geometry import day_night_slices  # noqa: E402
from exo_skryer.limb_asymmetry import split_limb_parameter_dict, validate_limb_parameter_names  # noqa: E402


@pytest.mark.parametrize("n_zenith", [2, 3, 6, 10])
@pytest.mark.parametrize("beta_deg", [0.0, 20.0, 90.0])
def test_day_night_slices(n_zenith, beta_deg):
    edges, w = day_night_slices(n_zenith, jnp.deg2rad(beta_deg))
    assert edges.shape == (n_zenith + 1,) and w.shape == (n_zenith,)
    np.testing.assert_allclose([edges[0], edges[-1]], [-np.pi / 2, np.pi / 2])
    assert bool(jnp.all(jnp.diff(edges) >= 0))
    assert float(w[0]) == 0.0 and float(w[-1]) == 1.0
    assert bool(jnp.all((w >= 0) & (w <= 1)) & jnp.all(jnp.diff(w) >= 0))
    if n_zenith > 2:
        np.testing.assert_allclose([edges[1], edges[-2]], [-np.deg2rad(beta_deg) / 2, np.deg2rad(beta_deg) / 2])


def test_identical_day_night_columns_any_slicing_reproduce_1d():
    st, rng = _state()
    k = jnp.asarray(10 ** rng.uniform(-4, 1, (NLAY, NWL)))
    D1, _ = compute_transit_depth_1d_os(st, {}, {"line": k}, None)
    dtau = vertical_tau(k, st["rho_lay"], st["dz"])
    for n_z, beta in ((2, 10.0), (6, 30.0), (10, 80.0)):
        edges, _ = day_night_slices(n_z, jnp.deg2rad(beta))
        r_low, r_up = _tile_columns(st, 1, n_z)
        D, _ = transit_depth_columns(r_low, r_up, edges, jnp.ones(1),
                                     lambda j: jnp.broadcast_to(dtau, (n_z,) + dtau.shape), st["R_s"])
        np.testing.assert_allclose(D, D1, rtol=1e-9)


def test_day_night_tagging_and_offsets():
    p = {"R_p_joint": 1.0, "T_iso_day": 2000.0, "delta_T_iso_night": -800.0, "beta_dn_joint": 20.0}
    joint, day, night = split_limb_parameter_dict({k: jnp.asarray(v) for k, v in p.items()}, ("day", "night"))
    assert float(night["T_iso"]) == 1200.0 and float(day["T_iso"]) == 2000.0
    assert set(joint) == {"R_p", "beta_dn"}
    with pytest.raises(ValueError):
        validate_limb_parameter_names({"T_iso_east": 1, "T_iso_west": 1}, ("day", "night"))
    with pytest.raises(ValueError):
        validate_limb_parameter_names({"T_iso_day": 1}, ("day", "night"))


def test_3d_tags_and_offset_chain():
    from exo_skryer.limb_asymmetry import ANCHOR_TAGS, split_limb_tag
    A = ANCHOR_TAGS["transit_3d"]
    assert split_limb_tag("T_iso_east_day") == ("T_iso", "east_day")
    assert split_limb_tag("log_10_f_H2O_west_night") == ("log_10_f_H2O", "west_night")
    p = {"R_p_joint": 1.0, "T_iso_east_day": 2000.0, "delta_T_iso_west_day": -100.0,
         "delta_T_iso_east_night": -600.0, "delta_T_iso_west_night": -700.0}
    _, ed, wd, en, wn = split_limb_parameter_dict({k: jnp.asarray(v) for k, v in p.items()}, A)
    assert [float(d["T_iso"]) for d in (ed, wd, en, wn)] == [2000.0, 1900.0, 1400.0, 1200.0]
    with pytest.raises(ValueError):   # days cannot come from nights
        validate_limb_parameter_names({"T_iso_east_night": 1, "T_iso_west_night": 1,
                                       "delta_T_iso_east_day": 0, "delta_T_iso_west_day": 0}, A)


# ---------------------------------------------------------------------------
# TRIDENT parameterisation (values checked against POSEIDON compute_T_field_gradient,
# compute_X_field_gradient and angular_grids to <= 2e-15; see experiments/TRIDENT_benchmark)
# ---------------------------------------------------------------------------

from exo_skryer.build_model_trans import trident_angles, trident_high_value, trident_vertical  # noqa: E402


def test_trident_angles_layout():
    e, c, w = trident_angles(None, 0.0)
    np.testing.assert_allclose(e, [-np.pi / 2, np.pi / 2]); np.testing.assert_allclose(c, [0.0])
    e, c, w = trident_angles(4, 40.0)
    a = np.deg2rad(40.0)
    np.testing.assert_allclose(e, [-np.pi / 2, -a / 2, -a / 4, 0.0, a / 4, a / 2, np.pi / 2], atol=1e-15)
    np.testing.assert_allclose(float(jnp.sum(w)), np.pi)
    np.testing.assert_allclose(c, 0.5 * (e[1:] + e[:-1]), atol=1e-15)


def test_trident_high_value_piecewise():
    bar, dterm, ddn, a, b = 1400.0, 500.0, 1000.0, 40.0, 20.0
    f = lambda ph, th: float(trident_high_value(jnp.asarray(ph), jnp.asarray(th), bar, dterm, ddn, a, b))
    assert f(-1.0, 0.0) == bar + dterm / 2          # evening, terminator plane
    assert f(1.0, 0.0) == bar - dterm / 2           # morning
    assert f(0.0, 0.0) == bar                       # middle of the azimuthal transition
    assert f(-1.0, -1.0) == bar + dterm / 2 + ddn / 2   # evening dayside
    assert f(1.0, 1.0) == bar - dterm / 2 - ddn / 2     # morning nightside
    q = np.deg2rad(b) / 4                            # halfway into the day-night transition (night side)
    np.testing.assert_allclose(f(0.0, q), bar - 0.5 * ddn / 2)


def test_trident_vertical_profile():
    p = jnp.asarray([1e-7, 1e-5, 1e-2, 10.0, 100.0])
    T = np.asarray(trident_vertical(p, 1000.0, 2000.0))
    np.testing.assert_allclose(T[[0, 1]], 1000.0); np.testing.assert_allclose(T[[3, 4]], 2000.0)
    np.testing.assert_allclose(T[2], 1000.0 + 1000.0 * 3.0 / 6.0)    # log10(1e-2/1e-5) / log10(10/1e-5)
