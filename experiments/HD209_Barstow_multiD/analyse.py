"""Summarise a finished retrieval in this directory and plot its spectrum fit.

    python analyse.py 1D [1_5D 2D 3D_iceberg ...]
    python analyse.py nautilus_250/1D ...     (runs in a subdirectory; summary.md written there)

Prints evidence, sampler cost, best-fit chi2 and parameter medians (16-84%), writes
<geom>/fit_spectrum.png (data, best fit, 1-sigma band; limb/side components for multi-D)
and appends a row to summary.md.
"""
import re, sys
from pathlib import Path
import numpy as np
import arviz as az
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exo_skryer.read_yaml import read_yaml
from exo_skryer.run_retrieval import build_retrieval_inputs
from exo_skryer.build_model import build_forward_model

HERE = Path(__file__).resolve().parent
COMPONENTS = {"1_5D": ("east", "west"), "3D_iceberg": ("east", "west")}


def analyse(geom):
    d = HERE / geom
    cfg = read_yaml(d / "retrieval_config.yaml")
    tree = az.from_netcdf(d / "posterior.nc")
    at = dict(tree.attrs)
    post = tree.posterior
    names = [q.name for q in cfg.params if q.dist != "delta"]
    samples = {n: np.asarray(post[n]).ravel() for n in names}
    ll = np.asarray(tree.sample_stats["log_likelihood"]).ravel()
    log = next((d / f).read_text() for f in ("run.log", "dynesty_run.log") if (d / f).exists())
    took = re.findall(r"\[done\] Full model took: (.*)", log)

    obs, fm = build_retrieval_inputs(cfg, d)
    fm_hi = build_forward_model(cfg, obs, exp_dir=d, return_highres=True)
    y, dy, wl = (np.asarray(obs[k]) for k in ("y", "dy", "wl"))
    ibest = int(np.argmax(ll))
    best = {n: jnp.asarray(samples[n][ibest]) for n in names}
    m_best = np.asarray(fm(best))
    chi2 = float(np.sum(((m_best - y) / dy) ** 2))
    dof = y.size - len(names)

    rng = np.random.default_rng(0)
    idx = rng.choice(ll.size, size=min(200, ll.size), replace=False)
    draws = np.stack([np.asarray(fm({n: jnp.asarray(samples[n][i]) for n in names})) for i in idx])
    lo, hi = np.percentile(draws, [16, 84], axis=0)

    print(f"\n===== {geom} =====")
    print(f"logZ = {at['logZ']:.2f} +- {at['logZ_err']:.2f}; max logL = {at['max_log_likelihood']:.2f}; "
          f"n_like = {int(at['n_like'])}; samples = {ll.size}; wall time = {took[-1] if took else '?'}")
    print(f"best-fit chi2 = {chi2:.1f} for N = {y.size}, {len(names)} free (reduced {chi2 / dof:.2f})")
    rows = []
    for n in names:
        q16, q50, q84 = np.percentile(samples[n], [16, 50, 84])
        rows.append((n, q50, q84 - q50, q50 - q16, samples[n][ibest]))
        print(f"  {n:28s} {q50:10.3f} +{q84 - q50:8.3f} -{q50 - q16:8.3f}   (best {samples[n][ibest]:10.3f})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    o = np.argsort(wl)
    ax.errorbar(wl, y * 1e2, dy * 1e2, fmt="o", ms=3, color="k", lw=0.8, label="Sing+2016", zorder=3)
    ax.fill_between(wl[o], lo[o] * 1e2, hi[o] * 1e2, color="C0", alpha=0.3, label="1$\\sigma$ (200 draws)")
    ax.plot(wl[o], m_best[o] * 1e2, color="C0", lw=1.2, label=f"best fit ($\\chi^2$={chi2:.0f})")
    if Path(geom).name in COMPONENTS:
        r = fm_hi(best)
        for k, c, lab in zip(COMPONENTS[Path(geom).name], ("C3", "C2"), ("evening (east)", "morning (west)")):
            comp = np.asarray(r[f"binned_{k}"])  # full-circle equivalent
            ax.plot(wl[o], comp[o] * 1e2, color=c, lw=0.9, ls="--", label=lab + " limb, full-circle equiv.")
    ax.set_xscale("log"); ax.set_xlabel("wavelength [$\\mu$m]"); ax.set_ylabel("transit depth [%]")
    ax.set_title(f"HD 209458b {geom}: logZ = {at['logZ']:.1f}")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(d / "fit_spectrum.png", dpi=130); plt.close(fig)

    summ = d.parent / "summary.md"
    if not summ.exists():
        summ.write_text("| geometry | free | logZ | max logL | chi2 (N=%d) | n_like | wall time |\n|---|---|---|---|---|---|---|\n" % y.size)
    with open(summ, "a") as f:
        f.write(f"| {Path(geom).name} | {len(names)} | {at['logZ']:.2f} +- {at['logZ_err']:.2f} | {at['max_log_likelihood']:.2f} | "
                f"{chi2:.1f} | {int(at['n_like'])} | {took[-1] if took else '?'} |\n")
    return rows


if __name__ == "__main__":
    for g in sys.argv[1:]:
        analyse(g)
