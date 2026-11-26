"""Smoke-test every limb-survey setup: build it, evaluate the nautilus log-likelihood at the initial
point (finite?), check that free offsets enter the likelihood, and time one forward call.
    python check_setups.py [planet/retrieval_x.yaml ...]"""
import sys, time
from pathlib import Path
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from exo_skryer.read_yaml import read_yaml
from exo_skryer.run_retrieval import build_retrieval_inputs
from exo_skryer.sampler_nautilus_NS import build_prior_nautilus, build_loglikelihood_nautilus
HERE = Path(__file__).resolve().parent
files = [Path(a) for a in sys.argv[1:]] or sorted(HERE.glob("*/retrieval_*.yaml"))
for f in files:
    f = (HERE / f) if not f.is_absolute() else f
    cfg = read_yaml(f); obs, fm = build_retrieval_inputs(cfg, f.parent)
    prior, names = build_prior_nautilus(cfg); ll = build_loglikelihood_nautilus(cfg, obs, fm, names)
    p = {q.name: float(q.init) for q in cfg.params if q.dist != "delta"}
    l0 = ll(p)
    free_off = [n for n in names if n.startswith("offset_")]
    moved = all(ll({**p, n: p[n] + 50.0}) != l0 for n in free_off)
    pj = {k: jnp.asarray(v) for k, v in p.items() if not k.startswith("offset_")}
    jax.block_until_ready(fm(pj)); t = time.time()
    for _ in range(5): jax.block_until_ready(fm(pj))
    print(f"CHECK {f.parent.name}/{f.name}: {len(names)} free, n_data {len(obs['y'])}, nwl {len(fm.runtime_args[0])}, "
          f"logL(init) {l0:.3g} finite {np.isfinite(l0)}, offsets {free_off or '-'} wired {moved}, "
          f"fwd {(time.time()-t)/5*1e3:.1f} ms", flush=True)
