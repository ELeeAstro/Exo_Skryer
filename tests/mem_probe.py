"""
mem_probe.py
============

Per-batch memory of the compiled forward model and log-likelihood gradient.

Builds the model exactly as ``run_retrieval`` does, then for each batch size B
compiles ``lax.map(f, P, batch_size=B)`` over B prior draws (one vmapped batch)
and reports XLA's memory analysis:

- temp: working memory of one batch (what grows with B)
- args: inputs, incl. the static opacity tables (fixed)

Usage::

    python tests/mem_probe.py --config experiments/_tmp_mem_test/retrieval_config.yaml \
        --batches 1 8 32 [--set opac.T_interp=pchip] [--check out.npz]

``--set`` overrides config entries; ``--check`` saves (or compares against)
spectra and gradients for 20 fixed prior draws.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _set(cfg, dotted: str):
    key, val = dotted.split("=", 1)
    *path, last = key.split(".")
    obj = cfg
    for p in path:
        obj = getattr(obj, p)
    low = val.strip().lower()
    val = {"true": True, "false": False}.get(low, val)
    setattr(obj, last, val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--check", default=None, help="npz to save, or compare against if it exists")
    ap.add_argument("--no-grad", action="store_true")
    ap.add_argument("--time", type=int, default=3, help="timed calls per compiled function")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from exo_skryer.read_yaml import read_yaml
    from exo_skryer.run_retrieval import build_retrieval_inputs
    from exo_skryer.build_common import runtime_args_of

    cfg_path = Path(args.config).resolve()
    cfg = read_yaml(cfg_path)
    for s in args.set:
        _set(cfg, s)
    os.chdir(cfg_path.parent)
    obs, fm = build_retrieval_inputs(cfg, cfg_path.parent)
    runtime = runtime_args_of(fm)

    free = [p for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]
    names = [p.name for p in free]
    lo = np.array([float(p.low) for p in free])
    hi = np.array([float(p.high) for p in free])
    y = jnp.asarray(obs["y"])
    dy = jnp.asarray(obs["dy"])

    def draws(n, seed=0):
        u = np.random.default_rng(seed).uniform(size=(n, len(names)))
        x = lo + u * (hi - lo)
        return {k: jnp.asarray(x[:, i]) for i, k in enumerate(names)}

    def model(p, rt):
        return fm.impl(p, *rt)

    def loglike(p, rt):
        r = (y - model(p, rt)) / dy
        return -0.5 * jnp.sum(r * r)

    grad_ll = jax.grad(loglike)

    def mb(x):
        return x / 1e6

    rows = []
    for B in args.batches:
        P = draws(B, seed=B)
        fns = [("fwd", lambda P, rt: jax.lax.map(lambda p: model(p, rt), P, batch_size=B))]
        if not args.no_grad:
            fns.append(("grad", lambda P, rt: jax.lax.map(lambda p: grad_ll(p, rt), P, batch_size=B)))
        for name, fn in fns:
            jf = jax.jit(fn)
            t0 = time.perf_counter()
            comp = jf.lower(P, runtime).compile()
            t_comp = time.perf_counter() - t0
            ma = comp.memory_analysis()
            jax.block_until_ready(comp(P, runtime))
            t0 = time.perf_counter()
            for _ in range(args.time):
                jax.block_until_ready(comp(P, runtime))
            t_call = (time.perf_counter() - t0) / max(args.time, 1)
            rows.append((name, B, mb(ma.temp_size_in_bytes), mb(ma.argument_size_in_bytes),
                         mb(ma.output_size_in_bytes), t_comp, t_call))
            print(f"[mem] {name:4s} B={B:<3d} temp={rows[-1][2]:9.1f} MB  args={rows[-1][3]:8.1f} MB  "
                  f"out={rows[-1][4]:7.2f} MB  compile={t_comp:6.1f} s  call={t_call*1e3:8.1f} ms", flush=True)

    # Per-particle slope of temp memory (MB per particle) between smallest and largest B
    for name in ("fwd", "grad"):
        r = [x for x in rows if x[0] == name]
        if len(r) >= 2:
            (b0, t0_), (b1, t1_) = (r[0][1], r[0][2]), (r[-1][1], r[-1][2])
            print(f"[mem] {name} temp slope: {(t1_ - t0_) / (b1 - b0):.2f} MB per particle", flush=True)

    if args.check:
        P = draws(20, seed=1234)
        spec = np.asarray(jax.jit(jax.vmap(model, in_axes=(0, None)))(P, runtime))
        g = jax.jit(jax.vmap(grad_ll, in_axes=(0, None)))(P, runtime)
        grads = np.stack([np.asarray(g[k]) for k in names], axis=1)
        ck = Path(args.check)
        if ck.exists():
            ref = np.load(ck)
            ds = np.max(np.abs(spec - ref["spec"]) / np.abs(ref["spec"]))
            scale = np.maximum(np.abs(ref["grads"]), 1e-12 * np.abs(ref["grads"]).max())
            dg = np.max(np.abs(grads - ref["grads"]) / scale)
            print(f"[check] vs {ck.name}: max rel spectrum diff {ds:.2e}, max rel gradient diff {dg:.2e}")
        else:
            np.savez(ck, spec=spec, grads=grads, names=np.array(names))
            print(f"[check] saved reference to {ck}")


if __name__ == "__main__":
    main()
