"""Rechunk opacity .zarr.zip tables for fast reading (no regeneration from DACE needed).

    python rechunk_zarr_zip.py TABLE.zarr.zip [TABLE2 ...] [--wl-chunk 2048] [--replace] [--keep-backup]

Old tables store cross_section in (1, 1, 2048) chunks: ~70,000 tiny zip members per os table,
and the per-chunk overhead dominates the read (~5 s for an R20000 table). Here every array is
rewritten with full chunks along all axes except wavelength, which gets --wl-chunk points
(default 2048). That gives tens of chunks: a full read takes ~0.1 s, and a wavelength slice reads
only the chunks it needs. Codec: Blosc lz4, clevel 1, shuffle (as before). Files come out ~20% smaller.

For each table it writes <name>.rechunked.zarr.zip next to it, then re-reads every array and
checks it is bit-identical to the original (shapes, dtypes, values, attributes). With --replace
the original is replaced only after that check (with --keep-backup it is kept as <name>.bak).
Group attributes are written once (old writers left a duplicate zarr.json entry in the zip).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.storage import ZipStore

LZ4 = BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")


def chunks_for(shape, wl_axis, wl_chunk):
    return tuple(min(wl_chunk, n) if ax == wl_axis else n for ax, n in enumerate(shape))


def rechunk(src: Path, wl_chunk: int) -> Path:
    dst = src.with_name(src.name.replace(".zarr.zip", ".rechunked.zarr.zip"))
    g = zarr.open_group(store=ZipStore(src, mode="r"), mode="r")
    attrs = dict(g.attrs)
    arrays = {k: g[k][:] for k in g.array_keys()}
    n_wl = arrays["wavelength"].shape[0] if "wavelength" in arrays else None
    if dst.exists():
        dst.unlink()
    with ZipStore(dst, mode="w") as st:
        out = zarr.create_group(store=st, attributes=attrs, zarr_format=3)
        for k, a in arrays.items():
            # the wavelength axis is the one matching the wavelength array (if any)
            wl_axis = next((ax for ax, n in enumerate(a.shape) if n == n_wl and a.ndim > 1), None)
            out.create_array(k, data=a, chunks=chunks_for(a.shape, wl_axis, wl_chunk), compressors=LZ4)
    return dst


def identical(a_path: Path, b_path: Path) -> bool:
    ga = zarr.open_group(store=ZipStore(a_path, mode="r"), mode="r")
    gb = zarr.open_group(store=ZipStore(b_path, mode="r"), mode="r")
    if dict(ga.attrs) != dict(gb.attrs) or set(ga.array_keys()) != set(gb.array_keys()):
        return False
    for k in ga.array_keys():
        x, y = ga[k][:], gb[k][:]
        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
            return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tables", nargs="+", type=Path)
    ap.add_argument("--wl-chunk", type=int, default=2048)
    ap.add_argument("--replace", action="store_true", help="replace each original after verification")
    ap.add_argument("--keep-backup", action="store_true", help="with --replace, keep the original as .bak")
    args = ap.parse_args()
    for src in args.tables:
        if src.name.endswith(".rechunked.zarr.zip") or not src.name.endswith(".zarr.zip"):
            print(f"[skip] {src}")
            continue
        t = time.time()
        dst = rechunk(src, args.wl_chunk)
        ok = identical(src, dst)
        msg = f"{src.name}: {src.stat().st_size / 1e6:.0f} -> {dst.stat().st_size / 1e6:.0f} MB, {time.time() - t:.1f} s, identical {ok}"
        if not ok:
            dst.unlink()
            print("[FAIL]", msg, "(rechunked file removed, original untouched)")
            continue
        if args.replace:
            if args.keep_backup:
                os.replace(src, src.with_name(src.name + ".bak"))
            os.replace(dst, src)
            msg += " -> replaced" + (" (backup kept)" if args.keep_backup else "")
        print("[ok]", msg, flush=True)
