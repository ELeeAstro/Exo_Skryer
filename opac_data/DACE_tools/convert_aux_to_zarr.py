"""Convert FastChem grids (.npz) and Mie optics grids (.h5) to .zarr.zip, verified bit-identical.

    python convert_aux_to_zarr.py FILE [FILE ...] [--remove-original]

- FastChem .npz: every array copied unchanged; the mixing-ratio cube gets one chunk per species
  (the last axis), so a run reads only its species (~ms instead of ~2 s for the 5D grid).
- Optics .h5: every dataset (with its path) and the file attributes copied unchanged.
Strings are stored as fixed-width bytes (zarr v3); the readers decode them.
Each output (<name>.zarr.zip next to the input) is re-read and compared with the input array by
array (shape, dtype, values; strings compared as text). The input is removed only with
--remove-original and only if that check passes. Codec: Blosc lz4, clevel 1, shuffle.
Loaders pick a .zarr.zip next to a configured .npz/.h5 automatically.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.storage import ZipStore

LZ4 = BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")


def _bytes(v):
    v = np.asarray(v)
    return v.astype(f"S{max(1, v.dtype.itemsize // 4)}") if v.dtype.kind == "U" else v


def read_npz(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}, {}


def read_h5(path):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
        attrs = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else
                     (v.tolist() if hasattr(v, "tolist") else str(v))) for k, v in f.attrs.items()}
    return out, attrs


def write(dst, arrays, attrs):
    if dst.exists():
        dst.unlink()
    with ZipStore(dst, mode="w") as st:
        g = zarr.create_group(store=st, attributes=attrs, zarr_format=3)
        for k, v in arrays.items():
            v = _bytes(v)
            chunks = v.shape[:-1] + (1,) if "mixing_ratios" in k and v.ndim == 5 else v.shape
            g.create_array(k, data=v, chunks=chunks, compressors=LZ4)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "SU" or b.dtype.kind in "SU":
        txt = lambda x: [s.decode() if isinstance(s, bytes) else str(s) for s in x.ravel()]
        return a.shape == b.shape and txt(a) == txt(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def verify(dst, arrays, attrs):
    g = zarr.open_group(store=ZipStore(dst, mode="r"), mode="r")
    if set(g.array_keys()) != set(arrays) and not all(k in g for k in arrays):
        return False
    return all(same(g[k][:], v) for k, v in arrays.items()) and all(g.attrs.get(k) == v for k, v in attrs.items())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--remove-original", action="store_true")
    args = ap.parse_args()
    for src in args.files:
        if src.suffix == ".npz":
            arrays, attrs = read_npz(src)
        elif src.suffix in (".h5", ".hdf5"):
            arrays, attrs = read_h5(src)
        else:
            print(f"[skip] {src} (not .npz/.h5)")
            continue
        dst = src.with_name(src.stem + ".zarr.zip")
        write(dst, arrays, attrs)
        ok = verify(dst, arrays, attrs)
        msg = f"{src.name} -> {dst.name}: {src.stat().st_size / 1e6:.1f} -> {dst.stat().st_size / 1e6:.1f} MB, identical {ok}"
        if not ok:
            dst.unlink()
            print("[FAIL]", msg, "(output removed, original untouched)")
            continue
        if args.remove_original:
            os.remove(src)
            msg += ", original removed"
        print("[ok]", msg, flush=True)
