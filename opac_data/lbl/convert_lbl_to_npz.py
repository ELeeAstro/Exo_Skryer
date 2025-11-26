#!/usr/bin/env python3
"""
convert_opac_data_to_npz.py
---------------------------

Convert all legacy opacity tables in a directory (e.g. opac_data/)
from the text format

    mol_name
    nT nP nb flag
    T[0..nT-1]
    P[0..nP-1]
    lam[0..nb-1]
    wn_line_ignored
    sigma[t=0,p=0,:]
    sigma[t=0,p=1,:]
    ...
    sigma[t=nT-1,p=nP-1,:]

into compressed NumPy .npz files with keys:

    p_bar  : (nP,)
    t_k    : (nT,)
    lam    : (nb,)
    sigma  : (nP, nT, nb)   # NOTE: P, then T, then λ
    mol    : scalar string

This matches the expectations of xs_registry._load_npz().
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


# ----------------- helpers to mirror your current reader -----------------

def _read_nonempty_line(f) -> str:
    line = f.readline()
    while line and line.strip() == "":
        line = f.readline()
    if not line:
        raise EOFError("Unexpected EOF while reading opacity file.")
    return line.strip()


def _fromstring(line: str, count: int | None = None) -> np.ndarray:
    arr = np.fromstring(line, sep=" ")
    if count is not None and arr.size != count:
        raise ValueError(f"Expected {count} numbers, found {arr.size}.")
    return arr


def read_legacy_opacity(path: Path) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read one legacy text/xsc file and return:
        mol_name, P_bar, T_k, lam_um, sigma_p_t_lam
    where sigma_p_t_lam has shape (nP, nT, nLam).
    """
    with path.open("r") as f:
        mol = _read_nonempty_line(f).strip()
        print(f"[read] {mol} @ {path}")

        nT, nP, nb, _flag = map(int, _read_nonempty_line(f).split())

        T  = _fromstring(_read_nonempty_line(f), count=nT)  # K
        P  = _fromstring(_read_nonempty_line(f), count=nP)  # bar
        wl = _fromstring(_read_nonempty_line(f), count=nb)  # μm
        wn  = _fromstring(_read_nonempty_line(f), count=nb) # wn                      # wn (unused)

        sigma_tp_lam = np.empty((nT, nP, nb), dtype=np.float64)
        for t in range(nT):
            for p in range(nP):
                sigma_tp_lam[t, p, :] = _fromstring(_read_nonempty_line(f), count=nb)

    if not np.all(np.diff(wl) > 0):
        raise ValueError(f"Wavelength grid must be strictly increasing (μm) in {path}")

    return mol, P, T, wl,sigma_tp_lam


def write_npz(out_path: Path, mol: str,
              P_bar: np.ndarray, T_k: np.ndarray,
              lam_um: np.ndarray, sigma_tp_lam: np.ndarray,
              overwrite: bool = False) -> None:
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} already exists (use --overwrite to replace).")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        P_bar=P_bar,
        T=T_k,
        wl=lam_um,
        sig=sigma_tp_lam,
        mol=np.array(mol, dtype=object),
    )
    print(f"[write] {out_path}")


# ----------------- directory driver -----------------

def convert_directory(
    src_dir: Path,
    pattern: str = "*.txt",
    suffix: str = ".npz",
    overwrite: bool = False,
) -> None:
    """
    Convert all files in src_dir matching pattern to .npz.
    By default: pattern='*.txt' and output name is stem + '.npz'.
    """
    src_dir = src_dir.resolve()
    if not src_dir.is_dir():
        raise NotADirectoryError(f"{src_dir} is not a directory")

    files = sorted(src_dir.glob(pattern))
    if not files:
        print(f"[info] No files matching {pattern} in {src_dir}")
        return

    print(f"[info] Converting {len(files)} file(s) in {src_dir}")
    for path in files:
        try:
            mol, P, T, lam, sigma = read_legacy_opacity(path)
            out_path = path.with_suffix(suffix)
            write_npz(out_path, mol, P, T, lam, sigma, overwrite=overwrite)
        except Exception as exc:
            print(f"[error] Failed to convert {path}: {exc}")


# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert legacy opacity tables in a directory to NumPy .npz format."
    )
    ap.add_argument(
        "--src",
        type=str,
        default="./",
        help="Source directory containing legacy opacity files (default: opac_data)",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="Glob pattern for input files (default: *.txt)",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default=".npz",
        help="Output file suffix (default: .npz)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz files if they already exist.",
    )
    args = ap.parse_args()

    convert_directory(
        src_dir=Path(args.src),
        pattern=args.pattern,
        suffix=args.suffix,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
