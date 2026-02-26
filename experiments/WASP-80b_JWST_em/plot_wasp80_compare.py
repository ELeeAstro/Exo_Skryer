#!/usr/bin/env python3
"""Plot WASP-80 spectra for side-by-side comparison."""

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required. Install with: python3 -m pip install matplotlib"
    ) from exc


def load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}")
    return data[:, 0], data[:, 1]


def resolve_spectrum_path(base: Path) -> Path:
    candidates = [
        base / "WASP-80_spectrum.txt",
        base / "WASP80_spectrum.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find spectrum file. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def main() -> None:
    base = Path(__file__).resolve().parent
    file_a = base / "WASP-80.txt"
    file_b = resolve_spectrum_path(base)

    x_a, y_a = load_two_column(file_a)
    x_b, y_b = load_two_column(file_b)
    y_b = y_b * 10.0

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_a, y_a, lw=1.2, label=file_a.name)
    ax.plot(x_b, y_b, lw=1.0, alpha=0.85, label=f"{file_b.name} x10")

    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    ax.set_title("WASP-80 spectrum comparison")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = base / "WASP-80_comparison.png"
    fig.savefig(out, dpi=180)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
