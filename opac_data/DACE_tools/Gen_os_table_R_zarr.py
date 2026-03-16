import argparse
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import ZipStore


Avo = 6.02214076e23

N_HEADER_LINES = 14
SPECIES_START_INDEX = N_HEADER_LINES - 1
N_P = 34
P_LIST = [
    "n800", "n766", "n733", "n700", "n666", "n633", "n600", "n566", "n533", "n500",
    "n466", "n433", "n400", "n366", "n333", "n300", "n266", "n233", "n200", "n166",
    "n133", "n100", "n066", "n033", "p000", "p033", "p066", "p100", "p133", "p166",
    "p200", "p233", "p266", "p300",
]


@dataclass(frozen=True)
class SpeciesConfig:
    molname: str
    molw: float
    min_temp: float
    max_temp: float
    swns: str
    swne: str
    basedir: str


@dataclass(frozen=True)
class RunConfig:
    wl_s: float
    wl_e: float
    res: float
    wl_ext: str
    out_ext: str
    inform: int
    species: tuple[SpeciesConfig, ...]


def _parse_header(lines: list[str]) -> tuple[float, float, float, str, str, int]:
    if len(lines) < N_HEADER_LINES:
        raise ValueError(f"Expected at least {N_HEADER_LINES} header lines, got {len(lines)}.")

    values = {}
    for i, line in enumerate(lines[:N_HEADER_LINES]):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if i == 1:
            values["wl_s"] = float(line)
        elif i == 3:
            values["wl_e"] = float(line)
        elif i == 5:
            values["res"] = float(line)
        elif i == 7:
            values["wl_ext"] = line.strip()
        elif i == 9:
            values["out_ext"] = line.strip()
        elif i == 11:
            values["inform"] = int(line)

    missing = [k for k in ("wl_s", "wl_e", "res", "wl_ext", "out_ext", "inform") if k not in values]
    if missing:
        raise ValueError(f"Missing required header fields in input file: {missing}")

    return (
        values["wl_s"],
        values["wl_e"],
        values["res"],
        values["wl_ext"],
        values["out_ext"],
        values["inform"],
    )


def _parse_species_block(lines: list[str]) -> tuple[SpeciesConfig, ...]:
    species = []
    started = False

    for raw in lines[SPECIES_START_INDEX:]:
        line = raw.strip()
        if not line:
            if started:
                break
            continue
        if line.startswith("#"):
            continue

        started = True
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(f"Invalid species line: {raw!r}")

        species.append(
            SpeciesConfig(
                molname=str(parts[0]),
                molw=float(parts[1]),
                min_temp=float(parts[2]),
                max_temp=float(parts[3]),
                swns=str(parts[4]).zfill(5),
                swne=str(parts[5]).zfill(5),
                basedir=str(parts[6]),
            )
        )

    if not species:
        raise ValueError("No active species found after the header block.")

    return tuple(species)


def load_input_config(path: str) -> RunConfig:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    wl_s, wl_e, res, wl_ext, out_ext, inform = _parse_header(lines)
    species = _parse_species_block(lines)
    return RunConfig(
        wl_s=wl_s,
        wl_e=wl_e,
        res=res,
        wl_ext=wl_ext,
        out_ext=out_ext,
        inform=inform,
        species=species,
    )


def pressure_grid_log10() -> np.ndarray:
    vals = []
    for token in P_LIST:
        line = token.replace("n", "-").replace("p", "+")
        vals.append(float(line[0:2] + "." + line[2:]))
    return np.asarray(vals, dtype=np.float64)


def discover_temperature_grid(spec: SpeciesConfig) -> tuple[np.ndarray, list[str]]:
    files = glob(spec.basedir + "/*.bin")
    lb = len(spec.basedir)
    t_list = []
    for idx in range(len(files)):
        fname = files[idx][lb + 1:]
        t_list.append(fname[16:21])

    t_list = sorted(set(t_list))
    n_t = len(t_list) - 14 if spec.molname == "CO" else len(t_list)
    temps = np.zeros(n_t, dtype=np.float64)
    for idx in range(n_t):
        temps[idx] = float(t_list[idx])
    return temps, t_list


def build_wavelength_grid(wl_s: float, wl_e: float, res: float) -> tuple[np.ndarray, np.ndarray]:
    wn_l = 1.0 / (wl_e * 1e-4)
    wn_h = 1.0 / (wl_s * 1e-4)

    wno = wn_l
    wnoarr = []
    while wno < wn_h:
        wno = wno + wno / res
        wnoarr.append(wno)

    wnoarr = np.asarray(wnoarr, dtype=np.float64)
    wl = 1.0 / wnoarr[::-1] * 1e4
    wn = wnoarr[::-1]
    return wl, wn


def write_wavelength_file(path: str, wl: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(wl.shape[0]) + "\n")
        for idx, value in enumerate(wl, start=1):
            handle.write(f"{idx} {value}\n")


def _species_output_name(spec: SpeciesConfig, template: str, res: float, n_species: int) -> str:
    if n_species == 1:
        return template
    if "{species}" in template:
        return template.format(species=spec.molname)
    suffix = f"_R{int(round(res))}.txt"
    return f"{spec.molname}{suffix}"


def _wn_grid(spec: SpeciesConfig) -> np.ndarray:
    wne = float(spec.swne) + 0.01 if spec.molname in ("Na", "K") else float(spec.swne)
    return np.arange(float(spec.swns), wne, 0.01, dtype=np.float64)


def _opacity_path(spec: SpeciesConfig, temp_token: str, pressure_token: str) -> str:
    if spec.molname in ("Fe", "FeII"):
        return f"{spec.basedir}/{temp_token}_{pressure_token}.bin"
    return f"{spec.basedir}/Out_{spec.swns}_{spec.swne}_{temp_token}_{pressure_token}.bin"


def generate_species_tables(cfg: RunConfig, spec: SpeciesConfig, wl: np.ndarray, wn: np.ndarray) -> None:
    print(f"=== {spec.molname} ===")
    print(cfg.wl_s, cfg.wl_e, cfg.res)
    print(cfg.wl_ext)
    print(cfg.out_ext)
    print(cfg.inform)
    print(spec)

    p_log = pressure_grid_log10()
    p_bar = 10.0 ** p_log
    temps, temp_tokens = discover_temperature_grid(spec)
    wn_grid = _wn_grid(spec)

    print(P_LIST)
    print(p_bar)
    print(temp_tokens)
    print(temps.shape[0])
    print(wn_grid)

    nb = wl.shape[0]
    n_t = temps.shape[0]
    n_p = p_bar.shape[0]

    k_data = np.zeros(len(wn_grid), dtype=np.float64)
    k_os = np.zeros(nb, dtype=np.float64)
    k_os_cube = np.zeros((n_t, n_p, nb), dtype=np.float32)

    for t_idx in range(n_t):
        for p_idx in range(n_p):
            op_fname = _opacity_path(spec, temp_tokens[t_idx], P_LIST[p_idx])
            print("------", t_idx, p_idx, temps[t_idx], p_bar[p_idx], "======")
            print(spec.molname, op_fname)
            data = np.fromfile(op_fname, dtype=np.float32, count=-1)
            k_data[:] = 1.0e-99
            if spec.molname in ("Fe", "FeII"):
                k_data[:] = data[:]
            else:
                k_data[:] = data[:] * spec.molw / Avo
            k_data[:] = np.maximum(k_data, 1e-99)
            k_os[:] = np.interp(wn, wn_grid, np.log10(k_data), left=-99.0, right=-99.0)
            k_os_cube[t_idx, p_idx, :] = k_os

    out_fname = _species_output_name(spec, cfg.out_ext, cfg.res, len(cfg.species))
    base_name = out_fname.rsplit(".", 1)[0]
    zarr_name = base_name + ".zarr"
    zip_name = base_name + ".zarr.zip"

    comp = BloscCodec(cname="lz4", clevel=1, shuffle=BloscShuffle.shuffle)
    chunks_k = (1, 1, min(nb, 2048))

    arrays = dict(
        temperature=(np.asarray(temps, np.float64), (temps.shape[0],)),
        pressure=(np.asarray(p_bar, np.float64), (p_bar.shape[0],)),
        wavelength=(np.asarray(wl, np.float64), (min(wl.shape[0], 8192),)),
        cross_section=(np.asarray(k_os_cube, np.float32), chunks_k),
    )

    root = zarr.open_group(zarr_name, mode="w", zarr_format=3)
    root.attrs["molecule"] = str(spec.molname)
    for key, (data, chunks) in arrays.items():
        root.create_array(key, data=data, chunks=chunks, compressors=comp)
    print("Wrote", zarr_name)

    with ZipStore(zip_name, mode="w") as zstore:
        root_zip = zarr.open_group(store=zstore, mode="w", zarr_format=3)
        root_zip.attrs["molecule"] = str(spec.molname)
        for key, (data, chunks) in arrays.items():
            root_zip.create_array(key, data=data, chunks=chunks, compressors=comp)
    print("Wrote", zip_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_os.txt", help="Path to the OS input file.")
    args = parser.parse_args()

    cfg = load_input_config(args.input)
    wl, wn = build_wavelength_grid(cfg.wl_s, cfg.wl_e, cfg.res)
    write_wavelength_file(cfg.wl_ext, wl)

    for spec in cfg.species:
        generate_species_tables(cfg, spec, wl, wn)


if __name__ == "__main__":
    main()
