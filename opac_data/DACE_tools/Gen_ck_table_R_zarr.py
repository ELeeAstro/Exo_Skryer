import numpy as np
from glob import glob
from numpy.polynomial.legendre import leggauss
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import ZipStore

Avo = 6.02214076e23


def rescale(x0, x1, gx, gw):
    gx = (x1 - x0) / 2.0 * gx + (x1 + x0) / 2.0
    gw = gw * (x1 - x0) / 2.0
    return gx, gw


# Read input files
fname = 'input_ck.txt'
f = open(fname, 'r')

for i in range(14):
    line = f.readline()
    if line[0] == '#':
        continue
    if i == 1:
        wl_s = float(line)
    elif i == 3:
        wl_e = float(line)
    elif i == 5:
        res = float(line)
    elif i == 7:
        wl_ext = str(line).strip()
    elif i == 9:
        out_ext = str(line).strip()
    elif i == 11:
        inform = int(line)
    elif i == 13:
        ls = line.split()
        molname = str(ls[0])
        molw = float(ls[1])
        swns = str(ls[4]).zfill(5)
        swne = str(ls[5]).zfill(5)
        basedir = str(ls[6])

print(wl_s, wl_e, res)
print(wl_ext)
print(out_ext)
print(inform)
print(molname, molw, swns, swne, basedir)


# Get pressures
nP = 34
P_list = ['n800', 'n766', 'n733', 'n700', 'n666', 'n633', 'n600', 'n566', 'n533', 'n500',
          'n466', 'n433', 'n400', 'n366', 'n333', 'n300', 'n266', 'n233', 'n200', 'n166', 'n133',
          'n100', 'n066', 'n033', 'p000', 'p033', 'p066', 'p100', 'p133', 'p166', 'p200', 'p233',
          'p266', 'p300']

P = []
for l in range(nP):
    line = P_list[l]
    line = line.replace('n', '-')
    line = line.replace('p', '+')
    line = line[0:2] + '.' + line[2:]
    P.append(float(line))
P = np.array(P)

print(P_list)
print(P)

# Get temperatures from directory itself
files = glob(basedir + '/*.bin')

lb = len(basedir)

T_list = []
for i in range(len(files)):
    files[i] = files[i][lb + 1:]
    T_list.append(files[i][16:21])

T_list = set(T_list)
T_list = sorted(T_list)

if molname == 'CO':
    nT = len(T_list) - 14
else:
    nT = len(T_list)

print(T_list)
print(nT)

T = np.zeros(nT)
for i in range(nT):
    T[i] = float(T_list[i])

# Construct Gauss-Legendre quadrature points and weights
ng = 16
g_split = True
g_split_point = 0.9

gx = np.zeros(ng)
gw = np.zeros(ng)
gx_scal = np.zeros(ng)
gw_scal = np.zeros(ng)

gx[:], gw[:] = leggauss(ng)

if not g_split:
    gx_scal[:], gw_scal[:] = rescale(0.0, 1.0, gx[:], gw[:])
else:
    ngh = int(ng / 2)
    gx[0:ngh], gw[0:ngh] = leggauss(ngh)
    gx_scal[0:ngh], gw_scal[0:ngh] = rescale(0.0, g_split_point, gx[0:ngh], gw[0:ngh])
    gx[ngh:], gw[ngh:] = leggauss(ngh)
    gx_scal[ngh:], gw_scal[ngh:] = rescale(g_split_point, 1.0, gx[ngh:], gw[ngh:])

# Generate wavelength grid
wn_l = 1.0 / (wl_e * 1e-4)
wn_h = 1.0 / (wl_s * 1e-4)

wno = wn_l
wnoarr = []

while wno < wn_h:
    dwno = wno / res
    wno = wno + dwno
    wnoarr = np.append(wnoarr, wno)

nb = len(wnoarr)

# Work internally with ascending wavenumber centers.
wn_centers = np.asarray(wnoarr, dtype=float)
wl_centers = 1.0 / wn_centers * 1e4

# Output grids in increasing wavelength order (Exo_Skryer convention).
wl = wl_centers[::-1].copy()
wn = wn_centers[::-1].copy()

# Write wavelength file
fname = wl_ext
f = open(fname, 'w')
f.write(str(nb) + '\n')
for l in range(nb):
    f.write(str(l + 1) + ' ' + str(wl[l]) + '\n')

if (molname == 'Na') or (molname == 'K'):
    wne = float(swne) + 0.01
else:
    wne = float(swne)

wns = float(swns)
dwn = 0.01
wn_grid = np.arange(wns, wne, dwn)

print(wnoarr)
print(wn_grid)

# Build wavenumber bin edges (length nb+1) for robust slicing on the master wn_grid.
wn_edges = np.zeros(nb + 1, dtype=float)
wn_edges[1:-1] = 0.5 * (wn_centers[:-1] + wn_centers[1:])
wn_edges[0] = wn_centers[0] - 0.5 * (wn_centers[1] - wn_centers[0])
wn_edges[-1] = wn_centers[-1] + 0.5 * (wn_centers[-1] - wn_centers[-2])

wn_edges[0] = max(wn_edges[0], wn_grid[0])
wn_edges[-1] = min(wn_edges[-1], wn_grid[-1] + dwn)

idx_edges = np.searchsorted(wn_grid, wn_edges, side="left")
idx_edges = np.clip(idx_edges, 0, wn_grid.size)

print("idx_edges:", idx_edges)

# Convert log-pressures to actual values (bar)
P[:] = 10.0 ** P[:]

# Compute k-coefficients
k_data = np.zeros(len(wn_grid))
k_coeff = np.zeros(ng)
k_ck_cube = np.zeros((nT, nP, nb, ng), dtype=np.float32)

print(nT, nP)

for t in range(nT):
    for p in range(nP):
        print('------', t, p, T[t], P[p], '======')
        if (molname == 'Fe') or (molname == 'FeII'):
            fname = basedir + '/' + T_list[t] + '_' + P_list[p] + '.bin'
        else:
            fname = basedir + '/' + 'Out_' + swns + '_' + swne + '_' + T_list[t] + '_' + P_list[p] + '.bin'
        print(molname, fname)
        data = np.fromfile(fname, dtype=np.float32, count=-1)
        k_data[:] = 1.0e-99
        if (molname == 'Fe') or (molname == 'FeII'):
            if len(data) > len(wn_grid):
                k_data[:] = data[:len(wn_grid)]
            else:
                k_data[:len(data)] = data[:]
        else:
            if len(data) > len(wn_grid):
                k_data[:] = data[:len(wn_grid)] * molw / Avo
            else:
                k_data[:len(data)] = data[:] * molw / Avo

        for b in range(nb):
            # Map output band index b -> source band index src in ascending wavenumber space.
            src = nb - 1 - b
            i0 = int(idx_edges[src])
            i1 = int(idx_edges[src + 1])
            if i1 <= i0:
                i0 = min(max(i0, 0), wn_grid.size - 1)
                i1 = i0 + 1

            k_band = np.maximum(k_data[i0:i1], 1e-99)
            k_sort = np.sort(k_band)
            x = np.linspace(0.0, 1.0, len(k_sort), endpoint=True)
            k_coeff[:] = np.interp(gx_scal, x, np.log10(k_sort), left=-99, right=-99)
            k_ck_cube[t, p, b, :] = k_coeff[:]

# Save to zarr (v3, Blosc/lz4 + byteshuffle)
base_name  = out_ext.rsplit('.', 1)[0]
zarr_name  = base_name + '.zarr'
zip_name   = base_name + '.zarr.zip'

comp     = BloscCodec(cname="lz4", clevel=1, shuffle=BloscShuffle.shuffle)
chunks_k = (1, 1, min(nb, 2048), ng)

arrays = dict(
    temperature   = (np.asarray(T,         np.float64), (T.shape[0],)),
    pressure      = (np.asarray(P,         np.float64), (P.shape[0],)),
    wavelength    = (np.asarray(wl,        np.float64), (min(wl.shape[0], 8192),)),
    g_points      = (np.asarray(gx_scal,   np.float64), (gx_scal.shape[0],)),
    g_weights     = (np.asarray(gw_scal,   np.float64), (gw_scal.shape[0],)),
    cross_section = (np.asarray(k_ck_cube, np.float32), chunks_k),
)

# Directory store — fast local reads
root = zarr.open_group(zarr_name, mode="w", zarr_format=3)
root.attrs["molecule"] = str(molname)
for key, (data, chunks) in arrays.items():
    root.create_array(key, data=data, chunks=chunks, compressors=comp)
print("Wrote", zarr_name)

# Zip store — single file for sharing/archiving (Google Drive etc.)
with ZipStore(zip_name, mode="w") as zstore:
    root_zip = zarr.open_group(store=zstore, mode="w", zarr_format=3)
    root_zip.attrs["molecule"] = str(molname)
    for key, (data, chunks) in arrays.items():
        root_zip.create_array(key, data=data, chunks=chunks, compressors=comp)
print("Wrote", zip_name)

