import numpy as np
from glob import glob
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import ZipStore

Avo = 6.02214076e23

# Read input files

fname = 'input_os.txt'

f = open(fname,'r')

for i in range(14):
  line = f.readline()
  if (line[0] == '#'):
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

print(wl_s,wl_e,res)
print(wl_ext)
print(out_ext)
print(inform)
print(molname, molw, swns, swne, basedir)


# Get pressures
nP = 34
P_list = ['n800', 'n766', 'n733', 'n700', 'n666', 'n633', 'n600', 'n566', 'n533', 'n500', \
  'n466', 'n433', 'n400', 'n366', 'n333', 'n300', 'n266', 'n233', 'n200','n166', 'n133', \
  'n100', 'n066', 'n033', 'p000', 'p033', 'p066', 'p100', 'p133', 'p166', 'p200', 'p233', \
  'p266', 'p300']

P = []
for l in range(nP):
  line = P_list[l]
  line = line.replace('n','-')
  line = line.replace('p','+')
  line = line[0:2] + '.' + line[2:]
  P.append(float(line))
P = np.array(P)

print(P_list)
print(P)

# Get temperatures from directory itself
files = glob(basedir+'/*.bin')

lb = len(basedir)

T_list = []
for i in range(len(files)):
  files[i] = files[i][lb+1:]
  T_list.append(files[i][16:21])
  #T_list.append(files[i][0:5])


T_list = set(T_list)
T_list = sorted(T_list)

if (molname == 'CO'):
  nT = len(T_list) -14
else:
  nT = len(T_list)

print(T_list)
print(nT)

T = np.zeros(nT)
for i in range(nT):
  T[i] = float(T_list[i])


# Generate wavelength grid
wn_l = 1.0/(wl_e * 1e-4)
wn_h = 1.0/(wl_s * 1e-4)

wno = wn_l
counter = 0
wnoarr = []

while (wno < wn_h):
    dwno = wno/res
    wno = wno + dwno
    wnoarr=np.append(wnoarr,wno)
    #print(counter, 1.0/wno*1e4, 'micron')
    #print(wnoarr[counter])
    counter += 1

nb = len(wnoarr)

wl = np.zeros(nb)
wl[:] = 1.0/wnoarr[::-1] * 1e4
wn = np.zeros(nb)
wn[:] = wnoarr[::-1]

fname = wl_ext
f = open(fname,'w')
f.write(str(nb) + '\n')
a = 1
for l in range(nb):
    #print(l)
    f.write(str(a) + ' ' + str(wl[l]) + '\n')
    a = a + 1
if ((molname == 'Na') or (molname == 'K')):
  wne = float(swne) + 0.01
else:
  wne = float(swne)

wns = float(swns)
dwn = 0.01
nwn = (wne - wns)/dwn

wn_grid = np.arange(wns,wne,dwn)

print(wnoarr)
print(wn_grid)

# Make output file
fname = out_ext
out_fname = out_ext
f = open(fname,'w')

f.write(molname + '\n')
f.write(str(nT) + ' ' +  str(nP) + ' ' +  str(nb) + ' ' + str(1) + '\n')
f.write(" ".join(str(g) for g in T[:]) + '\n')
P[:] = 10.0**P[:]
f.write(" ".join(str(g) for g in P[:]) + '\n')

f.write(' ' + '\n')

f.write(" ".join(str(g) for g in wl[:]) + '\n')
f.write(" ".join(str(g) for g in wn[:]) + '\n')

f.write(' ' + '\n')


k_data = np.zeros(len(wn_grid))
k_os = np.zeros(nb)
k_os_cube = np.zeros((nT, nP, nb), dtype=np.float32)

print(nT, nP)

for t in range(nT):
  for p in range(nP):
    print('------', t,p, T[t], P[p], '======')
    if ((molname == 'Fe') or (molname == 'FeII')):
      op_fname = basedir+'/'+T_list[t]+'_'+P_list[p]+'.bin'
    else:
      op_fname = basedir+'/'+'Out_'+swns+'_'+swne+'_'+T_list[t]+'_'+P_list[p]+'.bin'
    print(molname, op_fname)
    data = np.fromfile(op_fname,dtype=np.float32, count=-1)
    k_data[:] = 1.0e-99
    if ((molname == 'Fe') or (molname == 'FeII')):
      k_data[:] = data[:]
    else:
      k_data[:] = data[:] * molw / Avo
    k_data[:] = np.maximum(k_data[:],1e-99)
    # Interpolate opacity data to grid
    k_os[:] = np.interp(wn,wn_grid,np.log10(k_data),left=-99.0,right=-99.0)
    k_os_cube[t, p, :] = k_os[:]

# Save to zarr (v3, Blosc/lz4 + byteshuffle)
base_name = out_fname.rsplit('.', 1)[0]
zarr_name = base_name + '.zarr'
zip_name  = base_name + '.zarr.zip'

comp = BloscCodec(cname="lz4", clevel=1, shuffle=BloscShuffle.shuffle)
chunks_k = (1, 1, min(nb, 2048))

arrays = dict(
    temperature   = (np.asarray(T,          np.float64), (T.shape[0],)),
    pressure      = (np.asarray(P,          np.float64), (P.shape[0],)),
    wavelength    = (np.asarray(wl,         np.float64), (min(wl.shape[0], 8192),)),
    cross_section = (np.asarray(k_os_cube, np.float32), chunks_k),
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
