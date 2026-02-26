import numpy as np

files = {
    'F322W2': 'WASP80b_eclipse_F322W2_Eureka.txt',
    'F444W':  'WASP80b_eclipse_F444W_Eureka.txt',
    'LRS':    'WASP80b_eclipse_LRS_Eureka.txt',
}

rows = []
for label, fname in files.items():
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('wavelength'):
                continue
            vals = line.split()
            wavelength  = float(vals[0])
            bin_width   = float(vals[1])
            fp_value    = float(vals[2])
            fp_errorneg = float(vals[3])
            fp_errorpos = float(vals[4])
            rows.append((wavelength, bin_width, fp_value, fp_errorpos, fp_errorneg, label))

rows.sort(key=lambda r: r[0])

header = (
    "# WASP-80b emission data from Eureka pipeline\n"
    "# Format: wavelength(um)  delta_wavelength(um)  eclipse_depth  err_pos  err_neg  response_mode  offset_group\n"
    "# Eclipse depth and uncertainties in fraction\n"
    "# Data sorted by wavelength (lowest to highest)\n"
    "# F322W2: NIRCam 2.46-3.94 um, F444W: NIRCam 3.90-4.96 um, LRS: MIRI 5.25-11.75 um\n"
    "# ================================================================================"
)

with open('WASP-80b_JWST_em_1.txt', 'w') as out:
    out.write(header + '\n')
    for wavelength, bin_width, fp_value, err_pos, err_neg, label in rows:
        out.write(f"{wavelength:.7f} {bin_width:.9f} {fp_value:.11f} {err_pos:.11f} {err_neg:.11f} boxcar {label}\n")

print(f"Written {len(rows)} data points to WASP-80b_JWST_em_1.txt")
