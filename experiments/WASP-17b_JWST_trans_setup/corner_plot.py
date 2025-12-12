import arviz as az
import corner
from pathlib import Path
import matplotlib.pylab as plt

exp_dir = Path(".")   # folder containing posterior.nc
idata = az.from_netcdf(exp_dir / "posterior.nc")

# Choose which vars to plot (must match YAML names present in posterior)
#var_names = ["temperature", "k_cld_grey", "A_slope"]
#var_names = ['R_p','T_iso','cld_k0','cld_r','cld_Q0','cld_a']


#var_names = ["R_p","T_iso", "log_10_f_H2O"]
#scales = ['linear','linear','linear']

var_names = ["R_p","T_iso", "log_10_f_H2O", "log_10_f_Na", "log_10_f_K", 'f_cloud', 'log_10_cld_r']
scales = ['linear','linear','linear','linear','linear', 'linear', 'linear']

quants = [0.16, 0.5, 0.84]

fig = corner.corner(
    idata,
    var_names=var_names,   # which variables from idata.posterior to use
    divergences=False,     # you don't have divergence info from BlackJAX (yet)
    axes_scale=scales,
    show_titles=True,
    quantiles=quants
)

plt.show()
