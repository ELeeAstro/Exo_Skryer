import pyfastchem
from save_output import saveChemistryOutput, saveMonitorOutput, saveChemistryOutputPandas, saveMonitorOutputPandas
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy import constants as const


def round_in_log(x):
    log_floor = np.floor(np.log10(x))
    pre_exponent = np.round(x * 10 ** -log_floor, 1)
    return pre_exponent * 10 ** log_floor

# Load T and p grid from opacity data to ensure exact matching
opacity_file = '../opac_data/lbl/H2O_R10000.npz'
opac_data = np.load(opacity_file)
T = opac_data['temperature']  # Temperature in K
p = opac_data['pressure']      # Pressure in bar

T = [500.0,1000.0]
T = np.array(T)
p = [1e-3,1e-2]
p = np.array(p)

# T = np.full(1000, 1000)
# p = np.logspace(-8, 3, num=1000)

# Setup M/H and C/O grid
log_M_H = np.linspace(-1,3,11)
log_C_O = np.linspace(-1,3,16)

print(len(T), T)
print(len(p), p)
print(len(log_M_H),10.**log_M_H)
print(len(log_C_O),10.0**log_C_O)

# Fastchem output dir
output_dir = './'

plot_species = ['H2O1', 'C1O2', 'C1O1', 'C1H4', 'H3N1']
plot_species_labels = ['H2O', 'CO2', 'CO', 'CH4', 'NH3']

fastchem = pyfastchem.FastChem(
 '/Users/gl20y334/FastChem/input/element_abundances/asplund_2020.dat',
 '/Users/gl20y334/FastChem/input/logK/logK.dat',
 1)

input_data = pyfastchem.FastChemInput()
output_data = pyfastchem.FastChemOutput()

input_data.temperature = T
input_data.pressure = p

fastchem_flag = fastchem.calcDensities(input_data, output_data)

print("FastChem reports:")
print("  -", pyfastchem.FASTCHEM_MSG[fastchem_flag])

if np.amin(output_data.element_conserved[:]) == 1:
  print("  - element conservation: ok")
else:
  print("  - element conservation: fail")

os.makedirs(output_dir, exist_ok=True)


#save the monitor output to a file
saveMonitorOutput(output_dir + '/monitor.dat',
                  T, p,
                  output_data.element_conserved,
                  output_data.fastchem_flag,
                  output_data.nb_iterations,
                  output_data.nb_chemistry_iterations,
                  output_data.nb_cond_iterations,
                  output_data.total_element_density,
                  output_data.mean_molecular_weight,
                  fastchem)

#this would save the output of all species
saveChemistryOutput(output_dir + '/chemistry.dat',
                    T, p,
                    output_data.total_element_density,
                    output_data.mean_molecular_weight,
                    output_data.number_densities,
                    fastchem)

#this saves only selected species (here the species we also plot)
saveChemistryOutput(output_dir + '/chemistry_select.dat',
                    T, p,
                    output_data.total_element_density,
                    output_data.mean_molecular_weight,
                    output_data.number_densities,
                    fastchem,
                    plot_species)

plot_species_indices = []
plot_species_symbols = []

for i, species in enumerate(plot_species):
  index = fastchem.getGasSpeciesIndex(species)

  if index != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
    plot_species_indices.append(index)
    plot_species_symbols.append(plot_species_labels[i])
  else:
    print("Species", species, "to plot not found in FastChem")


#convert the output into a numpy array
number_densities = np.array(output_data.number_densities)


#total gas particle number density from the ideal gas law
#used later to convert the number densities to mixing ratios
gas_number_density = p*1e6 / (const.k_B.cgs * T)


#and plot...
for i in range(0, len(plot_species_symbols)):
  fig = plt.plot(number_densities[:, plot_species_indices[i]]/gas_number_density, p)

plt.xscale('log')
plt.yscale('log')
plt.gca().set_ylim(plt.gca().get_ylim()[::-1])

plt.xlabel("Mixing ratios")
plt.ylabel("Pressure (bar)")
plt.legend(plot_species_symbols)

plt.show()