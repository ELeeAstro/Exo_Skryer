"""
data_constants.py
=================
"""

R = 8.31446261815324e7
kb = 1.380649e-16
amu = 1.66053906892e-24
G = 6.67430e-8
h = 6.62607015e-27
c_light = 2.99792458e10

R_jup = 7.1492e9
R_sun  = 6.957e10
R_earth = 6.3781e8

M_sun = 1.988416e33
M_jup = 1.89813e30
M_earth = 5.9722e27


bar = 1.0e6
pa = 1.0e5



# Species data adapted from the BeAR code (Kitzmann 2024) for quick global access.
CHEM_SPECIES_DATA = [
    {"id": "_H", "symbol": "H", "fastchem_symbol": "H", "molecular_weight": 1.00784},
    {"id": "_He", "symbol": "He", "fastchem_symbol": "He", "molecular_weight": 4.002602},
    {"id": "_C", "symbol": "C", "fastchem_symbol": "C", "molecular_weight": 12.0107},
    {"id": "_O", "symbol": "O", "fastchem_symbol": "O", "molecular_weight": 15.999},
    {"id": "_Fe", "symbol": "Fe", "fastchem_symbol": "Fe", "molecular_weight": 55.845},
    {"id": "_Fep", "symbol": "Fe+", "fastchem_symbol": "Fe+", "molecular_weight": 55.845},
    {"id": "_Ca", "symbol": "Ca", "fastchem_symbol": "Ca", "molecular_weight": 40.078},
    {"id": "_Ti", "symbol": "Ti", "fastchem_symbol": "Ti", "molecular_weight": 47.867},
    {"id": "_Tip", "symbol": "Ti+", "fastchem_symbol": "Ti+", "molecular_weight": 47.867},
    {"id": "_H2", "symbol": "H2", "fastchem_symbol": "H2", "molecular_weight": 2.01588},
    {"id": "_H2O", "symbol": "H2O", "fastchem_symbol": "H2O1", "molecular_weight": 18.01528},
    {"id": "_CO2", "symbol": "CO2", "fastchem_symbol": "C1O2", "molecular_weight": 44.01},
    {"id": "_CO", "symbol": "CO", "fastchem_symbol": "C1O1", "molecular_weight": 28.0101},
    {"id": "_CH4", "symbol": "CH4", "fastchem_symbol": "C1H4", "molecular_weight": 16.04246},
    {"id": "_HCN", "symbol": "HCN", "fastchem_symbol": "C1H1N1_1", "molecular_weight": 27.0253},
    {"id": "_NH3", "symbol": "NH3", "fastchem_symbol": "H3N1", "molecular_weight": 17.03052},
    {"id": "_C2H2", "symbol": "C2H2", "fastchem_symbol": "C2H2", "molecular_weight": 26.04},
    {"id": "_N2", "symbol": "N2", "fastchem_symbol": "N2", "molecular_weight": 28.0134},
    {"id": "_Na", "symbol": "Na", "fastchem_symbol": "Na", "molecular_weight": 22.98977},
    {"id": "_K", "symbol": "K", "fastchem_symbol": "K", "molecular_weight": 39.0983},
    {"id": "_H2S", "symbol": "H2S", "fastchem_symbol": "H2S1", "molecular_weight": 34.09099},
    {"id": "_Hm", "symbol": "H-", "fastchem_symbol": "H1-", "molecular_weight": 1.00784},
    {"id": "_TiO", "symbol": "TiO", "fastchem_symbol": "O1Ti1", "molecular_weight": 63.8664},
    {"id": "_VO", "symbol": "VO", "fastchem_symbol": "O1V1", "molecular_weight": 66.9409},
    {"id": "_FeH", "symbol": "FeH", "fastchem_symbol": "H1Fe1", "molecular_weight": 56.853},
    {"id": "_SH", "symbol": "SH", "fastchem_symbol": "H1S1", "molecular_weight": 34.08},
    {"id": "_MgO", "symbol": "MgO", "fastchem_symbol": "Mg1O1", "molecular_weight": 40.3044},
    {"id": "_AlO", "symbol": "AlO", "fastchem_symbol": "Al1O1", "molecular_weight": 42.981},
    {"id": "_CaO", "symbol": "CaO", "fastchem_symbol": "Ca1O1", "molecular_weight": 56.0774},
    {"id": "_CrH", "symbol": "CrH", "fastchem_symbol": "Cr1H1", "molecular_weight": 54.004},
    {"id": "_MgH", "symbol": "MgH", "fastchem_symbol": "H1Mg1", "molecular_weight": 26.3209},
    {"id": "_CaH", "symbol": "CaH", "fastchem_symbol": "Ca1H1", "molecular_weight": 41.0859},
    {"id": "_TiH", "symbol": "TiH", "fastchem_symbol": "H1Ti1", "molecular_weight": 48.87484},
    {"id": "_OH", "symbol": "OH", "fastchem_symbol": "H1O1", "molecular_weight": 17.008},
    {"id": "_e", "symbol": "e-", "fastchem_symbol": "e-", "molecular_weight": 5.4857990907e-4},
    {"id": "_V", "symbol": "V", "fastchem_symbol": "V", "molecular_weight": 50.9415},
    {"id": "_Vp", "symbol": "V+", "fastchem_symbol": "V1+", "molecular_weight": 50.9415},
    {"id": "_Mn", "symbol": "Mn", "fastchem_symbol": "Mn", "molecular_weight": 54.938044},
    {"id": "_Si", "symbol": "Si", "fastchem_symbol": "Si", "molecular_weight": 28.085},
    {"id": "_Cr", "symbol": "Cr", "fastchem_symbol": "Cr", "molecular_weight": 51.996},
    {"id": "_Crp", "symbol": "Cr+", "fastchem_symbol": "Cr1+", "molecular_weight": 51.996},
    {"id": "_SiO", "symbol": "SiO", "fastchem_symbol": "O1Si1", "molecular_weight": 44.08},
    {"id": "_SiO2", "symbol": "SiO2", "fastchem_symbol": "O2Si1", "molecular_weight": 60.08},
    {"id": "_SO2", "symbol": "SO2", "fastchem_symbol": "O2S1", "molecular_weight": 64.066},
    {"id": "_CS2", "symbol": "CS2", "fastchem_symbol": "C1S2", "molecular_weight": 76.139},
]
