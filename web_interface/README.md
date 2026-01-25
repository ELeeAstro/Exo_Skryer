# Exo_Skryer Web Interface

A Streamlit-based web interface for generating retrieval configuration YAML files.

## Installation

```bash
cd web_interface
pip install -r requirements.txt
```

## Running the Interface

```bash
streamlit run app.py
```

This will open a browser window at `http://localhost:8501`.

## Features

- **Data Configuration**: Set paths to observation files, stellar spectra, and thermodynamic databases
- **Physics Configuration**: Configure radiative transfer scheme, T-P profiles, opacity sources, and cloud models
- **Opacity Configuration**: Add line species, Rayleigh scatterers, CIA pairs with custom paths
- **Parameters**: Add retrieval parameters with priors (uniform/delta), transforms, and bounds
- **Sampling**: Configure sampling engines (JAXNS, Dynesty, BlackJAX NS, NUTS, UltraNest, PyMultiNest)
- **Runtime**: Set compute platform (GPU/CPU) and resource allocation
- **Export**: Preview and download the generated YAML configuration file
