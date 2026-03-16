"""
Exo_Skryer Retrieval Configuration Generator
=============================================
A Streamlit web interface for generating YAML configuration files for
atmospheric retrieval runs.

HOW STREAMLIT WORKS:
--------------------
Streamlit is a Python framework that turns scripts into interactive web apps.
Key concepts:
1. The script runs top-to-bottom every time the user interacts with a widget
2. st.session_state persists data between reruns (like a global dictionary)
3. Widgets (buttons, inputs, etc.) automatically create interactive UI elements
4. st.rerun() forces the script to re-execute (useful after modifying state)

STRUCTURE OF THIS FILE:
-----------------------
1. CONSTANTS - All dropdown options and predefined choices
2. HELPER FUNCTIONS - State management and config building
3. UI COMPONENTS - Functions that render each section of the form
4. MAIN - Entry point that orchestrates everything
"""

import streamlit as st  # The main Streamlit library for building web UIs
import yaml             # For converting Python dicts to YAML format

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# This MUST be the first Streamlit command in the script.
# It sets browser tab title, favicon, and layout mode.
st.set_page_config(
    page_title="Exo_Skryer Config Generator",  # Browser tab title
    page_icon="🔭",                             # Favicon (can be emoji or path)
    layout="wide"                               # Use full browser width
)

# =============================================================================
# CUSTOM YAML HANDLING
# =============================================================================

class FlowStyleDict(dict):
    """Marker class for dicts that should be rendered in YAML flow style (single line)."""
    pass


class CustomDumper(yaml.SafeDumper):
    """Custom YAML dumper that renders FlowStyleDict in flow style."""
    pass


def represent_flow_dict(dumper, data):
    """Represent FlowStyleDict in flow style (e.g., {name: R_s, dist: delta, ...})."""
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items(), flow_style=True)


def represent_none(dumper, _):
    """Tell PyYAML how to represent Python None values in YAML output."""
    return dumper.represent_scalar('tag:yaml.org,2002:null', 'null')


# Register custom representers
CustomDumper.add_representer(FlowStyleDict, represent_flow_dict)
CustomDumper.add_representer(type(None), represent_none)


# =============================================================================
# CONSTANTS AND OPTIONS
# =============================================================================
# These lists define all the valid choices for dropdown menus throughout the app.
# Canonical names and aliases are defined in kernel_registry.py.

# Radiative transfer schemes: emission (thermal) vs transit (transmission)
# See: build_model.py _build_rt_kernel()
RT_SCHEMES = ["emission_1d", "transit_1d"]

# Emission calculation schemes (only used when rt_scheme is emission_1d)
# Passed to get_emission_solver() in RT_em_schemes.py
# - eaa / alpha_eaa: Alpha-EAA single-angle approximation (fast)
# - toon89 / toon89_picaso: Toon et al. (1989) multi-stream method (accurate for scattering)
EM_SCHEMES = ["eaa", "alpha_eaa", "toon89", "toon89_picaso"]

# Temperature-Pressure profile parameterizations
# See: kernel_registry.VERT_TP
# - isothermal: Constant temperature throughout atmosphere
# - Guillot: Analytic profile with T_eq, T_int, k_ir, gam_v parameters
# - Barstow: Profile with stratospheric temperature (T_strat)
# - Line: Line profile
# - Milne: Eddington-Milne approximation
# - Modified_Milne: Modified Milne profile
# - picket_fence: Picket-fence approximation
# - MandS: Madhusudhan & Seager profile
VERT_TP_OPTIONS = [
    "isothermal",
    "guillot",
    "modified_guillot",
    "barstow",
    "line",
    "milne",
    "modified_milne",
    "picket_fence",
    "mands",
]

# How altitude/height is calculated in the atmosphere model
# See: kernel_registry.VERT_ALT
# - hypsometric: Constant gravity (aliases: constant, constant_g, fixed)
# - variable_g: Gravity varies with altitude (more physical)
# - p_ref: Variable g with reference pressure level defining the radius
VERT_ALT_OPTIONS = ["p_ref", "variable_g", "hypsometric"]

# Chemistry profile options
# See: kernel_registry.VERT_CHEM
# - constant_vmr: Volume mixing ratios constant with altitude
# - fastchem_grid_jax: Chemical equilibrium via precomputed FastChem grid
# - rate_ce: Chemical equilibrium via rate equations (JAX)
# - easychem_jax: EasyChem-style gas-phase equilibrium backend
# - atmodeller: Atmodeller equilibrium backend
# - quench_approx: Quenched chemistry approximation
VERT_CHEM_OPTIONS = [
    "constant_vmr",
    "constant_vmr_clr",
    "ce",
    "fastchem_grid_jax",
    "rate_ce",
    "easychem_jax",
    "quench_approx",
    "atmodeller",
]

# Mean molecular weight handling
# See: kernel_registry.VERT_MU
# - dynamic: Calculated from VMR composition at each layer
# - constant: Fixed value (requires 'mu' parameter)
# - auto: Uses 'mu' param if present, else computes from VMR
VERT_MU_OPTIONS = ["dynamic", "constant", "auto"]

# Cloud vertical distribution profiles
# See: kernel_registry.VERT_CLOUD
# - None: No clouds
# - exponential_decay_profile: Exponential decay from cloud base
# - slab_profile: Cloud slab between two pressure levels
# - const_profile: Constant cloud throughout atmosphere
VERT_CLOUD_OPTIONS = ["None", "exponential", "slab", "constant"]

# Line opacity calculation methods
# See: build_model.py _select_kernels() (opac_line block)
# - ck: Correlated-k (fast, binned opacities)
# - os: Opacity sampling (high-resolution cross-sections)
# - None: No line opacity
OPAC_LINE_OPTIONS = ["ck", "os", "None"]

# Rayleigh scattering opacity
# See: build_model.py _resolve_os_ck_opac()
# Both 'os' and 'ck' use the same compute_ray_opacity kernel
OPAC_RAY_OPTIONS = ["ck", "os", "None"]

# Collision-Induced Absorption (CIA) opacity
# See: build_model.py _resolve_os_ck_opac()
# Both 'os' and 'ck' use the same compute_cia_opacity kernel
OPAC_CIA_OPTIONS = ["ck", "os", "None"]

# Cloud opacity models
# See: kernel_registry.OPAC_CLOUD
# - None: No cloud opacity
# - grey: Grey (wavelength-independent) cloud opacity
# - deck_and_powerlaw: Cloud deck with power-law wavelength dependence
# - F18: Fresnel 2018 parameterization
# - direct_nk: Use refractive index (n,k) data directly with Mie theory
# - madt_rayleigh: MADT Rayleigh approximation for Mie scattering
# - lxmie: Full Mie calculation (LX-MIE)
OPAC_CLOUD_OPTIONS = ["None", "grey", "powerlaw", "f18", "direct_nk", "madt_rayleigh", "lxmie"]

# Special opacity sources (like H- bound-free, free-free)
# See: build_model.py _select_kernels() (opac_special block)
OPAC_SPECIAL_OPTIONS = ["ck", "os", "on", "None"]

# Cloud particle size distributions
# See: build_model.py _extract_fixed_params() (cloud_dist block)
# - mono (or monodisperse): Single particle size
# - lognormal: Log-normal size distribution (requires cld_sigma parameter)
CLOUD_DIST_OPTIONS = ["mono", "log_normal", "lognormal"]

# Emission mode: how the planet's emission is calculated
# See: build_model.py build_forward_model() (emission_mode block)
# - planet: Standard planet emission (uses stellar flux for contrast)
# - brown_dwarf: Brown dwarf mode (no stellar flux needed)
EMISSION_MODES = ["planet", "brown_dwarf"]

# Correlated-k mixing rules for combining opacities from multiple species
# See: build_model.py _select_kernels() (ck_mix block)
# - RORR: Random Overlap with Resorting and Rebinning (default, most accurate)
# - TRANS: Transmission-optimized method (only for transit_1d)
CK_MIX_OPTIONS = ["RORR", "PRAS", "TRANS"]

# Bayesian sampling engines supported by Exo_Skryer
# Each has different strengths for exploring parameter space
SAMPLING_ENGINES = ["jaxns", "dynesty", "blackjax_ns", "nuts", "ultranest", "pymultinest"]

# Compute platforms
RUNTIME_PLATFORMS = ["gpu", "cpu"]

# Refraction modes (for transit_1d only)
# See: build_model.py _resolve_refraction()
# - None: No refraction
# - cutoff / refractive_cutoff: Apply refractive cutoff without ray tracing
REFRACTION_OPTIONS = ["None", "cutoff", "refractive_cutoff", "refraction_cutoff"]

# Common molecular/atomic species for line opacity (from opac_data directory)
# These are the absorbers that create spectral features
COMMON_LINE_SPECIES = ["H2O", "CO", "CO2", "CH4", "NH3", "H2S", "HCN", "C2H2", "SO2", "OH", "Na", "K"]

# Species that contribute to Rayleigh scattering (from registry_ray.py)
# Supported: H2, He, CO, CO2, CH4, O2, N2, NH3, Ar, N2O, SF6, HCl, HCN, H2S, OCS, SO2, C2H2, PH3, SO3, H2O, e-, H
COMMON_RAY_SPECIES = ["H2", "He", "N2", "CO", "CO2", "CH4", "O2", "NH3", "Ar", "H2O", "H2S", "HCN", "SO2", "C2H2"]

# Collision-induced absorption pairs (from opac_data/cia directory)
# Note: H- is handled as a special opacity source (bound-free/free-free), not a CIA pair.
COMMON_CIA_PAIRS = ["H2-H2", "H2-He", "H2-H", "He-H"]

# Special opacity sources (currently only H- continuum)
COMMON_SPECIAL_SPECIES = ["H-"]

# Prior distribution types for retrieval parameters
# - uniform: Flat prior between low and high bounds
# - delta: Fixed value (not retrieved, held constant)
DISTRIBUTION_TYPES = ["uniform", "delta"]

# Parameter space transforms for sampling efficiency
# - identity: No transform (sample in natural space)
# - logit: Maps bounded [low, high] to unbounded space (better for MCMC)
TRANSFORM_TYPES = ["identity", "logit"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def init_session_state():
    """
    Initialize Streamlit's session_state with default values.

    SESSION STATE EXPLANATION:
    --------------------------
    Streamlit reruns the entire script on every user interaction. Without
    session_state, all variables would reset. session_state is a persistent
    dictionary that survives between reruns.

    We only set defaults if the key doesn't already exist (preserving user changes).
    """
    defaults = {
        # Physics defaults
        'rt_scheme': 'emission_1d',
        'vert_tp': 'guillot',
        'vert_cloud': 'None',
        'opac_cloud': 'None',
        # Species lists (these are mutable lists that grow as user adds species)
        'line_species': [],
        'ray_species': ['H2', 'He'],      # Common defaults for gas giants
        'cia_species': ['H2-H2', 'H2-He'], # Common CIA pairs
        # Special opacity toggles
        'special_hminus_enabled': False,
        'special_hminus_bf': True,
        'special_hminus_ff': True,
        # Parameters list (will hold retrieval parameter definitions)
        'params': [],
        # Sampling
        'sampling_engine': 'dynesty',
        # Chemistry backend defaults
        'easychem_mode': 'scan',
        'easychem_max_steps': 64,
        'easychem_tol': 1.0e-11,
        'easychem_throw': True,
        'easychem_relax_limit': 0.75,
        'easychem_p0_bar': 1.0,
        'easychem_e_ref': 'H',
        'easychem_prefer_chord': False,
        'fastchem_grid_solver_mode': 'vmap',
        'fastchem_grid_bounds_mode': 'clip',
        'atmodeller_solver_multistart': 10,
        'atmodeller_solver_atol': 1.0e-6,
        'atmodeller_solver_rtol': 1.0e-6,
        'atmodeller_solver_max_steps': 256,
        'atmodeller_solver_jac': 'fwd',
    }
    # Only initialize keys that don't exist yet
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_parameter(name: str, dist: str, value: float = None, low: float = None,
                  high: float = None, transform: str = "identity", init: float = None):
    """
    Add a retrieval parameter to the session state's params list.

    Parameters define what gets retrieved (fitted) vs held fixed:
    - delta distribution: Parameter is FIXED at 'value' (not retrieved)
    - uniform distribution: Parameter is RETRIEVED between 'low' and 'high'

    Args:
        name: Parameter name (e.g., "log_10_f_H2O" for water abundance)
        dist: "delta" (fixed) or "uniform" (retrieved)
        value: Fixed value (only used if dist="delta")
        low: Lower bound (only used if dist="uniform")
        high: Upper bound (only used if dist="uniform")
        transform: "identity" or "logit" - how to transform for sampling
        init: Optional initial guess for the sampler
    """
    param = {"name": name, "dist": dist, "transform": transform}
    if dist == "delta":
        # Fixed parameter - just needs a value
        param["value"] = value
    else:
        # Retrieved parameter - needs bounds
        param["low"] = low
        param["high"] = high
        if init is not None:
            param["init"] = init
    # Append to the persistent list in session_state
    st.session_state.params.append(param)


def remove_parameter(index: int):
    """
    Remove a parameter from the params list by its index.

    Args:
        index: Position in the list (0-based)
    """
    if 0 <= index < len(st.session_state.params):
        st.session_state.params.pop(index)


def _parse_text_list(raw: str) -> list[str]:
    if not raw:
        return []
    parts = []
    for line in raw.replace(",", "\n").splitlines():
        item = line.strip()
        if item:
            parts.append(item)
    return parts


def _parse_mapping_lines(raw: str) -> dict[str, str]:
    mappings: dict[str, str] = {}
    if not raw:
        return mappings
    for line in raw.splitlines():
        item = line.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            mappings[key] = value
    return mappings


def _none_if_disabled(value):
    if value in (None, "None", "none", "off", "false", "0", ""):
        return None
    return value


def build_config() -> dict:
    """
    Build the complete configuration dictionary from current session state.

    This function reads all the values the user has entered via the UI
    (stored in st.session_state) and assembles them into a nested dictionary
    that matches the structure of retrieval_config.yaml files.

    Returns:
        dict: Complete configuration ready to be converted to YAML
    """
    config = {}

    # -------------------------------------------------------------------------
    # DATA SECTION
    # Paths to input files (observation data, stellar spectrum, thermo databases)
    # -------------------------------------------------------------------------
    config['data'] = {}
    # Only include paths that the user has actually filled in
    if st.session_state.get('obs_path'):
        config['data']['obs'] = st.session_state.obs_path
    if st.session_state.get('stellar_path'):
        config['data']['stellar'] = st.session_state.stellar_path
    if st.session_state.get('nasa9_path'):
        config['data']['nasa9'] = st.session_state.nasa9_path

    # -------------------------------------------------------------------------
    # PHYSICS SECTION
    # Controls atmospheric structure and radiative transfer settings
    # -------------------------------------------------------------------------
    config['physics'] = {
        'nlay': st.session_state.get('nlay', 99),  # Number of atmospheric layers
        'vert_Tp': st.session_state.get('vert_tp', 'guillot'),
        'vert_alt': st.session_state.get('vert_alt', 'p_ref'),
        'vert_chem': st.session_state.get('vert_chem', 'constant_vmr'),
        'vert_mu': st.session_state.get('vert_mu', 'dynamic'),
        # Cloud profile (convert string "None" to actual None for YAML)
        'vert_cloud': _none_if_disabled(st.session_state.get('vert_cloud', 'None')),
        'opac_line': _none_if_disabled(st.session_state.get('opac_line', 'ck')),
        # These opacity sources can be disabled (None)
        'opac_ray': _none_if_disabled(st.session_state.get('opac_ray', 'os')),
        'opac_cia': _none_if_disabled(st.session_state.get('opac_cia', 'os')),
        'opac_cloud': _none_if_disabled(st.session_state.get('opac_cloud', 'None')),
        'opac_special': _none_if_disabled(st.session_state.get('opac_special', 'None')),
        'cloud_dist': st.session_state.get('cloud_dist', 'mono'),
        'rt_scheme': st.session_state.get('rt_scheme', 'emission_1d'),
        # Emission-specific settings (only include if doing emission spectrum)
        'em_scheme': st.session_state.get('em_scheme', 'eaa') if st.session_state.get('rt_scheme') == 'emission_1d' else None,
        'emission_mode': st.session_state.get('emission_mode', 'planet') if st.session_state.get('rt_scheme') == 'emission_1d' else None,
        'contri_func': st.session_state.get('contri_func', False),
        # Refraction (only for transit mode)
        'refraction': _none_if_disabled(st.session_state.get('refraction', 'None')) if st.session_state.get('rt_scheme') == 'transit_1d' else None,
    }
    # Remove None values from the dict (cleaner YAML output)
    config['physics'] = {k: v for k, v in config['physics'].items() if v is not None}

    # -------------------------------------------------------------------------
    # OPAC SECTION
    # Opacity data files and wavelength grid settings
    # -------------------------------------------------------------------------
    config['opac'] = {
        'wl_master': st.session_state.get('wl_master', ''),  # Master wavelength grid file
        'full_grid': st.session_state.get('full_grid', True),  # Use full wavelength range
        'ck': st.session_state.get('use_ck', False),  # Use correlated-k tables
    }
    # CK mixing rule only relevant if using correlated-k
    if st.session_state.get('use_ck'):
        config['opac']['ck_mix'] = st.session_state.get('ck_mix', 'RORR')

    # Line opacity species (the main absorbers like H2O, CO, etc.)
    # Use FlowStyleDict for compact single-line YAML output
    if st.session_state.line_species:
        config['opac']['line'] = []
        for species in st.session_state.line_species:
            entry = {'species': species['name']}
            if species.get('path'):  # Path is optional (can use defaults)
                entry['path'] = species['path']
            config['opac']['line'].append(FlowStyleDict(entry))

    # Rayleigh scattering species (simpler format - no paths needed)
    if st.session_state.ray_species:
        config['opac']['ray'] = [FlowStyleDict({'species': s}) for s in st.session_state.ray_species]

    # Collision-induced absorption pairs
    if st.session_state.cia_species:
        config['opac']['cia'] = []
        for pair in st.session_state.cia_species:
            entry = {'species': pair}
            # Check if user specified a custom path for this CIA pair
            cia_path = st.session_state.get(f'cia_path_{pair}')
            if cia_path:
                entry['path'] = cia_path
            config['opac']['cia'].append(FlowStyleDict(entry))

    # Special opacity sources (e.g., H- bf/ff)
    if st.session_state.get("special_hminus_enabled", False):
        bf_on = bool(st.session_state.get("special_hminus_bf", True))
        ff_on = bool(st.session_state.get("special_hminus_ff", True))
        config['opac']['special'] = [FlowStyleDict({'species': 'H-', 'bf': bf_on, 'ff': ff_on})]

    # Cloud opacity data path
    if st.session_state.get('cloud_opac_path'):
        config['opac']['cloud'] = [FlowStyleDict({'path': st.session_state.cloud_opac_path})]

    # -------------------------------------------------------------------------
    # PARAMS SECTION
    # List of retrieval parameters with their priors
    # Wrap each param dict in FlowStyleDict for single-line YAML output
    # -------------------------------------------------------------------------
    if st.session_state.params:
        # Wrap each param in FlowStyleDict for compact YAML output
        config['params'] = [FlowStyleDict(p) for p in st.session_state.params]

    vert_chem = st.session_state.get('vert_chem', 'constant_vmr')
    if vert_chem == 'easychem_jax':
        easychem_species = _parse_text_list(st.session_state.get('easychem_species', ''))
        easychem_block = {
            'species': easychem_species,
            'solver': {
                'mode': st.session_state.get('easychem_mode', 'scan'),
                'max_steps': int(st.session_state.get('easychem_max_steps', 64)),
                'tol': float(st.session_state.get('easychem_tol', 1.0e-11)),
                'throw': bool(st.session_state.get('easychem_throw', True)),
                'relax_limit': float(st.session_state.get('easychem_relax_limit', 0.75)),
                'prefer_chord': bool(st.session_state.get('easychem_prefer_chord', False)),
            },
            'p0_bar': float(st.session_state.get('easychem_p0_bar', 1.0)),
            'e_ref': st.session_state.get('easychem_e_ref', 'H'),
        }
        easychem_elements = _parse_text_list(st.session_state.get('easychem_elements', ''))
        if easychem_elements:
            easychem_block['elements'] = easychem_elements
        config['easychem_jax'] = easychem_block

    elif vert_chem == 'fastchem_grid_jax':
        grid_path = st.session_state.get('fastchem_grid_path', '')
        fc_block = {
            'grid_path': grid_path,
            'solver': {
                'mode': st.session_state.get('fastchem_grid_solver_mode', 'vmap'),
            },
            'bounds': {
                'mode': st.session_state.get('fastchem_grid_bounds_mode', 'clip'),
            },
        }
        species_map = _parse_mapping_lines(st.session_state.get('fastchem_grid_species_map', ''))
        if species_map:
            fc_block['species_map'] = species_map
        config['fastchem_grid_jax'] = fc_block

    elif vert_chem == 'atmodeller':
        species_network = _parse_text_list(st.session_state.get('atmodeller_species_network', ''))
        config['atmodeller'] = {
            'species_network': species_network,
            'solver': {
                'multistart': int(st.session_state.get('atmodeller_solver_multistart', 10)),
                'atol': float(st.session_state.get('atmodeller_solver_atol', 1.0e-6)),
                'rtol': float(st.session_state.get('atmodeller_solver_rtol', 1.0e-6)),
                'max_steps': int(st.session_state.get('atmodeller_solver_max_steps', 256)),
                'jac': st.session_state.get('atmodeller_solver_jac', 'fwd'),
            },
        }

    # -------------------------------------------------------------------------
    # SAMPLING SECTION
    # Bayesian inference engine configuration
    # -------------------------------------------------------------------------
    engine = st.session_state.get('sampling_engine', 'jaxns')
    config['sampling'] = {'engine': engine}

    # Each sampling engine has its own set of hyperparameters
    # These control how the sampler explores parameter space

    if engine == 'jaxns':
        # JAXNS: JAX-accelerated nested sampling
        jaxns_config = {
            'max_samples': st.session_state.get('jaxns_max_samples', 100000),
            'num_live_points': st.session_state.get('jaxns_num_live_points', 500),  # Posterior resolution
            's': st.session_state.get('jaxns_s', 5),  # Slices per dimension
            'k': st.session_state.get('jaxns_k', 0),  # Phantom samples
            'c': st.session_state.get('jaxns_c') or None,  # Parallel chains (None = auto)
            'difficult_model': st.session_state.get('jaxns_difficult_model', False),
            'parameter_estimation': st.session_state.get('jaxns_parameter_estimation', True),
            'gradient_guided': st.session_state.get('jaxns_gradient_guided', False),
            'verbose': st.session_state.get('jaxns_verbose', True),
            'posterior_samples': st.session_state.get('jaxns_posterior_samples', 5000),
            'seed': st.session_state.get('jaxns_seed', 42),  # For reproducibility
        }
        # Optional advanced parameters
        if st.session_state.get('jaxns_shell_fraction'):
            jaxns_config['shell_fraction'] = st.session_state.get('jaxns_shell_fraction')
        if st.session_state.get('jaxns_init_efficiency_threshold'):
            jaxns_config['init_efficiency_threshold'] = st.session_state.get('jaxns_init_efficiency_threshold')

        # Termination criteria - when to stop sampling
        termination = {
            'ess': st.session_state.get('jaxns_ess', 500),  # Effective sample size target
            'dlogZ': st.session_state.get('jaxns_dlogz', 0.01),  # Evidence precision
            'max_samples': st.session_state.get('jaxns_term_max_samples', 100000),
        }
        # Optional termination criteria
        if st.session_state.get('jaxns_evidence_uncert'):
            termination['evidence_uncert'] = st.session_state.get('jaxns_evidence_uncert')
        if st.session_state.get('jaxns_max_num_likelihood_evaluations'):
            termination['max_num_likelihood_evaluations'] = st.session_state.get('jaxns_max_num_likelihood_evaluations')
        if st.session_state.get('jaxns_rtol'):
            termination['rtol'] = st.session_state.get('jaxns_rtol')
        if st.session_state.get('jaxns_atol'):
            termination['atol'] = st.session_state.get('jaxns_atol')

        jaxns_config['termination'] = termination
        config['sampling']['jaxns'] = jaxns_config
    elif engine == 'dynesty':
        # Dynesty: Dynamic nested sampling in pure Python
        dynesty_config = {
            'nlive': st.session_state.get('dynesty_nlive', 500),
            'bound': st.session_state.get('dynesty_bound', 'multi'),  # Bounding method
            'sample': st.session_state.get('dynesty_sample', 'auto'),  # Sampling method
            'dlogz': st.session_state.get('dynesty_dlogz', 0.01),
            'dynamic': st.session_state.get('dynesty_dynamic', False),
            'print_progress': st.session_state.get('dynesty_print_progress', True),
            'seed': st.session_state.get('dynesty_seed', 42),
        }
        # Optional parameters
        if st.session_state.get('dynesty_maxiter'):
            dynesty_config['maxiter'] = st.session_state.get('dynesty_maxiter')
        if st.session_state.get('dynesty_maxcall'):
            dynesty_config['maxcall'] = st.session_state.get('dynesty_maxcall')
        if st.session_state.get('dynesty_bootstrap'):
            dynesty_config['bootstrap'] = st.session_state.get('dynesty_bootstrap')
        if st.session_state.get('dynesty_enlarge'):
            dynesty_config['enlarge'] = st.session_state.get('dynesty_enlarge')
        if st.session_state.get('dynesty_update_interval'):
            dynesty_config['update_interval'] = st.session_state.get('dynesty_update_interval')

        config['sampling']['dynesty'] = dynesty_config
    elif engine == 'blackjax_ns':
        # BlackJAX nested sampling: JAX-based
        config['sampling']['blackjax_ns'] = {
            'num_live_points': st.session_state.get('blackjax_num_live_points', 500),
            'num_inner_steps': st.session_state.get('blackjax_num_inner_steps', 100),
            'num_delete': st.session_state.get('blackjax_num_delete', 1),
            'dlogz_stop': st.session_state.get('blackjax_dlogz_stop', 0.01),
            'seed': st.session_state.get('blackjax_seed', 42),
            'posterior_samples': st.session_state.get('blackjax_posterior_samples', 5000),
        }
    elif engine == 'nuts':
        # NUTS: No-U-Turn Sampler (Hamiltonian Monte Carlo variant)
        config['sampling']['nuts'] = {
            'backend': 'numpyro',  # Use NumPyro as the backend
            'warmup': st.session_state.get('nuts_warmup', 500),  # Burn-in steps
            'draws': st.session_state.get('nuts_draws', 2000),  # Posterior samples
            'chains': st.session_state.get('nuts_chains', 4),  # Parallel chains
            'seed': st.session_state.get('nuts_seed', 42),
        }
    elif engine == 'ultranest':
        # UltraNest: MLFriends nested sampling
        ultranest_config = {
            'num_live_points': st.session_state.get('ultranest_num_live_points', 500),
            'min_num_live_points': st.session_state.get('ultranest_min_live_points', 100),
            'dlogz': st.session_state.get('ultranest_dlogz', 0.01),
            'verbose': st.session_state.get('ultranest_verbose', True),
        }
        # Optional parameters
        if st.session_state.get('ultranest_max_iters'):
            ultranest_config['max_iters'] = st.session_state.get('ultranest_max_iters')
        if st.session_state.get('ultranest_show_status') is not None:
            ultranest_config['show_status'] = st.session_state.get('ultranest_show_status')

        config['sampling']['ultranest'] = ultranest_config
    elif engine == 'pymultinest':
        # PyMultiNest: Python wrapper for MultiNest
        pymultinest_config = {
            'n_live_points': st.session_state.get('pymultinest_n_live_points', 500),
            'evidence_tolerance': st.session_state.get('pymultinest_evidence_tolerance', 0.5),
            'sampling_efficiency': st.session_state.get('pymultinest_sampling_efficiency', 0.3),
            'seed': st.session_state.get('pymultinest_seed', 42),
            'verbose': st.session_state.get('pymultinest_verbose', True),
            'resume': st.session_state.get('pymultinest_resume', False),
        }
        # Optional advanced parameters
        if st.session_state.get('pymultinest_n_iter_before_update'):
            pymultinest_config['n_iter_before_update'] = st.session_state.get('pymultinest_n_iter_before_update')
        if st.session_state.get('pymultinest_null_log_evidence'):
            pymultinest_config['null_log_evidence'] = st.session_state.get('pymultinest_null_log_evidence')
        if st.session_state.get('pymultinest_max_modes'):
            pymultinest_config['max_modes'] = st.session_state.get('pymultinest_max_modes')
        if st.session_state.get('pymultinest_mode_tolerance'):
            pymultinest_config['mode_tolerance'] = st.session_state.get('pymultinest_mode_tolerance')
        if st.session_state.get('pymultinest_importance_nested_sampling') is not None:
            pymultinest_config['importance_nested_sampling'] = st.session_state.get('pymultinest_importance_nested_sampling')
        if st.session_state.get('pymultinest_multimodal') is not None:
            pymultinest_config['multimodal'] = st.session_state.get('pymultinest_multimodal')
        if st.session_state.get('pymultinest_const_efficiency_mode') is not None:
            pymultinest_config['const_efficiency_mode'] = st.session_state.get('pymultinest_const_efficiency_mode')

        config['sampling']['pymultinest'] = pymultinest_config

    # -------------------------------------------------------------------------
    # RUNTIME SECTION
    # Compute platform and resource allocation
    # -------------------------------------------------------------------------
    config['runtime'] = {
        'platform': st.session_state.get('platform', 'gpu'),
    }
    # Only include CUDA devices if specified
    if st.session_state.get('cuda_devices'):
        config['runtime']['cuda_visible_devices'] = st.session_state.cuda_devices
    if st.session_state.get('threads'):
        config['runtime']['threads'] = st.session_state.threads

    return config


def config_to_yaml(config: dict) -> str:
    """
    Convert a configuration dictionary to a YAML-formatted string.

    Uses CustomDumper to render params in flow style (single-line format)
    matching the style used in experiment config files.

    Args:
        config: The configuration dictionary from build_config()

    Returns:
        str: YAML-formatted configuration ready to save to file
    """
    return yaml.dump(
        config,
        Dumper=CustomDumper,       # Use custom dumper for flow-style params
        default_flow_style=False,  # Block style for everything else
        sort_keys=False,           # Preserve insertion order (Python 3.7+)
        allow_unicode=True         # Allow unicode characters
    )


# =============================================================================
# UI COMPONENTS
# =============================================================================
# Each function below renders one section of the configuration form.
# They use Streamlit widgets (st.text_input, st.selectbox, etc.) to create
# interactive form elements. The 'key' parameter links each widget to
# a specific entry in st.session_state for persistence.

def render_data_section():
    """
    Render the DATA configuration section.

    This section collects file paths for:
    - Observation data (the spectrum to fit)
    - Stellar spectrum (for emission spectra normalization)
    - Thermodynamic databases (NASA9 for chemistry calculations)
    """
    st.header("Data Configuration")
    st.markdown("Specify paths to input data files.")

    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)
    with col1:
        # st.text_input creates a single-line text field
        # key="obs_path" means the value is stored in st.session_state['obs_path']
        st.text_input("Observation file path", key="obs_path",
                      placeholder="e.g., ../data/spectrum.txt")
    with col2:
        st.text_input("Stellar spectrum path", key="stellar_path",
                      placeholder="e.g., ../data/stellar.txt")
        st.text_input("NASA9 database path", key="nasa9_path",
                      placeholder="e.g., ../../NASA9")


def render_physics_section():
    """
    Render the PHYSICS configuration section.

    This section configures:
    - Radiative transfer scheme (emission vs transmission)
    - Vertical atmosphere structure (T-P profile, chemistry, clouds)
    - Opacity sources to include (line, Rayleigh, CIA, clouds)
    """
    st.header("Physics Configuration")
    st.caption("Dropdown values use the canonical YAML scheme names used by the current Exo_Skryer registries.")

    # Three-column layout for organized grouping
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Radiative Transfer")
        # st.selectbox creates a dropdown menu
        st.selectbox("RT Scheme", RT_SCHEMES, key="rt_scheme")

        # Conditional UI: only show emission options if emission scheme selected
        # This demonstrates how to make the form dynamic
        if st.session_state.rt_scheme == "emission_1d":
            st.selectbox("Emission Scheme", EM_SCHEMES, key="em_scheme")
            st.selectbox("Emission Mode", EMISSION_MODES, key="emission_mode")

        # st.number_input creates a numeric input with optional bounds
        st.number_input("Number of layers", min_value=10, max_value=500, value=99, key="nlay")
        # st.checkbox creates a toggle
        st.checkbox("Calculate contribution function", key="contri_func")

    with col2:
        st.subheader("Vertical Structure")
        st.selectbox("T-P Profile", VERT_TP_OPTIONS, key="vert_tp")
        st.selectbox("Altitude Reference", VERT_ALT_OPTIONS, key="vert_alt")
        st.selectbox("Chemistry", VERT_CHEM_OPTIONS, key="vert_chem")
        st.selectbox("Mean Molecular Weight", VERT_MU_OPTIONS, key="vert_mu")
        st.selectbox("Cloud Profile", VERT_CLOUD_OPTIONS, key="vert_cloud")

    with col3:
        st.subheader("Opacity Sources")
        st.selectbox("Line Opacity", OPAC_LINE_OPTIONS, key="opac_line")
        st.selectbox("Rayleigh Scattering", OPAC_RAY_OPTIONS, key="opac_ray")
        st.selectbox("CIA Opacity", OPAC_CIA_OPTIONS, key="opac_cia")
        st.selectbox("Cloud Opacity", OPAC_CLOUD_OPTIONS, key="opac_cloud")
        st.selectbox("Special Opacity", OPAC_SPECIAL_OPTIONS, key="opac_special")
        # Only show cloud distribution if clouds are enabled
        if st.session_state.vert_cloud != "None":
            st.selectbox("Cloud Size Distribution", CLOUD_DIST_OPTIONS, key="cloud_dist")
        # Only show refraction for transit mode
        if st.session_state.get('rt_scheme') == "transit_1d":
            st.selectbox("Refraction", REFRACTION_OPTIONS, key="refraction")


def render_opac_section():
    """
    Render the OPAC (opacity) configuration section.

    This section configures:
    - Wavelength grid settings
    - Which molecular/atomic species to include for line opacity
    - Rayleigh scattering species
    - Collision-induced absorption pairs
    - Cloud opacity file paths
    """
    st.header("Opacity Configuration")
    st.info("Prefer `.zarr` or `.zarr.zip` opacity tables in new configs. The current loaders still accept legacy `.npz`/`.h5` in some places, but the app now defaults to Zarr paths.")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Wavelength master grid", key="wl_master",
                      placeholder="e.g., ../../opac_data/wl_R20000.txt")
        st.checkbox("Full wavelength grid", value=True, key="full_grid")
    with col2:
        st.checkbox("Use correlated-k tables", key="use_ck")
        # Only show CK mixing rule if CK is enabled
        if st.session_state.get('use_ck'):
            st.selectbox("CK mixing rule", CK_MIX_OPTIONS, key="ck_mix")

    # -------------------------------------------------------------------------
    # LINE OPACITY SPECIES
    # This demonstrates a dynamic list UI pattern in Streamlit
    # -------------------------------------------------------------------------
    st.subheader("Line Opacity Species")

    # Row for adding new species
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        # Dropdown with empty string as first option (no selection)
        new_species = st.selectbox("Add species", [""] + COMMON_LINE_SPECIES, key="new_line_species")
    with col2:
        new_path = st.text_input("Path (optional)", key="new_line_path",
                                  placeholder="e.g., ../../opac_data/os/H2O_R20000.zarr.zip")
    with col3:
        st.write("")  # Empty space for vertical alignment
        st.write("")  # (buttons need to align with inputs)
        # When button is clicked, add species to the list
        if st.button("Add", key="add_line_species") and new_species:
            st.session_state.line_species.append({'name': new_species, 'path': new_path})
            st.rerun()  # Force script rerun to show updated list

    # Display current species list with remove buttons
    if st.session_state.line_species:
        for i, species in enumerate(st.session_state.line_species):
            col1, col2, col3 = st.columns([2, 4, 1])
            with col1:
                st.text(species['name'])
            with col2:
                st.text(species.get('path', ''))
            with col3:
                # Unique key for each remove button (required by Streamlit)
                if st.button("Remove", key=f"rm_line_{i}"):
                    st.session_state.line_species.pop(i)
                    st.rerun()

    # -------------------------------------------------------------------------
    # RAYLEIGH AND CIA SPECIES
    # These use multiselect for simpler management
    # -------------------------------------------------------------------------
    st.subheader("Rayleigh Scattering Species")
    # st.multiselect allows selecting multiple items from a list
    ray_species = st.multiselect("Select species", COMMON_RAY_SPECIES,
                                  default=st.session_state.ray_species, key="ray_select")
    st.session_state.ray_species = ray_species  # Update session state

    st.subheader("CIA Pairs")
    cia_species = st.multiselect("Select CIA pairs", COMMON_CIA_PAIRS,
                                  default=st.session_state.cia_species, key="cia_select")
    st.session_state.cia_species = cia_species
    st.caption("CIA custom paths should point to `.zarr` or `.zarr.zip` tables when you override the defaults.")

    st.subheader("Special Opacity Sources")
    special_on = st.checkbox("Enable H- continuum (bf/ff)", key="special_hminus_enabled")
    if special_on:
        col_bf, col_ff = st.columns(2)
        with col_bf:
            st.checkbox("H- bound-free (bf)", value=st.session_state.get("special_hminus_bf", True), key="special_hminus_bf")
        with col_ff:
            st.checkbox("H- free-free (ff)", value=st.session_state.get("special_hminus_ff", True), key="special_hminus_ff")

    # Cloud opacity path (only show if cloud opacity is enabled)
    if st.session_state.get('opac_cloud') and st.session_state.opac_cloud != 'None':
        st.subheader("Cloud Opacity")
        st.text_input("Cloud opacity data path", key="cloud_opac_path",
                      placeholder="e.g., ../../opac_data/nk/silicate_nk.txt")


def render_chemistry_backend_section():
    """
    Render backend-specific chemistry configuration blocks.
    """
    st.header("Chemistry Backend")
    vert_chem = st.session_state.get("vert_chem", "constant_vmr")
    st.markdown(f"Current chemistry backend: `{vert_chem}`")

    if vert_chem == "easychem_jax":
        st.info("Requires `data.nasa9`, plus retrieval parameters `M_to_H` and `C_to_O`.")
        st.subheader("easychem_jax")
        st.text_area(
            "Species list",
            key="easychem_species",
            height=220,
            placeholder="One species per line, e.g.\nH2O\nCO\nCO2\nCH4",
        )
        st.text_area(
            "Elements (optional)",
            key="easychem_elements",
            height=120,
            placeholder="One element per line, e.g.\nH\nHe\nC\nN\nO",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Solver mode", ["scan", "vmap"], key="easychem_mode")
            st.number_input("Max steps", min_value=1, value=64, key="easychem_max_steps")
        with col2:
            st.number_input("Tolerance", value=1.0e-11, format="%.3e", key="easychem_tol")
            st.number_input("Relax limit", min_value=0.0, max_value=1.0, value=0.75, key="easychem_relax_limit")
        with col3:
            st.checkbox("Throw on solver failure", key="easychem_throw")
            st.checkbox("Prefer chord fallback", key="easychem_prefer_chord")
            st.number_input("P0 [bar]", min_value=0.0, value=1.0, key="easychem_p0_bar")
            st.text_input("Reference element", key="easychem_e_ref")

    elif vert_chem == "fastchem_grid_jax":
        st.info("Requires retrieval parameters `M_to_H` and `C_to_O`, plus a FastChem 5D NPZ grid file.")
        st.subheader("fastchem_grid_jax")
        st.text_input(
            "Grid path",
            key="fastchem_grid_path",
            placeholder="e.g. ../../FastChem/fastchem_grid_5d_log10.npz",
        )
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Solver mode", ["vmap", "scan"], key="fastchem_grid_solver_mode")
        with col2:
            st.selectbox("Bounds mode", ["clip"], key="fastchem_grid_bounds_mode")
        st.text_area(
            "Species map overrides (optional)",
            key="fastchem_grid_species_map",
            height=120,
            placeholder="One mapping per line, e.g.\nCO: C1O1\nH2O: H2O1",
        )

    elif vert_chem == "ce":
        st.info("FastChem equilibrium backend. No extra top-level config block is required, but add retrieval parameters `M_to_H` and `C_to_O`.")

    elif vert_chem == "rate_ce":
        st.info("RateJAX equilibrium backend. Requires `data.nasa9`, plus retrieval parameters `M_to_H` and `C_to_O`.")

    elif vert_chem == "quench_approx":
        st.info("Quenched-chemistry backend. Requires `data.nasa9`, plus retrieval parameters `M_to_H`, `C_to_O`, `Kzz`, and `log_10_g`.")

    elif vert_chem == "atmodeller":
        st.info("Optional backend. Requires the `atmodeller` extra to be installed, plus retrieval parameters `M_to_H` and `C_to_O`.")
        st.subheader("atmodeller")
        st.text_area(
            "Species network",
            key="atmodeller_species_network",
            height=220,
            placeholder="One species per line, e.g.\nH2O_g\nCO_g\nCH4_g\nH2_g\nHe_g",
        )
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Multistart", min_value=1, value=10, key="atmodeller_solver_multistart")
            st.number_input("Absolute tolerance", value=1.0e-6, format="%.3e", key="atmodeller_solver_atol")
            st.number_input("Relative tolerance", value=1.0e-6, format="%.3e", key="atmodeller_solver_rtol")
        with col2:
            st.number_input("Max steps", min_value=1, value=256, key="atmodeller_solver_max_steps")
            st.selectbox("Jacobian mode", ["fwd", "bwd"], key="atmodeller_solver_jac")

    else:
        st.info("No backend-specific chemistry block is required for the currently selected chemistry scheme.")


def render_params_section():
    """
    Render the PARAMS (retrieval parameters) configuration section.

    This section has two parts:
    1. Fixed Parameters (delta) - constants that are not retrieved
    2. Sampled Parameters (uniform) - parameters to be retrieved with priors

    PARAMETER TYPES:
    - delta: Fixed value (not retrieved) - use for known quantities
    - uniform: Retrieved with flat prior between bounds

    TRANSFORMS:
    - identity: Sample in natural space
    - logit: Transform bounded params to unbounded space (better sampling)
    """
    st.header("Retrieval Parameters")

    # =========================================================================
    # FIXED PARAMETERS (DELTA)
    # =========================================================================
    st.subheader("Fixed Parameters (constants)")
    st.markdown("These parameters are held constant during retrieval.")

    # Input form for fixed parameters
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        delta_name = st.text_input("Parameter name", key="new_delta_name",
                                    placeholder="e.g., R_s, p_bot, p_top")
    with col2:
        delta_value = st.number_input("Value", key="new_delta_value", value=0.0, format="%g")
    with col3:
        st.write("")  # Spacer for alignment
        if st.button("Add Fixed", key="add_delta_btn"):
            if delta_name:
                add_parameter(delta_name, "delta", value=delta_value, transform="identity")
                st.rerun()

    # Display current fixed parameters
    delta_params = [p for p in st.session_state.params if p['dist'] == 'delta']
    if delta_params:
        for i, param in enumerate(st.session_state.params):
            if param['dist'] != 'delta':
                continue
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.text(param['name'])
            with col2:
                st.text(f"{param.get('value', 'N/A')}")
            with col3:
                if st.button("X", key=f"rm_delta_{i}"):
                    remove_parameter(i)
                    st.rerun()
    else:
        st.info("No fixed parameters added yet.")

    st.divider()

    # =========================================================================
    # SAMPLED PARAMETERS (UNIFORM)
    # =========================================================================
    st.subheader("Sampled Parameters (retrieved)")
    st.markdown("These parameters are retrieved with uniform priors.")

    chem_backend = st.session_state.get("vert_chem", "constant_vmr")
    if chem_backend in {"ce", "rate_ce", "fastchem_grid_jax", "easychem_jax", "atmodeller", "quench_approx"}:
        st.markdown("**Chemistry helper**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Add M_to_H and C_to_O", key="add_bulk_chem_params"):
                existing = {p.get("name") for p in st.session_state.params}
                if "M_to_H" not in existing:
                    add_parameter("M_to_H", "uniform", low=-2.0, high=3.0, transform="logit", init=0.0)
                if "C_to_O" not in existing:
                    add_parameter("C_to_O", "uniform", low=0.1, high=2.0, transform="logit", init=0.55)
                st.rerun()
        with col_b:
            if chem_backend == "quench_approx" and st.button("Add quench params", key="add_quench_param_bundle"):
                existing = {p.get("name") for p in st.session_state.params}
                if "M_to_H" not in existing:
                    add_parameter("M_to_H", "uniform", low=-2.0, high=3.0, transform="logit", init=0.0)
                if "C_to_O" not in existing:
                    add_parameter("C_to_O", "uniform", low=0.1, high=2.0, transform="logit", init=0.55)
                if "Kzz" not in existing:
                    add_parameter("Kzz", "uniform", low=1e6, high=1e10, transform="logit", init=1e8)
                if "log_10_g" not in existing:
                    add_parameter("log_10_g", "uniform", low=2.0, high=5.5, transform="logit", init=3.5)
                st.rerun()

    # Quick-add helpers for common parameter bundles
    if st.session_state.get("special_hminus_enabled", False):
        st.markdown("**H- helper**: H- free-free uses `log_10_ne_over_ntot` (log10 of n_e/n_tot) and `log_10_H_over_H2` (log10 of H/H2).")
        if st.button("Add H- params (recommended)", key="add_hminus_param_bundle"):
            existing = {p.get("name") for p in st.session_state.params}

            def ensure_uniform(name, low, high, init):
                if name in existing:
                    return
                add_parameter(name, "uniform", low=low, high=high, transform="logit", init=init)
                existing.add(name)

            ensure_uniform("log_10_f_H-", low=-12, high=-2, init=-6)
            if st.session_state.get("special_hminus_ff", True):
                ensure_uniform("log_10_H_over_H2", low=-12, high=2, init=-6)
                ensure_uniform("log_10_ne_over_ntot", low=-12, high=-2, init=-7)
            st.rerun()

    # Input form for sampled parameters
    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 1])
    with col1:
        uniform_name = st.text_input("Parameter name", key="new_uniform_name",
                                      placeholder="e.g., log_10_f_H2O")
    with col2:
        uniform_low = st.number_input("Lower", key="new_uniform_low", value=0.0, format="%g")
    with col3:
        uniform_high = st.number_input("Upper", key="new_uniform_high", value=0.0, format="%g")
    with col4:
        uniform_init = st.number_input("Init", key="new_uniform_init", value=0.0, format="%g",
                                        help="Initial value (optional)")
    with col5:
        st.write("")  # Spacer for alignment
        if st.button("Add Sampled", key="add_uniform_btn"):
            if uniform_name:
                init_val = uniform_init if uniform_init != 0 else None
                add_parameter(uniform_name, "uniform", low=uniform_low, high=uniform_high,
                            transform="logit", init=init_val)
                st.rerun()

    # Display current sampled parameters
    uniform_params = [p for p in st.session_state.params if p['dist'] == 'uniform']
    if uniform_params:
        # Header row
        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 1])
        with col1:
            st.markdown("**Name**")
        with col2:
            st.markdown("**Lower**")
        with col3:
            st.markdown("**Upper**")
        with col4:
            st.markdown("**Init**")

        for i, param in enumerate(st.session_state.params):
            if param['dist'] != 'uniform':
                continue
            col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 1])
            with col1:
                st.text(param['name'])
            with col2:
                st.text(f"{param.get('low', 'N/A')}")
            with col3:
                st.text(f"{param.get('high', 'N/A')}")
            with col4:
                st.text(f"{param.get('init', '-')}")
            with col5:
                if st.button("X", key=f"rm_uniform_{i}"):
                    remove_parameter(i)
                    st.rerun()
    else:
        st.info("No sampled parameters added yet.")

    st.divider()

    # Clear all button
    if st.session_state.params and st.button("Clear All Parameters"):
        st.session_state.params = []
        st.rerun()


def render_sampling_section():
    """
    Render the SAMPLING configuration section.

    This section configures the Bayesian inference engine used to explore
    the parameter space and estimate posteriors. Each engine has different
    strengths and hyperparameters.

    NESTED SAMPLING ENGINES (estimate evidence + posteriors):
    - jaxns: JAX-accelerated, good for GPU
    - dynesty: Pure Python, well-tested
    - blackjax_ns: JAX-based, modern
    - ultranest: MLFriends algorithm
    - pymultinest: Wrapper for MultiNest

    MCMC ENGINE:
    - nuts: No-U-Turn Sampler (posteriors only, no evidence)
    """
    st.header("Sampling Configuration")

    # Main engine selection
    engine = st.selectbox("Sampling Engine", SAMPLING_ENGINES, key="sampling_engine")

    st.divider()

    # -------------------------------------------------------------------------
    # ENGINE-SPECIFIC OPTIONS
    # Each engine has its own hyperparameters shown conditionally
    # -------------------------------------------------------------------------

    if engine == "jaxns":
        st.subheader("JAXNS Configuration")

        with st.expander("Basic Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.number_input("Max samples", min_value=1000, value=100000, key="jaxns_max_samples")
                st.number_input("Num live points", min_value=50, value=1000, key="jaxns_num_live_points")
                st.number_input("Slices per dimension (s)", min_value=1, value=4, key="jaxns_s")
            with col2:
                st.number_input("Phantom samples (k)", min_value=0, value=0, key="jaxns_k")
                st.number_input("Parallel chains (c)", min_value=0, value=0, key="jaxns_c",
                               help="0 = auto")
                st.number_input("Posterior samples", min_value=100, value=10000, key="jaxns_posterior_samples")
            with col3:
                st.checkbox("Difficult model", key="jaxns_difficult_model")
                st.checkbox("Parameter estimation", value=True, key="jaxns_parameter_estimation")
                st.checkbox("Gradient guided", key="jaxns_gradient_guided")
                st.checkbox("Verbose", value=True, key="jaxns_verbose")

        with st.expander("Termination Criteria", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.number_input("ESS target", min_value=10, value=10000, key="jaxns_ess")
                st.number_input("dlogZ", min_value=0.001, value=0.1, format="%.4f", key="jaxns_dlogz")
            with col2:
                st.number_input("Term max samples", min_value=1000, value=100000, key="jaxns_term_max_samples")
                st.number_input("Evidence uncert (optional)", min_value=0.0, value=0.0, format="%.4f", key="jaxns_evidence_uncert")
            with col3:
                st.number_input("Max likelihood evals (optional)", min_value=0, value=0, key="jaxns_max_num_likelihood_evaluations")
                st.number_input("rtol (optional)", min_value=0.0, value=0.0, format="%.6f", key="jaxns_rtol")

        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Shell fraction (optional)", min_value=0.0, max_value=1.0, value=0.0, format="%.2f", key="jaxns_shell_fraction")
                st.number_input("Init efficiency threshold (optional)", min_value=0.0, max_value=1.0, value=0.0, format="%.2f", key="jaxns_init_efficiency_threshold")
            with col2:
                st.number_input("atol (optional)", min_value=0.0, value=0.0, format="%.6f", key="jaxns_atol")
                st.number_input("Random seed", min_value=0, value=42, key="jaxns_seed")

    elif engine == "dynesty":
        st.subheader("Dynesty Configuration")

        with st.expander("Basic Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("N live points", min_value=50, value=1000, key="dynesty_nlive")
                st.selectbox("Bounding method", ["multi", "single", "balls", "cubes"], key="dynesty_bound")
                st.selectbox("Sampling method", ["auto", "unif", "rwalk", "slice", "rslice"], key="dynesty_sample")
            with col2:
                st.number_input("dlogz", min_value=0.001, value=0.1, format="%.4f", key="dynesty_dlogz")
                st.checkbox("Dynamic nested sampling", key="dynesty_dynamic")
                st.checkbox("Print progress", value=True, key="dynesty_print_progress")

        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Max iterations (optional)", min_value=0, value=0, key="dynesty_maxiter", help="0 = unlimited")
                st.number_input("Max function calls (optional)", min_value=0, value=0, key="dynesty_maxcall", help="0 = unlimited")
                st.number_input("Bootstrap (optional)", min_value=0, value=0, key="dynesty_bootstrap")
            with col2:
                st.number_input("Enlarge factor (optional)", min_value=0.0, value=0.0, format="%.2f", key="dynesty_enlarge")
                st.number_input("Update interval (optional)", min_value=0, value=0, key="dynesty_update_interval")
                st.number_input("Random seed", min_value=0, value=42, key="dynesty_seed")

    elif engine == "blackjax_ns":
        st.subheader("BlackJAX NS Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Num live points", min_value=50, value=1000, key="blackjax_num_live_points")
            st.number_input("Num inner steps", min_value=10, value=30, key="blackjax_num_inner_steps")
            st.number_input("Num delete", min_value=1, value=500, key="blackjax_num_delete")
        with col2:
            st.number_input("dlogz stop", min_value=0.001, value=0.1, format="%.4f", key="blackjax_dlogz_stop")
            st.number_input("Posterior samples", min_value=100, value=10000, key="blackjax_posterior_samples")
            st.number_input("Random seed", min_value=0, value=42, key="blackjax_seed")

    elif engine == "nuts":
        st.subheader("NUTS (NumPyro) Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Warmup steps", min_value=100, value=1000, key="nuts_warmup")
            st.number_input("Draws", min_value=100, value=1000, key="nuts_draws")
        with col2:
            st.number_input("Chains", min_value=1, value=1, key="nuts_chains")
            st.number_input("Random seed", min_value=0, value=42, key="nuts_seed")

    elif engine == "ultranest":
        st.subheader("UltraNest Configuration")

        with st.expander("Basic Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Num live points", min_value=50, value=1000, key="ultranest_num_live_points")
                st.number_input("Min live points", min_value=10, value=1000, key="ultranest_min_live_points")
            with col2:
                st.number_input("dlogz", min_value=0.001, value=0.1, format="%.4f", key="ultranest_dlogz")
                st.checkbox("Verbose", value=True, key="ultranest_verbose")

        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Max iterations (optional)", min_value=0, value=0, key="ultranest_max_iters", help="0 = unlimited")
            with col2:
                st.checkbox("Show status", value=True, key="ultranest_show_status")

    elif engine == "pymultinest":
        st.subheader("PyMultiNest Configuration")

        with st.expander("Basic Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("N live points", min_value=50, value=1000, key="pymultinest_n_live_points")
                st.number_input("Evidence tolerance", min_value=0.01, value=0.1, format="%.2f", key="pymultinest_evidence_tolerance")
                st.number_input("Sampling efficiency", min_value=0.01, max_value=1.0, value=0.3, format="%.2f", key="pymultinest_sampling_efficiency")
            with col2:
                st.checkbox("Verbose", value=True, key="pymultinest_verbose")
                st.checkbox("Resume", value=True, key="pymultinest_resume")
                st.number_input("Random seed", min_value=-1, value=-1, key="pymultinest_seed",
                               help="-1 = random seed")

        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("N iter before update (optional)", min_value=0, value=0, key="pymultinest_n_iter_before_update")
                st.number_input("Null log evidence (optional)", value=0.0, format="%.2e", key="pymultinest_null_log_evidence")
                st.number_input("Max modes (optional)", min_value=0, value=0, key="pymultinest_max_modes")
                st.number_input("Mode tolerance (optional)", value=0.0, format="%.2e", key="pymultinest_mode_tolerance")
            with col2:
                st.checkbox("Importance nested sampling", key="pymultinest_importance_nested_sampling")
                st.checkbox("Multimodal", value=True, key="pymultinest_multimodal")
                st.checkbox("Constant efficiency mode", key="pymultinest_const_efficiency_mode")


def render_runtime_section():
    """
    Render the RUNTIME configuration section.

    This section configures:
    - Compute platform (GPU vs CPU)
    - CUDA device selection (for multi-GPU systems)
    - CPU thread count
    """
    st.header("Runtime Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Platform", RUNTIME_PLATFORMS, key="platform")
    with col2:
        st.text_input("CUDA visible devices", key="cuda_devices", placeholder="e.g., 0,1")
    with col3:
        st.number_input("CPU threads", min_value=1, value=1, key="threads")


def render_yaml_preview():
    """
    Render the YAML preview and download section.

    This section:
    - Builds the config from current session state
    - Converts it to YAML format
    - Displays the YAML for review
    - Provides download button to save the file
    """
    st.header("Generated YAML Configuration")

    # Build and convert the configuration
    config = build_config()
    yaml_str = config_to_yaml(config)

    col1, col2 = st.columns([3, 1])
    with col1:
        # st.code displays formatted code with syntax highlighting
        st.code(yaml_str, language="yaml")
    with col2:
        # st.download_button creates a file download
        st.download_button(
            label="Download YAML",
            data=yaml_str,
            file_name="retrieval_config.yaml",
            mime="text/yaml"
        )

        if st.button("Copy to Clipboard"):
            # Note: Direct clipboard access isn't available in Streamlit
            # This is a workaround to show copyable text
            st.code(yaml_str)
            st.info("Copy the YAML above manually (Ctrl+C / Cmd+C)")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main entry point for the Streamlit application.

    This function:
    1. Initializes session state with defaults
    2. Renders the title and description
    3. Creates sidebar navigation
    4. Renders the selected section based on navigation choice
    """
    # Initialize session state (only sets values if they don't exist)
    init_session_state()

    # Page title
    st.title("Exo_Skryer Configuration Generator")
    st.markdown("Generate YAML configuration files for atmospheric retrieval.")

    # -------------------------------------------------------------------------
    # SIDEBAR NAVIGATION
    # The sidebar provides persistent navigation across sections
    # -------------------------------------------------------------------------
    st.sidebar.title("Sections")
    # st.radio creates a single-select list
    section = st.sidebar.radio(
        "Navigate to:",
        ["Data", "Physics", "Chemistry Backend", "Opacity", "Parameters", "Sampling", "Runtime", "Preview & Export"]
    )

    # Quick info panel in sidebar (shows current state at a glance)
    st.sidebar.divider()
    st.sidebar.markdown("### Quick Info")
    st.sidebar.markdown(f"**RT Scheme:** {st.session_state.get('rt_scheme', 'N/A')}")
    st.sidebar.markdown(f"**Sampler:** {st.session_state.get('sampling_engine', 'N/A')}")
    st.sidebar.markdown(f"**Parameters:** {len(st.session_state.params)}")
    st.sidebar.markdown(f"**Line Species:** {len(st.session_state.line_species)}")

    # -------------------------------------------------------------------------
    # RENDER SELECTED SECTION
    # Based on sidebar selection, render the appropriate section
    # -------------------------------------------------------------------------
    if section == "Data":
        render_data_section()
    elif section == "Physics":
        render_physics_section()
    elif section == "Chemistry Backend":
        render_chemistry_backend_section()
    elif section == "Opacity":
        render_opac_section()
    elif section == "Parameters":
        render_params_section()
    elif section == "Sampling":
        render_sampling_section()
    elif section == "Runtime":
        render_runtime_section()
    elif section == "Preview & Export":
        render_yaml_preview()


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
# This block runs when the script is executed directly (not imported)
# Streamlit internally calls main() when serving the app

if __name__ == "__main__":
    main()
