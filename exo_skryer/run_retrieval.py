"""
run_retrieval.py
================

Overview:
    Main entry point for running a YARN atmospheric retrieval defined by a YAML
    configuration. This script:

    1. Loads the retrieval configuration.
    2. Sets up the JAX/NumPyro runtime.
    3. Reads observational data and optional stellar spectrum.
    4. Builds opacities, instrument responses, and the forward model.
    5. Runs the chosen sampler (NUTS or nested sampling).
    6. Writes posterior samples and a CSV copy of observations to the experiment
       directory.

Sections to complete:
    - Usage
    - Key Functions
    - Notes

Usage
-----
From the repository root:

    python exo_skryer/run_retrieval.py --config experiments/<name>/retrieval_config.yaml

Key Functions
-------------
`main`
    Orchestrates the full retrieval workflow.
`format_duration`
    Helper to pretty-print wall-clock runtimes.

Notes
-----
- Paths in the YAML are resolved relative to the experiment folder.
- For correlated-k opacities, the master wavelength grid must match the table.
"""

import os
import time
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


__all__ = [
    "format_duration",
    "main"
]


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Parameters
    ----------
    seconds
        Duration in seconds.

    Returns
    -------
    str
        Formatted duration as ``"<d>d <h>h <m>m <s>s"``.
    """

    days, rem = divmod(seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    return f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.3f}s"

def _resolve_obs_path(cfg: Any) -> str:
    """Resolve the observational data path from config.

    Parameters
    ----------
    cfg
        Parsed YAML configuration object.

    Returns
    -------
    str
        Observational data path as given in ``cfg.data.obs``.

    Raises
    ------
    ValueError
        If no observational data path is present.
    """
    data_cfg = getattr(cfg, "data", None)
    rel_obs_path: Optional[str] = None

    if data_cfg is not None:
        rel_obs_path = getattr(data_cfg, "obs", None)

    if rel_obs_path is None:
        raise ValueError(
            "No observational data path found. Set cfg.data.obs in the YAML config."
        )

    return rel_obs_path


def _maybe_load_gibbs_cache(cfg: Any, exp_dir: Path) -> None:
    """Load Gibbs free energy tables if chemical equilibrium requires it.

    This is only relevant when ``physics.vert_chem`` indicates a RateJAX chemical
    equilibrium mode.

    Parameters
    ----------
    cfg
        Parsed YAML configuration object.
    exp_dir
        Experiment directory used to resolve relative paths.
    """
    phys = getattr(cfg, "physics", None)
    if phys is None:
        return

    vert_chem_raw = getattr(phys, "vert_chem", None)
    if vert_chem_raw is None:
        return

    vert_chem_name = str(vert_chem_raw).lower()
    if vert_chem_name not in ("rate_ce", "rate_jax", "ce_rate_jax"):
        return

    from rate_jax import is_gibbs_cache_loaded, load_gibbs_cache

    if is_gibbs_cache_loaded():
        print("[info] Gibbs cache already loaded")
        return

    data_cfg = getattr(cfg, "data", None)
    janaf_rel_path = getattr(data_cfg, "janaf", None) if data_cfg is not None else None
    if janaf_rel_path is None:
        raise ValueError(
            "JANAF data path not found in config. Please add 'janaf: path/to/JANAF_data' "
            "under 'data:' section in your YAML config."
        )

    janaf_path = (
        str(exp_dir / janaf_rel_path)
        if not Path(janaf_rel_path).is_absolute()
        else janaf_rel_path
    )

    print(f"[info] Loading Gibbs free energy tables from {janaf_path}")
    gibbs = load_gibbs_cache(janaf_path)
    print(f"[info] Gibbs cache loaded: {len(gibbs.data)} species")


def main() -> None:
    """Run a retrieval defined by a YAML configuration.

    This function coordinates reading configuration and data, preparing opacities
    and instrument responses, building the forward model, running the sampler,
    and saving outputs to the experiment directory.

    Returns
    -------
    None
    """

    # Start runtime counter
    t_start = time.perf_counter()

    # Parse YAML config file from command line
    # Format is --config /path/to/config.yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    args = p.parse_args()

    # Print process ID (for easy kill)
    process_id = os.getpid()
    print("[info] Process ID: ", process_id)
 
    # Resolve experiment folder (for read & write)
    config_path = Path(args.config).resolve()
    exp_dir = config_path.parent

    # Load YAML parameters - make into a dot namespace
    from read_yaml import read_yaml
    cfg = read_yaml(config_path)
    
    # Print main yaml parameters to command line
    from help_print import print_cfg
    print_cfg(cfg)

    # Set runtime environment to use cpus or gpus (for JAX)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.runtime.cuda_visible_devices)
    platform = str(cfg.runtime.platform)  # "cpu" or "gpu"
    #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    #os.environ["TF_GPU_ALLOCATOR=cuda_malloc_async"] = "cuda_malloc_async"

    # Prepare JAX and numpyro JAX settings
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
    jax_config.update("jax_platform_name", platform)
    #jax_config.update("jax_debug_nans", True)

    import numpyro
    numpyro.enable_x64(True)
    numpyro.set_platform(platform)
    if platform == "cpu":
        numpyro.set_host_device_count(cfg.runtime.threads)

    import jax

    # Print the JAX setup
    print(f"[info] JAX backend: {jax.default_backend()}")
    print(f"[info] JAX devices: {jax.local_device_count()} {jax.devices()}")

    # Load the observational data - return a dictionary obs
    from read_obs import read_obs_data
    from read_stellar import read_stellar_spectrum
    rel_obs_path = _resolve_obs_path(cfg)
    obs = read_obs_data(rel_obs_path, base_dir=exp_dir)

    # Load the opacities (if present in YAML file)
    from build_opacities import build_opacities, master_wavelength, master_wavelength_cut
    build_opacities(cfg, obs, exp_dir)
    full_grid = np.asarray(master_wavelength(), dtype=float)
    cut_grid = np.asarray(master_wavelength_cut(), dtype=float)
    print(
        f"[info] Master grid: N={full_grid.size}, range=[{full_grid.min():.5f}, {full_grid.max():.5f}]"
    )
    print(
        f"[info] Cut grid:    N={cut_grid.size}, range=[{cut_grid.min():.5f}, {cut_grid.max():.5f}]"
    )
    # No longer need to pre-create KK cache for direct_nk cloud model

    # Read and prepare any response functions and bandpasses for each observational band
    from registry_bandpass import load_bandpass_registry
    load_bandpass_registry(obs, full_grid, cut_grid)

    # Load Gibbs free energy tables for chemical equilibrium (if using rate_jax)
    _maybe_load_gibbs_cache(cfg, exp_dir)

    # Build the forward model from the YAML options - return a function that samplers can use
    from build_model import build_forward_model
    stellar_flux = read_stellar_spectrum(cfg, cut_grid, bool(cfg.opac.ck), base_dir=exp_dir)

    fm_fnc = build_forward_model(cfg, obs, stellar_flux=stellar_flux)

    # Prepare the main dataclass required for all samplers and forward model
    from build_prepared import build_prepared
    prep = build_prepared(cfg, obs, fm=fm_fnc)

    t_start_2 = time.perf_counter()

    # Prepare the sampling schemes
    print(f"[info] Starting Sampling")

    # Dictionaries for the output of the samplers
    evidence_info: Dict[str, Any] = {}
    samples_dict: Dict[str, Any] = {}

    # Which high-level sampler to use: "nuts", "hmc", "jaxns", "numpyro_ns", or "blackjax_ns"
    engine = cfg.sampling.engine

    if engine == "nuts":
        # Extract backend driver
        backend = cfg.sampling.nuts.backend
        if backend == "blackjax":
            # Blackjax MCMC driver
            from sampler_blackjax_MCMC import run_nuts_blackjax
            samples_dict = run_nuts_blackjax(cfg, prep, exp_dir)
        elif backend == "numpyro":
            # Numpyro MCMC driver
            from sampler_numpyro_MCMC import run_nuts_numpyro
            samples_dict = run_nuts_numpyro(cfg, prep, exp_dir)
        else:
            raise ValueError(f"Unknown backend for NUTS: {backend!r}")

    elif engine == "jaxns":
        # jaxns nested-sampling driver
        from sampler_jaxns_NS import run_nested_jaxns
        samples_dict, evidence_info = run_nested_jaxns(cfg, prep, exp_dir)

    elif engine == "blackjax_ns":
        # BlackJAX nested-sampling driver
        from sampler_blackjax_NS import run_nested_blackjax
        samples_dict, evidence_info = run_nested_blackjax(cfg, prep, exp_dir)
    else:
        raise ValueError(f"Unknown sampling.engine: {engine!r}")

    print(f"[info] Finished Sampling")

    t_end_2 = time.perf_counter()
    print(f"[info] Sampling took:", format_duration(t_end_2 - t_start_2))

    # Output 
    from help_io import to_inferencedata, save_inferencedata, save_observed_data_csv
    samples_np = {k: np.asarray(v) for k, v in samples_dict.items()}
    idata = to_inferencedata(samples_np, cfg, include_fixed=False)
    out_nc = save_inferencedata(idata, exp_dir)
    print(f"[info] ArviZ posterior -> {out_nc}")

    # Save a copy of the observational data to a csv in the experiment directory
    save_observed_data_csv(
        exp_dir,
        lam=obs["wl"],
        dlam=obs["dwl"],
        y=obs["y"],
        dy=obs["dy"],
        response_mode=obs.get("response_mode"),
    )

    print(f"[info] Results saved to: {exp_dir.resolve()}")

    # Print overall runtime
    t_end = time.perf_counter()
    print(f"[done] Full model took:", format_duration(t_end - t_start))

# Calling function
if __name__ == "__main__":
    main()
