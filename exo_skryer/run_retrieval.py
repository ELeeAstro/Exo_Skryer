"""
run_retrieval.py
================
"""

import os
import time
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .help_print import format_duration


__all__ = [
    "main",
    "format_duration",
]


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
    from .read_yaml import read_yaml
    cfg = read_yaml(config_path)

    # Platform selection from YAML (used for sampler setup). We keep CPU environment
    # variables at JAX defaults; GPU-specific flags remain configurable.
    platform = str(getattr(cfg.runtime, "platform", "cpu")).lower()

    if platform == "gpu":
        cuda_devices = str(getattr(cfg.runtime, "cuda_visible_devices", ""))
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
        tf_gpu_allocator = getattr(cfg.runtime, "tf_gpu_allocator", None)
        if tf_gpu_allocator:
            os.environ.setdefault("TF_GPU_ALLOCATOR", str(tf_gpu_allocator))

        xla_flags = (
            "--xla_gpu_enable_latency_hiding_scheduler=true "
            "--xla_gpu_enable_highest_priority_async_stream=true "
            "--xla_gpu_enable_fast_min_max=true "
            "--xla_gpu_deterministic_ops=true"
        )
        os.environ["XLA_FLAGS"] = xla_flags

        print(f"[info] Platform: GPU (CUDA_VISIBLE_DEVICES={cuda_devices})")
        print("[info] XLA GPU: latency hiding, async streams, fast math enabled")
    else:
        print("[info] Platform: CPU (JAX defaults)")

    # Print main yaml parameters to command line (after setting environment)
    from .help_print import print_cfg
    print_cfg(cfg)

    # Prepare JAX and numpyro JAX settings
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
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
    from .read_obs import resolve_obs_path, read_obs_data
    from .read_stellar import read_stellar_spectrum
    rel_obs_path = resolve_obs_path(cfg)
    obs = read_obs_data(rel_obs_path, base_dir=exp_dir)

    # Load the opacities (if present in YAML file)
    from .build_opacities import build_opacities, master_wavelength, master_wavelength_cut
    build_opacities(cfg, obs, exp_dir)
    full_grid = np.asarray(master_wavelength(), dtype=float)
    cut_grid = np.asarray(master_wavelength_cut(), dtype=float)
    print(
        f"[info] Master grid: N={full_grid.size}, range=[{full_grid.min():.5f}, {full_grid.max():.5f}]"
    )
    print(
        f"[info] Cut grid:    N={cut_grid.size}, range=[{cut_grid.min():.5f}, {cut_grid.max():.5f}]"
    )

    # Read and prepare any response functions and bandpasses for each observational band
    from .registry_bandpass import load_bandpass_registry
    load_bandpass_registry(obs, full_grid, cut_grid)

    # Load Gibbs free energy tables for chemical equilibrium (if using rate_jax)
    from .build_chem import load_nasa9_if_needed
    load_nasa9_if_needed(cfg, exp_dir)

    # Build the forward model from the YAML options - return a function that samplers can use
    from .build_model import build_forward_model
    stellar_flux = read_stellar_spectrum(cfg, cut_grid, bool(cfg.opac.ck), base_dir=exp_dir)

    fm_fnc = build_forward_model(cfg, obs, stellar_flux=stellar_flux)

    # All samplers are now self-contained - no build_prepared needed!
    t_start_2 = time.perf_counter()

    # Prepare the sampling schemes
    print(f"[info] Starting Sampling")

    # Dictionaries for the output of the samplers
    evidence_info: Dict[str, Any] = {}
    samples_dict: Dict[str, Any] = {}

    engine = cfg.sampling.engine

    if engine == "nuts":
        # Extract backend driver
        backend = cfg.sampling.nuts.backend
        if backend == "blackjax":
            # Blackjax MCMC driver (self-contained)
            from .sampler_blackjax_MCMC import run_nuts_blackjax
            samples_dict = run_nuts_blackjax(cfg, obs, fm_fnc, exp_dir)
        elif backend == "numpyro":
            # Numpyro MCMC driver (self-contained)
            from .sampler_numpyro_MCMC import run_nuts_numpyro
            samples_dict = run_nuts_numpyro(cfg, obs, fm_fnc, exp_dir)
        else:
            raise ValueError(f"Unknown backend for NUTS: {backend!r}")

    elif engine == "jaxns":
        # jaxns nested-sampling driver (self-contained)
        from .sampler_jaxns_NS import run_nested_jaxns
        samples_dict, evidence_info = run_nested_jaxns(cfg, obs, fm_fnc, exp_dir)

    elif engine == "blackjax_ns":
        # BlackJAX nested-sampling driver (self-contained)
        from .sampler_blackjax_NS import run_nested_blackjax
        samples_dict, evidence_info = run_nested_blackjax(cfg, obs, fm_fnc, exp_dir)

    elif engine == "ultranest":
        # UltraNest nested-sampling driver (self-contained)
        from .sampler_ultranest_NS import run_nested_ultranest
        samples_dict, evidence_info = run_nested_ultranest(cfg, obs, fm_fnc, exp_dir)

    elif engine == "dynesty":
        # Dynesty nested-sampling driver (self-contained)
        from .sampler_dynesty_NS import run_nested_dynesty
        samples_dict, evidence_info = run_nested_dynesty(cfg, obs, fm_fnc, exp_dir)

    elif engine == "pymultinest":
        # PyMultiNest nested-sampling driver (self-contained)
        from .sampler_pymultinest_NS import run_nested_pymultinest
        samples_dict, evidence_info = run_nested_pymultinest(cfg, obs, fm_fnc, exp_dir)

    elif engine == "nautilus":
        # Nautilus nested-sampling driver (self-contained)
        from .sampler_nautilus_NS import run_nested_nautilus
        samples_dict, evidence_info = run_nested_nautilus(cfg, obs, fm_fnc, exp_dir)

    else:
        raise ValueError(f"Unknown sampling.engine: {engine!r}. Options: nuts, jaxns, blackjax_ns, ultranest, dynesty, pymultinest, nautilus")

    print(f"[info] Finished Sampling")

    t_end_2 = time.perf_counter()
    print(f"[info] Sampling took:", format_duration(t_end_2 - t_start_2))

    # Output
    from .help_io import to_inferencedata, save_inferencedata, save_observed_data_csv
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
        offset_group=obs.get("offset_group"),
    )

    print(f"[info] Results saved to: {exp_dir.resolve()}")

    # Print overall runtime
    t_end = time.perf_counter()
    print(f"[done] Full model took:", format_duration(t_end - t_start))

# Calling function
if __name__ == "__main__":
    main()
