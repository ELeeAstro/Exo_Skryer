"""
help_runtime.py
===============

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

import os
import multiprocessing  # still here if you want to auto-detect cores later

__all__ = ['apply_runtime_env']


def apply_runtime_env(rc):
    """
    Apply optional rc.runtime settings *before* importing JAX / NumPyro.

    Expected (all optional):

      runtime:
        platform: cpu | gpu | cuda | metal
        cuda_visible_devices: "0" or "0,1"
        preallocate: true/false
        mem_fraction: 0.8
        jax_threads: 4
        blackjax_threads: 4
        numpyro_threads: 4
        numpyro_platform: cpu | gpu   # optional override for NumPyro/JAX device
        jax_x64: true/false

    Returns
    -------
    numpyro_threads : int | None
        Hint for numpyro.set_host_device_count in your driver.
    """
    rt = getattr(rc, "runtime", None)
    if rt is None:
        return None

    # --------------------
    # JAX platform / device
    # --------------------
    plat = getattr(rt, "platform", None)
    if plat:
        plat_norm = str(plat).lower()

        # Convenience alias
        if plat_norm == "gpu":
            try:
                sysname = os.uname().sysname.lower()
            except Exception:
                sysname = ""
            # Rough heuristic: macOS → metal, else cuda
            if "darwin" in sysname or "mac" in sysname:
                plat_norm = "metal"
            else:
                plat_norm = "cuda"

        # JAX >= 0.4 prefers JAX_PLATFORMS; we also set JAX_PLATFORM_NAME for older code.
        os.environ["JAX_PLATFORMS"] = plat_norm
        if plat_norm == "cpu":
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
        elif plat_norm in ("cuda", "metal", "rocm"):
            # JAX treats these as "gpu" at the higher level
            os.environ["JAX_PLATFORM_NAME"] = "gpu"

    # Optional: NumPyro-specific override (still via JAX, but gives you a knob
    # that you conceptually treat as "NumPyro device").
    np_plat = getattr(rt, "numpyro_platform", None)
    if np_plat:
        np_plat_norm = str(np_plat).lower()
        if np_plat_norm in ("cpu", "gpu"):
            os.environ["JAX_PLATFORM_NAME"] = np_plat_norm

    # --------------------
    # CUDA device visibility
    # --------------------
    cvis = getattr(rt, "cuda_visible_devices", None)
    if cvis is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cvis)

    # --------------------
    # XLA memory behaviour
    # --------------------
    prealloc = getattr(rt, "preallocate", None)
    if prealloc is not None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if prealloc else "false"

    memfrac = getattr(rt, "mem_fraction", None)
    if memfrac is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memfrac)

    # --------------------
    # Thread / host-device counts
    # --------------------
    jax_threads = getattr(rt, "jax_threads", None)
    if jax_threads is not None:
        os.environ["JAX_NUM_THREADS"] = str(int(jax_threads))

    # For BlackJAX on CPU we sometimes want XLA to expose N "host devices"
    blackjax_threads = getattr(rt, "blackjax_threads", None)
    if blackjax_threads is not None:
        os.environ["XLA_FLAGS"] = (
            "--xla_force_host_platform_device_count="
            + str(int(blackjax_threads))
        )

    # --------------------
    # NumPyro hint: how many host devices / chains to use
    # --------------------
    numpyro_threads = getattr(rt, "numpyro_threads", None)
    return int(numpyro_threads) if numpyro_threads is not None else None
