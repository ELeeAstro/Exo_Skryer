"""Runtime environment configuration.

This module intentionally does not import JAX. The environment must be set
before JAX (and therefore NumPyro) is imported so that the requested backend
and devices are discovered correctly.
"""

import os

__all__ = ["apply_runtime_env", "normalize_runtime_platform"]


_PLATFORM_ALIASES = {
    "gpu": "cuda",  # Backwards compatibility with existing configurations.
    "amd": "rocm",
    "hip": "rocm",
}
_SUPPORTED_PLATFORMS = frozenset(("cpu", "cuda", "rocm"))


def normalize_runtime_platform(platform):
    """Return the JAX backend name for a configured runtime platform."""
    normalized = str(platform or "cpu").strip().lower()
    normalized = _PLATFORM_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_PLATFORMS:
        choices = "cpu, cuda, rocm (or the legacy gpu alias)"
        raise ValueError(
            f"Unsupported runtime.platform {platform!r}; choose one of: {choices}"
        )
    return normalized


def _append_env_flags(name, flags):
    """Append flags without discarding flags supplied by the user."""
    current = os.environ.get(name, "").strip()
    os.environ[name] = " ".join(part for part in (current, flags) if part)


# Launcher variables giving a process's rank among those on the same node.
_LOCAL_RANK_VARS = (
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "PMI_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID",
)


def _limit_cpu_threads(threads):
    """Restrict this process to ``threads`` CPU cores.

    XLA has no flag for its CPU thread count, but it sizes its thread pool
    from the cores the process may run on, so CPU affinity is the supported
    limit. Processes sharing a node (MPI/Slurm) take consecutive blocks of
    cores by local rank so they do not pile onto the same cores. Affinity is
    Linux-only; elsewhere JAX uses every core.
    """
    if threads < 1:
        raise ValueError(f"runtime.threads must be >= 1, got {threads}")
    if not hasattr(os, "sched_setaffinity"):
        print(
            f"[warn] runtime.threads={threads} cannot be enforced on this OS; "
            "JAX will use all available cores."
        )
        return
    allowed = sorted(os.sched_getaffinity(0))
    if threads >= len(allowed):
        return
    local_rank = next(
        (int(os.environ[v]) for v in _LOCAL_RANK_VARS if os.environ.get(v, "").isdigit()),
        0,
    )
    start = (local_rank * threads) % len(allowed)
    cores = {allowed[(start + i) % len(allowed)] for i in range(threads)}
    os.sched_setaffinity(0, cores)


def apply_runtime_env(rc):
    """Apply runtime settings before importing JAX or NumPyro.

    ``rc`` may be the complete configuration object (with a ``runtime``
    attribute) or the runtime namespace itself. Supported settings are::

      runtime:
        platform: cpu | gpu | cuda | rocm
        cuda_visible_devices: "0" or "0,1"
        rocm_visible_devices: "0" or "0,1"
        preallocate: true/false
        mem_fraction: 0.75
        threads: 4
        jax_threads: 4
        blackjax_threads: 4
        numpyro_threads: 4
        numpyro_platform: cpu | gpu

    ``gpu`` remains an alias for ``cuda`` for backwards compatibility. AMD
    GPUs and APUs must explicitly select ``rocm``.

    Returns
    -------
    int or None
        The optional NumPyro host-device count hint.
    """
    rt = getattr(rc, "runtime", rc)
    if rt is None:
        return None

    platform = normalize_runtime_platform(getattr(rt, "platform", "cpu"))
    os.environ["JAX_PLATFORMS"] = platform
    # The deprecated setting takes precedence over JAX_PLATFORMS in current
    # JAX releases.
    os.environ.pop("JAX_PLATFORM_NAME", None)

    if platform == "cuda":
        visible = getattr(rt, "cuda_visible_devices", None)
        if visible is not None and str(visible) != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(visible)

        tf_gpu_allocator = getattr(rt, "tf_gpu_allocator", None)
        if tf_gpu_allocator:
            os.environ.setdefault("TF_GPU_ALLOCATOR", str(tf_gpu_allocator))

        _append_env_flags(
            "XLA_FLAGS",
            "--xla_gpu_enable_latency_hiding_scheduler=true "
            "--xla_gpu_enable_highest_priority_async_stream=true "
            "--xla_gpu_deterministic_ops=true",
        )
    elif platform == "rocm":
        visible = getattr(rt, "rocm_visible_devices", None)
        if visible is not None and str(visible) != "":
            # AMD recommends ROCR_VISIBLE_DEVICES for isolation on Linux.
            os.environ["ROCR_VISIBLE_DEVICES"] = str(visible)
    else:
        # Avoid accelerator probing and expose configured CPU parallelism.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # jax_threads is a legacy alias for threads.
        threads = getattr(rt, "threads", None)
        if threads is None:
            threads = getattr(rt, "jax_threads", None)
        if threads is not None:
            _limit_cpu_threads(int(threads))

    if platform in ("cuda", "rocm"):
        preallocate = getattr(rt, "preallocate", False)
        os.environ.setdefault(
            "XLA_PYTHON_CLIENT_PREALLOCATE", "true" if preallocate else "false"
        )
        mem_fraction = getattr(rt, "mem_fraction", 0.75)
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))

    # JAX_PLATFORM_NAME (what numpyro.set_platform used) no longer exists in
    # JAX, so an explicit NumPyro platform overrides JAX_PLATFORMS instead.
    np_platform = getattr(rt, "numpyro_platform", None)
    if np_platform:
        os.environ["JAX_PLATFORMS"] = normalize_runtime_platform(np_platform)

    blackjax_threads = getattr(rt, "blackjax_threads", None)
    if blackjax_threads is not None:
        _append_env_flags(
            "XLA_FLAGS",
            f"--xla_force_host_platform_device_count={int(blackjax_threads)}",
        )

    numpyro_threads = getattr(rt, "numpyro_threads", None)
    return int(numpyro_threads) if numpyro_threads is not None else None
