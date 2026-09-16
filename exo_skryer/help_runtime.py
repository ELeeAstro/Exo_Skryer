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
_SUPPORTED_PLATFORMS = frozenset(("cpu", "cuda", "rocm", "metal"))


def normalize_runtime_platform(platform):
    """Return the JAX backend name for a configured runtime platform."""
    normalized = str(platform or "cpu").strip().lower()
    normalized = _PLATFORM_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_PLATFORMS:
        choices = "cpu, cuda, rocm, metal (or the legacy gpu alias)"
        raise ValueError(
            f"Unsupported runtime.platform {platform!r}; choose one of: {choices}"
        )
    return normalized


def _append_env_flags(name, flags):
    """Append flags without discarding flags supplied by the user."""
    current = os.environ.get(name, "").strip()
    os.environ[name] = " ".join(part for part in (current, flags) if part)


def apply_runtime_env(rc):
    """Apply runtime settings before importing JAX or NumPyro.

    ``rc`` may be the complete configuration object (with a ``runtime``
    attribute) or the runtime namespace itself. Supported settings are::

      runtime:
        platform: cpu | gpu | cuda | rocm | metal
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
    # JAX releases and its generic "gpu" value does not include Metal.
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
            "--xla_gpu_enable_fast_min_max=true "
            "--xla_gpu_deterministic_ops=true",
        )
    elif platform == "rocm":
        visible = getattr(rt, "rocm_visible_devices", None)
        if visible is not None and str(visible) != "":
            # AMD recommends ROCR_VISIBLE_DEVICES for isolation on Linux.
            os.environ["ROCR_VISIBLE_DEVICES"] = str(visible)
    elif platform == "metal":
        # Avoid probing an installed CUDA plugin on an Apple-only run.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # Avoid accelerator probing and expose configured CPU parallelism.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        threads = getattr(rt, "threads", None)
        if threads is not None:
            _append_env_flags(
                "XLA_FLAGS",
                "--xla_cpu_multi_thread_eigen=true "
                f"intra_op_parallelism_threads={int(threads)}",
            )

    if platform in ("cuda", "rocm"):
        preallocate = getattr(rt, "preallocate", False)
        os.environ.setdefault(
            "XLA_PYTHON_CLIENT_PREALLOCATE", "true" if preallocate else "false"
        )
        mem_fraction = getattr(rt, "mem_fraction", 0.75)
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))

    np_platform = getattr(rt, "numpyro_platform", None)
    if np_platform:
        np_platform = str(np_platform).lower()
        if np_platform not in ("cpu", "gpu"):
            raise ValueError("runtime.numpyro_platform must be 'cpu' or 'gpu'")
        os.environ["JAX_PLATFORM_NAME"] = np_platform

    jax_threads = getattr(rt, "jax_threads", None)
    if jax_threads is not None:
        os.environ["JAX_NUM_THREADS"] = str(int(jax_threads))

    blackjax_threads = getattr(rt, "blackjax_threads", None)
    if blackjax_threads is not None:
        _append_env_flags(
            "XLA_FLAGS",
            f"--xla_force_host_platform_device_count={int(blackjax_threads)}",
        )

    numpyro_threads = getattr(rt, "numpyro_threads", None)
    return int(numpyro_threads) if numpyro_threads is not None else None
