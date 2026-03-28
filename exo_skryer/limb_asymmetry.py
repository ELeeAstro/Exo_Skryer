"""
limb_asymmetry.py
=================

Helpers for the transit_2d explicit _joint/_east/_west parameter mode.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import jax.numpy as jnp

__all__ = [
    "LIMB_TAGS",
    "JOINT_TAG",
    "EAST_TAG",
    "WEST_TAG",
    "OFFSET_VECTOR_KEY",
    "split_limb_tag",
    "is_internal_parameter_name",
    "validate_limb_parameter_names",
    "split_limb_parameter_dict",
    "merge_limb_parameter_dict",
    "get_limb_value",
    "jitter_param_name",
    "parse_offset_group_name",
]

JOINT_TAG = "joint"
EAST_TAG = "east"
WEST_TAG = "west"
LIMB_TAGS: tuple[str, str, str] = (JOINT_TAG, EAST_TAG, WEST_TAG)
OFFSET_VECTOR_KEY = "__offset_values__"


def split_limb_tag(name: str) -> tuple[str, str | None]:
    """Return ``(base_name, tag)`` where tag is joint/east/west or ``None``."""
    for tag in LIMB_TAGS:
        token = f"_{tag}"
        if name.endswith(token):
            return name[: -len(token)], tag
    return name, None


def is_internal_parameter_name(name: str) -> bool:
    """Return whether a runtime-only parameter name is exempt from tag validation."""
    return str(name).startswith("__")


def validate_limb_parameter_names(names: Iterable[str]) -> set[str]:
    """Validate explicit transit_2d tagging and return east/west duplicated bases."""
    tagged: dict[str, set[str]] = {}
    untagged: set[str] = set()

    for raw_name in names:
        name = str(raw_name)
        if is_internal_parameter_name(name):
            continue
        base, tag = split_limb_tag(name)
        if tag is None:
            untagged.add(base)
            continue
        tagged.setdefault(base, set()).add(tag)

    if untagged:
        joined = ", ".join(sorted(untagged))
        raise ValueError(
            "transit_2d requires every YAML parameter to use an explicit suffix: "
            f"_joint, _east, or _west. Untagged parameter(s): {joined}"
        )

    mixed_joint = sorted(
        base for base, tags in tagged.items() if JOINT_TAG in tags and (EAST_TAG in tags or WEST_TAG in tags)
    )
    if mixed_joint:
        joined = ", ".join(mixed_joint)
        raise ValueError(
            "transit_2d parameters cannot mix _joint with _east/_west for the same base name: "
            f"{joined}"
        )

    missing_pairs = sorted(
        base for base, tags in tagged.items() if tags in ({EAST_TAG}, {WEST_TAG}, {EAST_TAG, JOINT_TAG}, {WEST_TAG, JOINT_TAG})
    )
    if missing_pairs:
        joined = ", ".join(missing_pairs)
        raise ValueError(
            "transit_2d requires both _east and _west entries for each limb-specific base name. "
            f"Missing partner(s) for: {joined}"
        )

    duplicated = {base for base, tags in tagged.items() if tags == {EAST_TAG, WEST_TAG}}
    return duplicated


def split_limb_parameter_dict(
    params: Mapping[str, jnp.ndarray],
) -> tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Split explicit-tag transit_2d params into joint/east/west dictionaries."""
    validate_limb_parameter_names(params.keys())

    joint: Dict[str, jnp.ndarray] = {}
    east: Dict[str, jnp.ndarray] = {}
    west: Dict[str, jnp.ndarray] = {}

    for name, value in params.items():
        if is_internal_parameter_name(name):
            joint[name] = value
            continue
        base, tag = split_limb_tag(str(name))
        if tag == JOINT_TAG:
            joint[base] = value
        elif tag == EAST_TAG:
            east[base] = value
        elif tag == WEST_TAG:
            west[base] = value
        else:
            raise ValueError(f"Parameter '{name}' is missing an explicit _joint/_east/_west tag.")

    return joint, east, west


def merge_limb_parameter_dict(
    joint: Mapping[str, jnp.ndarray],
    limb_specific: Mapping[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Merge explicit joint and limb dictionaries into the 1D kernel parameter shape."""
    out = dict(joint)
    out.update(limb_specific)
    return out


def get_limb_value(param_map: Mapping[str, object], base_name: str, limb: str) -> object:
    """Return a tagged value for the given limb, falling back to _joint only."""
    limb_key = f"{base_name}_{limb}"
    joint_key = f"{base_name}_{JOINT_TAG}"
    if limb_key in param_map:
        return param_map[limb_key]
    if joint_key in param_map:
        return param_map[joint_key]
    raise KeyError(f"Missing parameter '{limb_key}' or '{joint_key}'")


def jitter_param_name(rt_scheme: str | None) -> str:
    """Return the jitter parameter name for the requested RT scheme."""
    return "c_joint" if str(rt_scheme).lower() == "transit_2d" else "c"


def parse_offset_group_name(param_name: str, rt_scheme: str | None) -> str | None:
    """Return the observation offset group name encoded in a parameter name."""
    name = str(param_name)
    if str(rt_scheme).lower() == "transit_2d":
        base, tag = split_limb_tag(name)
        if tag != JOINT_TAG:
            return None
        if not base.startswith("offset_"):
            return None
        return base[7:]

    if name.startswith("offset_"):
        return name[7:]
    return None
