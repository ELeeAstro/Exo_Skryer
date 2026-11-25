"""
limb_asymmetry.py
=================

Helpers for the transit_1_5d explicit _joint/_east/_west parameter mode.
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


DELTA_PREFIX = "delta_"


def _delta_target(base: str, tag: str | None, tagged: Mapping[str, set]) -> str | None:
    """For ``delta_<name>_<limb>``: the parameter it offsets (<name>), if that is a valid limb offset.

    ``delta_<name>_west`` sets <name>_west = <name>_east + delta (and vice versa for _east). It is
    valid when <name> is given for the other limb only and is not joint.
    """
    if not base.startswith(DELTA_PREFIX) or tag not in (EAST_TAG, WEST_TAG):
        return None
    target = base[len(DELTA_PREFIX):]
    other = WEST_TAG if tag == EAST_TAG else EAST_TAG
    tags = tagged.get(target, set())
    return target if tags == {other} else None


def validate_limb_parameter_names(names: Iterable[str]) -> set[str]:
    """Validate explicit transit_1_5d tagging and return east/west duplicated bases.

    A limb may instead be given as an offset from the other limb: ``delta_<name>_west``
    (``<name>_west = <name>_east + delta_<name>_west``), or the mirror with _east.
    """
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
            "transit_1_5d requires every YAML parameter to use an explicit suffix: "
            f"_joint, _east, or _west. Untagged parameter(s): {joined}"
        )

    mixed_joint = sorted(
        base for base, tags in tagged.items() if JOINT_TAG in tags and (EAST_TAG in tags or WEST_TAG in tags)
    )
    if mixed_joint:
        joined = ", ".join(mixed_joint)
        raise ValueError(
            "transit_1_5d parameters cannot mix _joint with _east/_west for the same base name: "
            f"{joined}"
        )

    # Limb offsets: delta_<name>_<limb> supplies the missing partner of <name>_<other limb>.
    delta_of = {}
    for base, tags in tagged.items():
        for tag in tags:
            target = _delta_target(base, tag, tagged)
            if target is not None:
                delta_of[base] = target
    if base_bad := sorted(b for b, t in tagged.items()
                          if b.startswith(DELTA_PREFIX) and b not in delta_of and t <= {EAST_TAG, WEST_TAG}
                          and b[len(DELTA_PREFIX):] in tagged):
        raise ValueError(
            "transit_1_5d limb offsets delta_<name>_<limb> need <name> given for the other limb only "
            f"(not joint, not both limbs). Invalid: {', '.join(base_bad)}"
        )
    offset_targets = set(delta_of.values())

    missing_pairs = sorted(
        base for base, tags in tagged.items()
        if base not in delta_of and base not in offset_targets
        and tags in ({EAST_TAG}, {WEST_TAG}, {EAST_TAG, JOINT_TAG}, {WEST_TAG, JOINT_TAG})
    )
    if missing_pairs:
        joined = ", ".join(missing_pairs)
        raise ValueError(
            "transit_1_5d requires both _east and _west entries for each limb-specific base name. "
            f"Missing partner(s) for: {joined}"
        )

    duplicated = {base for base, tags in tagged.items() if tags == {EAST_TAG, WEST_TAG}}
    return duplicated


def split_limb_parameter_dict(
    params: Mapping[str, jnp.ndarray],
) -> tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Split explicit-tag transit_1_5d params into joint/east/west dictionaries."""
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

    # Limb offsets: delta_<name> on one limb -> <name> = other limb's <name> + delta.
    for this, other in ((east, west), (west, east)):
        for key in [k for k in this if k.startswith(DELTA_PREFIX)]:
            target = key[len(DELTA_PREFIX):]
            if target in other and target not in this:
                this[target] = other[target] + this.pop(key)

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
    return "c_joint" if str(rt_scheme).lower() == "transit_1_5d" else "c"


def parse_offset_group_name(param_name: str, rt_scheme: str | None) -> str | None:
    """Return the observation offset group name encoded in a parameter name."""
    name = str(param_name)
    if str(rt_scheme).lower() == "transit_1_5d":
        base, tag = split_limb_tag(name)
        if tag != JOINT_TAG:
            return None
        if not base.startswith("offset_"):
            return None
        return base[7:]

    if name.startswith("offset_"):
        return name[7:]
    return None
