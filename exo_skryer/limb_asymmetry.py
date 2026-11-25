"""
limb_asymmetry.py
=================

Helpers for tagged multi-column transmission parameters: every YAML parameter ends
in ``_joint`` (shared by all columns) or an anchor tag. The anchors are ``_east`` /
``_west`` for ``transit_1_5d`` (evening / morning limbs), ``_day`` / ``_night``
for ``transit_2d``, and ``_east_day`` / ``_west_day`` / ``_east_night`` /
``_west_night`` for ``transit_3d``. ``delta_<name>_<anchor>`` gives one anchor as an offset from the
other: ``<name>_<anchor> = <name>_<other> + delta``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import jax.numpy as jnp

__all__ = [
    "LIMB_TAGS",
    "JOINT_TAG",
    "EAST_TAG",
    "WEST_TAG",
    "DAY_TAG",
    "NIGHT_TAG",
    "EAST_DAY_TAG",
    "WEST_DAY_TAG",
    "EAST_NIGHT_TAG",
    "WEST_NIGHT_TAG",
    "ANCHOR_TAGS",
    "OFFSET_VECTOR_KEY",
    "anchors_for",
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
DAY_TAG = "day"
NIGHT_TAG = "night"
LIMB_TAGS: tuple[str, str, str] = (JOINT_TAG, EAST_TAG, WEST_TAG)
# Anchor tags of each tagged rt_scheme.
EAST_DAY_TAG, WEST_DAY_TAG = "east_day", "west_day"
EAST_NIGHT_TAG, WEST_NIGHT_TAG = "east_night", "west_night"
# Anchor tags of each tagged rt_scheme (offsets are resolved in this order).
ANCHOR_TAGS: Dict[str, tuple[str, ...]] = {
    "transit_1_5d": (EAST_TAG, WEST_TAG),
    "transit_2d": (DAY_TAG, NIGHT_TAG),
    "transit_3d": (EAST_DAY_TAG, WEST_DAY_TAG, EAST_NIGHT_TAG, WEST_NIGHT_TAG),
}
# delta_<name>_<anchor> = <name>_<anchor> - <name>_<reference>. With two anchors the
# reference is the other one; for 3D: west day from east day, each night from its day.
_DELTA_REFERENCE: Dict[str, tuple[str, ...]] = {
    EAST_DAY_TAG: (WEST_DAY_TAG,),
    WEST_DAY_TAG: (EAST_DAY_TAG,),
    EAST_NIGHT_TAG: (EAST_DAY_TAG,),
    WEST_NIGHT_TAG: (WEST_DAY_TAG,),
}
# Compound tags first so "_east_day" is not read as "_day".
_ALL_TAGS = (EAST_DAY_TAG, WEST_DAY_TAG, EAST_NIGHT_TAG, WEST_NIGHT_TAG,
             JOINT_TAG, EAST_TAG, WEST_TAG, DAY_TAG, NIGHT_TAG)
OFFSET_VECTOR_KEY = "__offset_values__"


def anchors_for(rt_scheme: str | None) -> tuple[str, ...] | None:
    """Anchor tags of a tagged rt_scheme (None for untagged schemes such as transit_1d)."""
    return ANCHOR_TAGS.get(str(rt_scheme).strip().lower())


def split_limb_tag(name: str) -> tuple[str, str | None]:
    """Return ``(base_name, tag)`` where tag is joint or an anchor tag (east, day, east_day, ...), else ``None``."""
    for tag in _ALL_TAGS:
        token = f"_{tag}"
        if name.endswith(token):
            return name[: -len(token)], tag
    return name, None


def is_internal_parameter_name(name: str) -> bool:
    """Return whether a runtime-only parameter name is exempt from tag validation."""
    return str(name).startswith("__")


DELTA_PREFIX = "delta_"


def _references(tag: str, anchors: Sequence[str]) -> tuple[str, ...]:
    """Anchors an offset ``delta_<name>_<tag>`` may be taken from."""
    if len(anchors) == 2:
        return tuple(t for t in anchors if t != tag)
    return _DELTA_REFERENCE.get(tag, ())


def _resolve_order(given: set, deltas: set, anchors: Sequence[str]) -> list | None:
    """(anchor, reference) pairs that fill ``deltas`` from ``given``; None if some stay unresolved."""
    resolved, order, todo = set(given), [], set(deltas)
    while todo:
        step = [(a, r) for a in anchors if a in todo for r in _references(a, anchors) if r in resolved][:1]
        if not step:
            return None
        a, r = step[0]
        order.append((a, r)); resolved.add(a); todo.discard(a)
    return order


def validate_limb_parameter_names(names: Iterable[str], anchors: Sequence[str] = (EAST_TAG, WEST_TAG)) -> set[str]:
    """Validate explicit _joint/<anchor> tagging and return the bases given for every anchor.

    An anchor may instead be given as an offset ``delta_<name>_<anchor>`` from its reference
    anchor (the other one for two anchors; for 3D: west day <- east day, night <- same limb's day).
    """
    anchors = tuple(anchors)
    label = "/".join(f"_{t}" for t in anchors)
    tagged: dict[str, set[str]] = {}
    untagged: set[str] = set()
    foreign: set[str] = set()

    for raw_name in names:
        name = str(raw_name)
        if is_internal_parameter_name(name):
            continue
        base, tag = split_limb_tag(name)
        if tag is None:
            untagged.add(base)
            continue
        if tag != JOINT_TAG and tag not in anchors:
            foreign.add(name)
            continue
        tagged.setdefault(base, set()).add(tag)

    if untagged:
        raise ValueError(
            f"Tagged transmission requires every YAML parameter to end in _joint or {label}. "
            f"Untagged parameter(s): {', '.join(sorted(untagged))}"
        )
    if foreign:
        raise ValueError(f"Parameters with tags of another geometry (expected _joint or {label}): "
                         f"{', '.join(sorted(foreign))}")

    mixed_joint = sorted(base for base, tags in tagged.items() if JOINT_TAG in tags and tags & set(anchors))
    if mixed_joint:
        raise ValueError(
            f"Parameters cannot mix _joint with {label} for the same base name: {', '.join(mixed_joint)}"
        )

    targets = {b for b in tagged if not b.startswith(DELTA_PREFIX)}
    bad_delta, missing = [], []
    for base, tags in tagged.items():
        if base.startswith(DELTA_PREFIX):
            target = base[len(DELTA_PREFIX):]
            if target not in targets or JOINT_TAG in tagged.get(target, set()) or tags & tagged.get(target, set()):
                bad_delta.append(base)
    for target in targets:
        tags = tagged[target]
        if JOINT_TAG in tags:
            continue
        deltas = tagged.get(DELTA_PREFIX + target, set())
        if tags | deltas != set(anchors) or _resolve_order(tags, deltas, anchors) is None:
            missing.append(target)
    if bad_delta:
        raise ValueError(
            "Offsets delta_<name>_<anchor> need <name> given (not joint, not for the same anchor) and "
            f"resolvable from its reference anchor. Invalid: {', '.join(sorted(bad_delta))}"
        )
    if missing:
        raise ValueError(
            f"Each anchor-specific base name needs an entry (or a delta_ offset) for every anchor ({label}). "
            f"Missing or unresolvable for: {', '.join(sorted(missing))}"
        )
    return {base for base, tags in tagged.items() if tags == set(anchors)}


def split_limb_parameter_dict(
    params: Mapping[str, jnp.ndarray],
    anchors: Sequence[str] = (EAST_TAG, WEST_TAG),
) -> tuple[Dict[str, jnp.ndarray], ...]:
    """Split tagged params into ``(joint, <anchor 1>, <anchor 2>, ...)`` dictionaries.

    For transit_1_5d (default anchors) this is ``(joint, east, west)``. Offsets
    ``delta_<name>_<anchor>`` are resolved to ``<name>`` for that anchor.
    """
    anchors = tuple(anchors)
    validate_limb_parameter_names(params.keys(), anchors)

    joint: Dict[str, jnp.ndarray] = {}
    per: Dict[str, Dict[str, jnp.ndarray]] = {t: {} for t in anchors}
    for name, value in params.items():
        if is_internal_parameter_name(name):
            joint[name] = value
            continue
        base, tag = split_limb_tag(str(name))
        if tag == JOINT_TAG:
            joint[base] = value
        elif tag in per:
            per[tag][base] = value
        else:
            raise ValueError(f"Parameter '{name}' is missing an explicit _joint/{'/'.join('_' + t for t in anchors)} tag.")

    targets = {k[len(DELTA_PREFIX):] for t in anchors for k in per[t] if k.startswith(DELTA_PREFIX)}
    for target in targets:
        given = {t for t in anchors if target in per[t]}
        deltas = {t for t in anchors if DELTA_PREFIX + target in per[t]}
        for a, r in _resolve_order(given, deltas, anchors):
            per[a][target] = per[r][target] + per[a].pop(DELTA_PREFIX + target)

    return (joint, *(per[t] for t in anchors))


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
    return "c_joint" if anchors_for(rt_scheme) else "c"


def parse_offset_group_name(param_name: str, rt_scheme: str | None) -> str | None:
    """Return the observation offset group name encoded in a parameter name."""
    name = str(param_name)
    if anchors_for(rt_scheme):
        base, tag = split_limb_tag(name)
        if tag != JOINT_TAG:
            return None
        if not base.startswith("offset_"):
            return None
        return base[7:]

    if name.startswith("offset_"):
        return name[7:]
    return None
