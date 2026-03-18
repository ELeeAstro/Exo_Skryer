"""
read_yaml.py
============
"""

import collections
import collections.abc
from pathlib import Path
from types import SimpleNamespace
import yaml

__all__ = ['read_yaml']


# PyYAML 3.x still references collections.Hashable, which was removed in Python 3.12.
if not hasattr(collections, "Hashable"):
    collections.Hashable = collections.abc.Hashable


def _to_ns(x):
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    return x


def read_yaml(path: Path):
    '''
      Input: path to YAML configuration file.
      Output: Dot accessible namespace cfg with all YAML parameters
    '''

    cfg = yaml.safe_load(path.read_text())

    return _to_ns(cfg)
