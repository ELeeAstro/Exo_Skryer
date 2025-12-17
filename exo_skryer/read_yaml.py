"""
read_yaml.py
============

Overview:
    Reads the YAML configuration file and names a dot namespace for easy accessability.
"""

from pathlib import Path
from types import SimpleNamespace
import yaml

__all__ = ['read_yaml']


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
