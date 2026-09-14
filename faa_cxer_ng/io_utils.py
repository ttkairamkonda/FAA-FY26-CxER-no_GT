"""Shared, dependency-free I/O helpers. Deliberately not shared with the
FAA_CxER package — this pipeline is meant to stand alone."""

from __future__ import annotations

import json
import os
from typing import Any, Union

PathLike = Union[str, os.PathLike]


def load_json(path: PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json_atomic(path: PathLike, obj: Any, indent: int = 2) -> None:
    """Write JSON via temp-file + os.replace so a crash mid-write never
    leaves a truncated/corrupt file behind."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp_path, path)
