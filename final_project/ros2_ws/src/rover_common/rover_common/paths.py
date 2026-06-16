"""Path discovery helpers for vehicle-side imports.

The Jetson scripts need modules from the repository root (`calibration`,
`control`) and from `final_project`. Prefer explicit environment variables, but
fall back to walking up from source files when running with `--symlink-install`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _valid_repo_root(path: Path) -> bool:
    return (path / "calibration").is_dir() and (path / "control").is_dir() and (
        path / "final_project"
    ).is_dir()


def find_repo_root(start: str | Path | None = None) -> Path:
    env = os.environ.get("AUE4040_REPO_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if _valid_repo_root(root):
            return root
        raise RuntimeError(f"AUE4040_REPO_ROOT does not look like this repo: {root}")

    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for parent in (here, *here.parents):
        if _valid_repo_root(parent):
            return parent

    raise RuntimeError(
        "could not locate AUE4040 repo root. Set AUE4040_REPO_ROOT=/path/to/AUE4040"
    )


def find_final_project_root(start: str | Path | None = None) -> Path:
    env = os.environ.get("AUE4040_FINAL_PROJECT_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if (root / "data_pipeline" / "extract_labels.py").exists():
            return root
        raise RuntimeError(
            f"AUE4040_FINAL_PROJECT_ROOT does not look like final_project: {root}"
        )
    return find_repo_root(start) / "final_project"


def ensure_repo_on_path(start: str | Path | None = None) -> Path:
    root = find_repo_root(start)
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def ensure_final_project_on_path(start: str | Path | None = None) -> Path:
    root = find_final_project_root(start)
    for path in (root, root / "training"):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)
    return root
