"""Filesystem locations, read from the environment on every call so tests can override them."""

import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "data")).resolve()


def runs_dir() -> Path:
    return Path(os.getenv("RUNS_DIR", "runs")).resolve()


def competition_data_dir(slug: str) -> Path:
    return data_dir() / slug


def competition_runs_dir(slug: str) -> Path:
    return runs_dir() / slug
