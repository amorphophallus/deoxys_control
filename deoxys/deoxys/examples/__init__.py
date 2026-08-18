"""Module entry points for Deoxys example programs."""

import os
import sys
from pathlib import Path


def configure_furniture_bench_paths():
    """Make the sibling robust/FurnitureBench checkout importable."""
    code_root = Path(__file__).resolve().parents[4]
    robust_root = Path(
        os.environ.get(
            "ROBUST_REARRANGEMENT_ROOT",
            code_root / "robust-rearrangement-custom",
        )
    ).expanduser()
    for path in (robust_root, robust_root / "furniture-bench"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
