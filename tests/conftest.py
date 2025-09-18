"""Pytest configuration for AutoKaggler tests."""
from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure the project source directory is importable during tests."""
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
