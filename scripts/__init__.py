"""Script namespace for artifact entry points."""

from __future__ import annotations

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent / "analysis"))
__path__.append(str(Path(__file__).resolve().parent / "pipeline"))
