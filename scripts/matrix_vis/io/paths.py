from __future__ import annotations

from pathlib import Path


def resolve_input_path(config_path: Path, candidate: Path) -> Path:
    if candidate.is_absolute():
        return candidate

    config_relative = (config_path.parent / candidate).resolve()
    if config_relative.exists():
        return config_relative
    return candidate.resolve()


def resolve_output_path(candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else candidate.resolve()
