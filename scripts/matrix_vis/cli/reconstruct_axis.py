#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import sys

import fire

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.pipelines.reconstruct import run_axis_reconstruction

def reconstruct(
    config: str,
    axis: str | None = None,
    output_dir: str | None = None,
) -> dict:
    return run_axis_reconstruction(config=config, axis=axis, output_dir=output_dir)


if __name__ == "__main__":
    fire.Fire({"reconstruct": reconstruct})
