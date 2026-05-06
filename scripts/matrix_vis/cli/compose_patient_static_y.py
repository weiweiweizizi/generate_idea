#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import sys

import fire

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.pipelines.compose_patient_static_y import compose_patient_static_y


def compose(
    sequence_dir: str,
    output_dir: str | None = None,
    mesh_source: str | None = None,
) -> dict:
    kwargs = {"sequence_dir": sequence_dir, "output_dir": output_dir}
    if mesh_source is not None:
        kwargs["mesh_source"] = mesh_source
    return compose_patient_static_y(**kwargs)


if __name__ == "__main__":
    fire.Fire({"compose": compose})
