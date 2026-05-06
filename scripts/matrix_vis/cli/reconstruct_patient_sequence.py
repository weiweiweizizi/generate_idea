#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import sys

import fire

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.pipelines.patient_sequence import run_patient_sequence


def reconstruct(
    patient_bundle_path: str,
    output_dir: str | None = None,
    mesh_source: str | None = None,
) -> dict:
    kwargs = {"patient_bundle_path": patient_bundle_path, "output_dir": output_dir}
    if mesh_source is not None:
        kwargs["mesh_source"] = mesh_source
    return run_patient_sequence(**kwargs)


if __name__ == "__main__":
    fire.Fire({"reconstruct": reconstruct})
