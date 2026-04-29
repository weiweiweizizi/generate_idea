#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import sys

import fire

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.matrix_vis.pipelines.inspect import inspect_axis_config


def inspect(config: str) -> dict:
    return inspect_axis_config(config)


if __name__ == "__main__":
    fire.Fire({"inspect": inspect})
