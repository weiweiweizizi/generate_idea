from __future__ import annotations

from pathlib import Path

import fire

from disentangleNet.analysis.exporters import export_basis as run_export_basis


def export(
    checkpoint_path: str,
    output_dir: str | None = None,
    save_heatmaps: bool = True,
):
    return run_export_basis(
        checkpoint_path=str(Path(checkpoint_path).expanduser()),
        output_dir=output_dir,
        save_heatmaps=save_heatmaps,
    )


if __name__ == "__main__":
    fire.Fire({"export": export})
