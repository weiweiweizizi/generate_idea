from __future__ import annotations

from pathlib import Path

import fire

from disentangleNet.analysis.exporters import export_patient as run_export_patient


def export(
    checkpoint_path: str,
    subject: str = "844697",
    data_roots: str | None = None,
    output_dir: str | None = None,
    batch_size: int = 8,
):
    return run_export_patient(
        checkpoint_path=str(Path(checkpoint_path).expanduser()),
        subject=subject,
        data_roots=data_roots,
        output_dir=output_dir,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    fire.Fire({"export": export})
