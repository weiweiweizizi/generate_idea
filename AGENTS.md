# Repository Guidelines

## Project Structure & Module Organization
This repository is a facial-motion research workspace, not a packaged library. Core analysis code lives in `scripts/`, with decomposition pipelines such as `svd_*`, `dmd_*`, `grassmann_*`, and `nmf_*`. The `scripts/lq/` subtree contains the learning-based prototype (`train.py`, datasets, model code, and utilities). Validation experiments live in `scripts/val_codebook/`. Research outputs and large generated artifacts are stored under `data/`, while narrative context lives in `IDEA_REPORT.md`, `IDEA_EXPERIMENTS.md`, `RESEARCH_PROGRESS.md`, `docs/`, `literature_notes/`, and `papers/`.

## Build, Test, and Development Commands
Use the existing Conda environment noted in `CLAUDE.md`: `conda activate openmmlab`.

Typical commands:

```bash
python scripts/svd_single_patient.py
python scripts/dmd_blendshape_correlation.py
python scripts/val_codebook/sweep.py --run
python scripts/lq/train.py --data_roots=data/win10-step10/IMR,data/win10-step10/TT
```

These run single analyses, correlation experiments, validation sweeps, and LQ model training. There is no root-level build system or Makefile in this checkout.

## Coding Style & Naming Conventions
Follow the existing Python style in `scripts/`: 4-space indentation, module-level constants in `UPPER_SNAKE_CASE`, functions in `snake_case`, and descriptive filenames such as `svd_multi_patient_win5.py`. Keep scripts focused on one experiment. Prefer `pathlib.Path` for filesystem paths and keep dataset/window names explicit in output directories. Avoid committing generated caches such as `__pycache__/`.

## Testing Guidelines
There is no automated unit-test suite yet. Validate changes by rerunning the affected script on a small, representative dataset slice and checking that expected outputs appear under `data/.../*results*` or `outputs/`. For `scripts/val_codebook/`, use `python scripts/val_codebook/sweep.py --summary` after runs to confirm aggregate metrics. Document any manual validation in the relevant research note or checklist under `docs/`.

## Commit & Pull Request Guidelines
Git history is not available in this workspace, so no repository-specific commit convention can be inferred. Use short imperative commit subjects, optionally scoped by area, for example: `scripts: refine win5 DMD correlation output`. Pull requests should state the research goal, list touched datasets/scripts, identify newly generated output paths, and include plots or screenshots when a change affects figures or reported metrics.

## Data & Output Hygiene
Treat `data/` as source data plus reproducible outputs: do not rename dataset folders casually, and keep new result directories method-specific and window-specific. If a script depends on absolute local paths, replace them with repository-relative paths before merging.
