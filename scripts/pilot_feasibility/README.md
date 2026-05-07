## Pilot Feasibility Scripts

Early feasibility experiments are grouped by method:

- `svd/`
- `dmd/`
- `grassmann/`
- `blendshape/`
- `nmf/`
- `tucker/`

Conventions:

- Source datasets stay under `data/`
- Experiment outputs go under `outputs/pilot_feasibility/<method>/<window-setting>/`
- Dataset-like directories that remain in `data/` include:
  - `data/win20-step20/TT-SVD`
  - `data/win20-step20/IMR-SVD`
  - `data/win20-step20/cal_diff`
  - `data/win20-step20/cal_diff_mouth-only`
