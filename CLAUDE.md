# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Repository Status

**facial motion decomposition research project** using matrix factorization (NMF, SVD, Tucker/CP decomposition) on keypoint distance matrices for facial paralysis grading (面瘫分级).

## Research Direction

**Core Problem**: Decompose facial keypoint distance matrices into interpretable sub-motion bases for identity-motion disentanglement.

**Key Innovation**: Using **difference distance matrices** (ΔD = D_t - D_{t-1}) to capture motion changes between frames.

## Method Evolution

1. **NMF → Failed**: Negative values in ΔD; raw distance matrix D has identity dominance
2. **SVD → Success**: ΔD + SVD captures motion semantics (mouth-dominant PC1)
3. **Grassmann Validation → Confirmed**: Shared bases are "truly shared" not "computationally forced"
4. **Tucker → Not suitable**: Spatial factor cannot reshape to 341×341 heatmap
5. **DMD → Current**: Better temporal dynamics; win5-step5 enables reliable correlation (25-130 windows vs ~5 for win20)
6. **Next**: PARAFAC/CP decomposition, blendshape weak supervision

## Data

### Datasets

| Dataset | Path | Patients | Window | Notes |
|---------|------|----------|--------|-------|
| IMR | `data/*/IMR/` | ~227 | various | Lab-controlled environment |
| TT | `data/*/TT/` | ~42 | various | Hospital-collected, more heterogeneous |

### Window Configurations

| Config | Path | Use Case |
|--------|------|----------|
| win20-step20 | `data/win20-step20/` | Original SVD decomposition |
| win10-step10 | `data/win10-step10/` | Intermediate window |
| win5-step5 | `data/win5-step5/` | DMD + reliable correlation analysis |

- **Matrix**: 341×341 (pairwise distances)
- **⚠️ Window count issue**: win20-step20 IMR patients have only ~5 windows, making Pearson correlation unreliable. win5-step5 provides 25-130+ windows for robust correlation.

### Blendshape Data
- `data/blendshape/`: Action unit (AU) annotations, 52 AU types
  - Located in `IMR/blendshape/` and `TT/blendshape/` subdirectories

### Landmark Regions (341 points, grouped ordering)

```
boundaries = [18, 38, 100, 130, 140, 188, 233, 307, 331, 341]
regions = ["forehead", "eyebrow", "eyehole", "eye_contour",
           "eye_iris", "nose", "around_mouth", "mouth", "cheek", "jaw"]
```

## Scripts

### Decomposition Methods
| Script | Purpose |
|--------|---------|
| `scripts/pilot_feasibility/svd/svd_single_patient.py` | Single-patient SVD ΔD decomposition |
| `scripts/pilot_feasibility/svd/svd_multi_patient.py` | Multi-patient joint SVD |
| `scripts/pilot_feasibility/svd/svd_single_patient_raw.py` | RAW vs DIFF comparison |
| `scripts/pilot_feasibility/svd/svd_multi_patient_win5.py` | Multi-patient SVD on win5-step5 data |
| `scripts/pilot_feasibility/nmf/nmf_baseline_x_y.py` | NMF baseline (failed) |
| `scripts/pilot_feasibility/tucker/tucker_multi_patient.py` | Tucker decomposition attempt |

### DMD (Dynamic Mode Decomposition)
| Script | Purpose |
|--------|---------|
| `scripts/pilot_feasibility/dmd/dmd_single_patient.py` | Single-patient DMD with spatial mode heatmaps |
| `scripts/pilot_feasibility/dmd/dmd_multi_patient.py` | Multi-patient joint DMD |
| `scripts/pilot_feasibility/dmd/dmd_blendshape_correlation.py` | DMD mode vs blendshape correlation |

### Blendshape Validation
| Script | Purpose |
|--------|---------|
| `scripts/pilot_feasibility/blendshape/blendshape_correlation_analysis.py` | SVD time coeff vs blendshape correlation (win20) |
| `scripts/pilot_feasibility/blendshape/blendshape_correlation_analysis_win5.py` | Blendshape correlation with win5 data |
| `scripts/pilot_feasibility/blendshape/blendshape_correlation_analysis_win10.py` | Blendshape correlation with win10 data |

### Grassmann Analysis
| Script | Purpose |
|--------|---------|
| `scripts/pilot_feasibility/grassmann/grassmann_analysis.py` | Early Grassmann analysis |
| `scripts/pilot_feasibility/grassmann/grassmann_cross_analysis.py` | Cross-dataset Grassmann validation |

**Conda env**: `conda activate openmmlab`

## Related Projects

### corelation-lm (`/home/weizilin/code_reproduction/corelation-lm/`)

Facial landmark similarity analysis toolkit.

- `project/utils/landmark_ordering.py` - Landmark ordering (grouped/interleaved/left_minus_right)
- `project/similarity/heatmap_renderer.py` - Heatmap rendering with region dividers
- `project/configs/extractors.yaml` - Facial region and symmetric point config
- `project/similarity/distance_metrics.py` - Distance metrics (Euclidean, DTW, correlation)

### corelation-classify (`/home/weizilin/code_reproduction/corelation-classsify/`)

Classification analysis scripts based on keypoint similarity matrices.

- `scripts/precompute_matrices.py` - Precompute distance matrices
- `scripts/step1_1_explore.py` - Difference heatmap visualization

## Key Experimental Results

### SVD ΔD: Motion Semantics Confirmed
- PC1 dominant region = **mouth** (all patients, both datasets)
- RAW matrix: PC1 dominant = eyehole (identity)
- **ΔD is necessary** for capturing motion

### Multi-Patient Joint SVD
| Dataset | X PC1 | X PC2 | PC1 dominant |
|---------|-------|-------|--------------|
| IMR (227p) | 64.6% | 25.1% | mouth |
| TT (42p) | 47.0% | 27.2% | mouth |

### Grassmann Validation (2026-03-30)
| Comparison | X PC1 | Y PC1 |
|------------|-------|-------|
| IMR_single vs IMR_joint | 12.9° | 10.0° |
| TT_single vs TT_joint | 14.8° | 8.9° |
| **TT_joint vs IMR_joint** | **13.5°** | **7.1°** |

**Conclusion**: Shared bases are "truly shared" — patients align better with their own dataset's joint basis (~13°) than cross-dataset (~20°).

### DMD Blendshape Correlation (win5-step5, 2026-04-01)
| DMD Mode | Top Blendshape | r |
|----------|----------------|-----|
| X Mode1 | jawForward | 0.59 |
| Y Mode1 | cheekPuff | 0.60 |
| Y Mode2 | cheekPuff | 0.64 |

**Note**: Using win5-step5 (25-130 windows) instead of win20 (~5 windows) for statistically reliable correlation.

## Idea Report

- `IDEA_REPORT.md`: Idea landscape, ranked recommendations, next steps
- `IDEA_EXPERIMENTS.md`: Detailed experiment logs with script/folder mapping
- `RESEARCH_PROGRESS.md`: Research timeline and key findings
- `literature_notes/`: Paper summaries (Hallac 2017/2024 most relevant)

**Most relevant PubMed papers**:
- Hallac 2017 (PMID 28011182): Frame-to-frame distance + Procrustes
- Hallac 2024 (PMID 39476531): Facial palsy 3D topography

## Notes

- WebFetch permitted for github.com, arxiv.org, pubmed.ncbi.nlm.nih.gov
- ArXiv rate-limited; use OpenAlex API as alternative
