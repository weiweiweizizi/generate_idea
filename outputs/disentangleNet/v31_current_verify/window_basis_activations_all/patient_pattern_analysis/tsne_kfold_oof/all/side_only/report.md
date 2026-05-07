# Patient-level t-SNE Report

- patient_profiles_csv: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/patient_profile_summary/patient_activation_profiles.csv`
- output_root: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/all/side_only`
- n_patients: `267`
- feature_families: `usage, activation, combined`
- excluded_basis_indices: `[0, 1, 2, 3, 4, 5, 6, 7]`
- plots_per_family: `10`
- total_plots: `30`

## OOF Mode

- subject_fold_assignments_csv: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/subject_fold_assignments.csv`
- num_folds: `5`
- num_patients_oof: `267`
- patients_per_fold: `{'0': 54, '1': 54, '2': 54, '3': 51, '4': 54}`

## Families

### usage

- feature_count: `3`
- excluded_basis_indices: `[0, 1, 2, 3, 4, 5, 6, 7]`
- perplexity_used: `30.0`
- random_state: `42`
- 2D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/all/side_only/usage/tsne_2d/usage_tsne_2d_embeddings.csv`
- 3D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/all/side_only/usage/tsne_3d/usage_tsne_3d_embeddings.csv`
- 3D combined GIF: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/all/side_only/usage/tsne_3d/usage_tsne_3d_combined.gif`

### activation

- feature_count: `3`
- excluded_basis_indices: `[0, 1, 2, 3, 4, 5, 6, 7]`
- perplexity_used: `30.0`
- random_state: `42`
- 2D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/all/side_only/activation/tsne_2d/activation_tsne_2d_embeddings.csv`
- 3D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/all/side_only/activation/tsne_3d/activation_tsne_3d_embeddings.csv`
- 3D combined GIF: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/all/side_only/activation/tsne_3d/activation_tsne_3d_combined.gif`

### combined

- feature_count: `6`
- excluded_basis_indices: `[0, 1, 2, 3, 4, 5, 6, 7]`
- perplexity_used: `30.0`
- random_state: `42`
- 2D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/all/side_only/combined/tsne_2d/combined_tsne_2d_embeddings.csv`
- 3D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/all/side_only/combined/tsne_3d/combined_tsne_3d_embeddings.csv`
- 3D combined GIF: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/all/side_only/combined/tsne_3d/combined_tsne_3d_combined.gif`
