# Patient-level t-SNE Report

- patient_profiles_csv: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/patient_profile_summary/patient_activation_profiles_tt.csv`
- output_root: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/tt/all_basis`
- n_patients: `42`
- feature_families: `usage, activation, combined`
- excluded_basis_indices: `[]`
- plots_per_family: `10`
- total_plots: `30`

## OOF Mode

- subject_fold_assignments_csv: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/subject_fold_assignments.csv`
- num_folds: `5`
- num_patients_oof: `42`
- patients_per_fold: `{'0': 8, '1': 9, '2': 9, '3': 6, '4': 10}`

## Families

### usage

- feature_count: `11`
- excluded_basis_indices: `[]`
- perplexity_used: `13.0`
- random_state: `42`
- 2D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/tt/all_basis/usage/tsne_2d/usage_tsne_2d_embeddings.csv`
- 3D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/tt/all_basis/usage/tsne_3d/usage_tsne_3d_embeddings.csv`
- 3D combined GIF: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/tt/all_basis/usage/tsne_3d/usage_tsne_3d_combined.gif`

### activation

- feature_count: `11`
- excluded_basis_indices: `[]`
- perplexity_used: `13.0`
- random_state: `42`
- 2D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/tt/all_basis/activation/tsne_2d/activation_tsne_2d_embeddings.csv`
- 3D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/tt/all_basis/activation/tsne_3d/activation_tsne_3d_embeddings.csv`
- 3D combined GIF: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/tt/all_basis/activation/tsne_3d/activation_tsne_3d_combined.gif`

### combined

- feature_count: `22`
- excluded_basis_indices: `[]`
- perplexity_used: `13.0`
- random_state: `42`
- 2D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/tt/all_basis/combined/tsne_2d/combined_tsne_2d_embeddings.csv`
- 3D embeddings: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/tt/all_basis/combined/tsne_3d/combined_tsne_3d_embeddings.csv`
- 3D combined GIF: `/home/weizilin/generate_idea/outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_kfold_oof/tt/all_basis/combined/tsne_3d/combined_tsne_3d_combined.gif`
