# V31 Full Post-hoc K-Fold Report

- checkpoint: `outputs/disentangleNet/v31_internal_compact_verify_e50/best.pt`
- output_dir: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report`
- total_groups: `293`
- total_subjects: `267`
- requested_splits: `5`
- resolved_splits: `5`
- subject_stratification: `joint_side_dataset`

## Fold Summary

| fold | num_subjects | num_groups | side_counts | dataset_counts |
| --- | --- | --- | --- | --- |
| 0 | 54 | 59 | {"Left": 17, "Normal": 22, "Right": 20} | {"IMR": 46, "TT": 13} |
| 1 | 54 | 58 | {"Left": 17, "Normal": 21, "Right": 20} | {"IMR": 45, "TT": 13} |
| 2 | 54 | 58 | {"Left": 16, "Normal": 22, "Right": 20} | {"IMR": 45, "TT": 13} |
| 3 | 51 | 59 | {"Left": 17, "Normal": 23, "Right": 19} | {"IMR": 46, "TT": 13} |
| 4 | 54 | 59 | {"Left": 16, "Normal": 23, "Right": 20} | {"IMR": 46, "TT": 13} |

## Probe Metrics

| task_name | num_groups | num_features | num_classes | accuracy | balanced_accuracy | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| dataset_from_coeff | 293 | 1 | 2 | 0.7815699658703071 | 0.5076923076923077 | 0.4536130536130536 |
| dataset_from_free_rep | 293 | 32 | 2 | 0.8054607508532423 | 0.6935222672064778 | 0.7031724873367102 |
| dataset_from_private_rep | 293 | 32 | 2 | 0.8907849829351536 | 0.8253373819163293 | 0.8363357073034492 |
| dataset_from_side_rep | 293 | 3 | 2 | 0.7781569965870307 | 0.5 | 0.43761996161228406 |
| dataset_from_usage | 293 | 3 | 2 | 0.7781569965870307 | 0.5 | 0.43761996161228406 |
| dataset_from_usage_coeff | 293 | 4 | 2 | 0.7508532423208191 | 0.4989541160593792 | 0.4659825730906549 |
| side_from_coeff | 293 | 1 | 3 | 0.9283276450511946 | 0.9247802018886356 | 0.9271679030868829 |
| side_from_free_rep | 293 | 32 | 3 | 0.4709897610921502 | 0.4372674975084614 | 0.3737054143443032 |
| side_from_side_rep | 293 | 3 | 3 | 0.931740614334471 | 0.9287173263076878 | 0.9300872788139646 |
| side_from_usage | 293 | 3 | 3 | 0.9283276450511946 | 0.9283105668647836 | 0.9276011719278366 |
| side_from_usage_coeff | 293 | 4 | 3 | 0.9283276450511946 | 0.9283105668647836 | 0.9272054640660654 |

## Task Artifacts

### side_from_side_rep

- accuracy: `0.9317`
- balanced_accuracy: `0.9287`
- macro_f1: `0.9301`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/side_from_side_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/side_from_side_rep_confusion.png`

### side_from_free_rep

- accuracy: `0.4710`
- balanced_accuracy: `0.4373`
- macro_f1: `0.3737`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/side_from_free_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/side_from_free_rep_confusion.png`

### dataset_from_side_rep

- accuracy: `0.7782`
- balanced_accuracy: `0.5000`
- macro_f1: `0.4376`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_side_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_side_rep_confusion.png`

### dataset_from_free_rep

- accuracy: `0.8055`
- balanced_accuracy: `0.6935`
- macro_f1: `0.7032`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_free_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_free_rep_confusion.png`

### dataset_from_private_rep

- accuracy: `0.8908`
- balanced_accuracy: `0.8253`
- macro_f1: `0.8363`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_private_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_private_rep_confusion.png`

### side_from_usage

- accuracy: `0.9283`
- balanced_accuracy: `0.9283`
- macro_f1: `0.9276`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/side_from_usage_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/side_from_usage_confusion.png`

### side_from_coeff

- accuracy: `0.9283`
- balanced_accuracy: `0.9248`
- macro_f1: `0.9272`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/side_from_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/side_from_coeff_confusion.png`

### side_from_usage_coeff

- accuracy: `0.9283`
- balanced_accuracy: `0.9283`
- macro_f1: `0.9272`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/side_from_usage_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/side_from_usage_coeff_confusion.png`

### dataset_from_usage

- accuracy: `0.7782`
- balanced_accuracy: `0.5000`
- macro_f1: `0.4376`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_usage_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_usage_confusion.png`

### dataset_from_coeff

- accuracy: `0.7816`
- balanced_accuracy: `0.5077`
- macro_f1: `0.4536`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_coeff_confusion.png`

### dataset_from_usage_coeff

- accuracy: `0.7509`
- balanced_accuracy: `0.4990`
- macro_f1: `0.4660`
- confusion_csv: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_usage_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_internal_compact_verify_e50/kfold_report/dataset_from_usage_coeff_confusion.png`

