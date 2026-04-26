# V31 Full Post-hoc K-Fold Report

- checkpoint: `outputs/disentangleNet/v31_current_verify/best.pt`
- output_dir: `outputs/disentangleNet/v31_current_verify/kfold_report`
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
| dataset_from_coeff | 293 | 1 | 2 | 0.7781569965870307 | 0.5 | 0.43761996161228406 |
| dataset_from_free_rep | 293 | 32 | 2 | 0.7986348122866894 | 0.6451417004048583 | 0.662712426589663 |
| dataset_from_private_rep | 293 | 32 | 2 | 0.8839590443686007 | 0.8099527665317139 | 0.8240178066704352 |
| dataset_from_side_rep | 293 | 3 | 2 | 0.7713310580204779 | 0.5286099865047234 | 0.5098744163982722 |
| dataset_from_usage | 293 | 3 | 2 | 0.7781569965870307 | 0.5 | 0.43761996161228406 |
| dataset_from_usage_coeff | 293 | 4 | 2 | 0.7508532423208191 | 0.4879554655870445 | 0.44190476190476186 |
| side_from_coeff | 293 | 1 | 3 | 0.9351535836177475 | 0.9314352687846664 | 0.933911221776896 |
| side_from_free_rep | 293 | 32 | 3 | 0.4709897610921502 | 0.43986374106856035 | 0.39658859963993515 |
| side_from_side_rep | 293 | 3 | 3 | 0.9283276450511946 | 0.9247012620506596 | 0.9267017025118615 |
| side_from_usage | 293 | 3 | 3 | 0.9283276450511946 | 0.9289596277548084 | 0.928786205203731 |
| side_from_usage_coeff | 293 | 4 | 3 | 0.9419795221843004 | 0.9416996404948211 | 0.9416051214127067 |

## Task Artifacts

### side_from_side_rep

- accuracy: `0.9283`
- balanced_accuracy: `0.9247`
- macro_f1: `0.9267`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/side_from_side_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/side_from_side_rep_confusion.png`

### side_from_free_rep

- accuracy: `0.4710`
- balanced_accuracy: `0.4399`
- macro_f1: `0.3966`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/side_from_free_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/side_from_free_rep_confusion.png`

### dataset_from_side_rep

- accuracy: `0.7713`
- balanced_accuracy: `0.5286`
- macro_f1: `0.5099`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_side_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_side_rep_confusion.png`

### dataset_from_free_rep

- accuracy: `0.7986`
- balanced_accuracy: `0.6451`
- macro_f1: `0.6627`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_free_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_free_rep_confusion.png`

### dataset_from_private_rep

- accuracy: `0.8840`
- balanced_accuracy: `0.8100`
- macro_f1: `0.8240`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_private_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_private_rep_confusion.png`

### side_from_usage

- accuracy: `0.9283`
- balanced_accuracy: `0.9290`
- macro_f1: `0.9288`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/side_from_usage_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/side_from_usage_confusion.png`

### side_from_coeff

- accuracy: `0.9352`
- balanced_accuracy: `0.9314`
- macro_f1: `0.9339`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/side_from_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/side_from_coeff_confusion.png`

### side_from_usage_coeff

- accuracy: `0.9420`
- balanced_accuracy: `0.9417`
- macro_f1: `0.9416`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/side_from_usage_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/side_from_usage_coeff_confusion.png`

### dataset_from_usage

- accuracy: `0.7782`
- balanced_accuracy: `0.5000`
- macro_f1: `0.4376`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_usage_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_usage_confusion.png`

### dataset_from_coeff

- accuracy: `0.7782`
- balanced_accuracy: `0.5000`
- macro_f1: `0.4376`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_coeff_confusion.png`

### dataset_from_usage_coeff

- accuracy: `0.7509`
- balanced_accuracy: `0.4880`
- macro_f1: `0.4419`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_usage_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report/dataset_from_usage_coeff_confusion.png`

