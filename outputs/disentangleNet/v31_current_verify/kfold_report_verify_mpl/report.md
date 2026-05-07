# V31 Full Post-hoc K-Fold Report

- checkpoint: `outputs/disentangleNet/v31_current_verify/best.pt`
- output_dir: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl`
- total_groups: `293`
- total_subjects: `267`
- requested_splits: `5`
- resolved_splits: `5`
- subject_stratification: `joint_side_dataset`

## Fold Summary

| fold | num_subjects | num_groups | side_counts | label5_counts | dataset_counts |
| --- | --- | --- | --- | --- | --- |
| 0 | 54 | 59 | {"Left": 17, "Normal": 22, "Right": 20} | {"0": 8, "1": 9, "2": 22, "3": 1, "4": 19} | {"IMR": 46, "TT": 13} |
| 1 | 54 | 58 | {"Left": 17, "Normal": 21, "Right": 20} | {"0": 6, "1": 11, "2": 21, "3": 9, "4": 11} | {"IMR": 45, "TT": 13} |
| 2 | 54 | 58 | {"Left": 16, "Normal": 22, "Right": 20} | {"0": 9, "1": 7, "2": 22, "3": 6, "4": 14} | {"IMR": 45, "TT": 13} |
| 3 | 51 | 59 | {"Left": 17, "Normal": 23, "Right": 19} | {"0": 7, "1": 10, "2": 23, "3": 8, "4": 11} | {"IMR": 46, "TT": 13} |
| 4 | 54 | 59 | {"Left": 16, "Normal": 23, "Right": 20} | {"0": 9, "1": 7, "2": 23, "3": 10, "4": 10} | {"IMR": 46, "TT": 13} |

## Probe Metrics

| task_name | num_groups | num_features | num_classes | accuracy | balanced_accuracy | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| dataset_from_coeff | 293 | 1 | 2 | 0.7781569965870307 | 0.5 | 0.43761996161228406 |
| dataset_from_free_rep | 293 | 32 | 2 | 0.8156996587030717 | 0.639608636977058 | 0.6636479591836735 |
| dataset_from_private_rep | 293 | 32 | 2 | 0.9044368600682594 | 0.839608636977058 | 0.8550734878462407 |
| dataset_from_side_rep | 293 | 3 | 2 | 0.7781569965870307 | 0.5 | 0.43761996161228406 |
| dataset_from_usage | 293 | 3 | 2 | 0.7781569965870307 | 0.5 | 0.43761996161228406 |
| dataset_from_usage_coeff | 293 | 4 | 2 | 0.7781569965870307 | 0.5 | 0.43761996161228406 |
| label5_from_coeff | 293 | 1 | 5 | 0.7406143344709898 | 0.6217766017766018 | 0.5835277762079515 |
| label5_from_free_rep | 293 | 32 | 5 | 0.4402730375426621 | 0.2745114345114345 | 0.20797775195085508 |
| label5_from_side_rep | 293 | 3 | 5 | 0.7406143344709898 | 0.6234385434385434 | 0.5894773777558588 |
| label5_from_usage | 293 | 3 | 5 | 0.7167235494880546 | 0.5956133056133056 | 0.5424903642704029 |
| label5_from_usage_coeff | 293 | 4 | 5 | 0.7542662116040956 | 0.6418137718137717 | 0.6081643878811912 |
| side_from_coeff | 293 | 1 | 3 | 0.8907849829351536 | 0.8852258611294755 | 0.888461330665519 |
| side_from_free_rep | 293 | 32 | 3 | 0.48464163822525597 | 0.4502136309365225 | 0.3934371053306219 |
| side_from_side_rep | 293 | 3 | 3 | 0.9044368600682594 | 0.8978869340315123 | 0.9003102316195295 |
| side_from_usage | 293 | 3 | 3 | 0.9146757679180887 | 0.9115183091086706 | 0.9130610001899195 |
| side_from_usage_coeff | 293 | 4 | 3 | 0.9180887372013652 | 0.9148853124756738 | 0.9158117012642207 |

## Task Artifacts

### side_from_side_rep

- accuracy: `0.9044`
- balanced_accuracy: `0.8979`
- macro_f1: `0.9003`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/side_from_side_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/side_from_side_rep_confusion.png`

### side_from_free_rep

- accuracy: `0.4846`
- balanced_accuracy: `0.4502`
- macro_f1: `0.3934`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/side_from_free_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/side_from_free_rep_confusion.png`

### label5_from_side_rep

- accuracy: `0.7406`
- balanced_accuracy: `0.6234`
- macro_f1: `0.5895`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/label5_from_side_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/label5_from_side_rep_confusion.png`

### label5_from_free_rep

- accuracy: `0.4403`
- balanced_accuracy: `0.2745`
- macro_f1: `0.2080`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/label5_from_free_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/label5_from_free_rep_confusion.png`

### dataset_from_side_rep

- accuracy: `0.7782`
- balanced_accuracy: `0.5000`
- macro_f1: `0.4376`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_side_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_side_rep_confusion.png`

### dataset_from_free_rep

- accuracy: `0.8157`
- balanced_accuracy: `0.6396`
- macro_f1: `0.6636`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_free_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_free_rep_confusion.png`

### dataset_from_private_rep

- accuracy: `0.9044`
- balanced_accuracy: `0.8396`
- macro_f1: `0.8551`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_private_rep_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_private_rep_confusion.png`

### side_from_usage

- accuracy: `0.9147`
- balanced_accuracy: `0.9115`
- macro_f1: `0.9131`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/side_from_usage_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/side_from_usage_confusion.png`

### label5_from_usage

- accuracy: `0.7167`
- balanced_accuracy: `0.5956`
- macro_f1: `0.5425`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/label5_from_usage_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/label5_from_usage_confusion.png`

### side_from_coeff

- accuracy: `0.8908`
- balanced_accuracy: `0.8852`
- macro_f1: `0.8885`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/side_from_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/side_from_coeff_confusion.png`

### label5_from_coeff

- accuracy: `0.7406`
- balanced_accuracy: `0.6218`
- macro_f1: `0.5835`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/label5_from_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/label5_from_coeff_confusion.png`

### side_from_usage_coeff

- accuracy: `0.9181`
- balanced_accuracy: `0.9149`
- macro_f1: `0.9158`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/side_from_usage_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/side_from_usage_coeff_confusion.png`

### label5_from_usage_coeff

- accuracy: `0.7543`
- balanced_accuracy: `0.6418`
- macro_f1: `0.6082`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/label5_from_usage_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/label5_from_usage_coeff_confusion.png`

### dataset_from_usage

- accuracy: `0.7782`
- balanced_accuracy: `0.5000`
- macro_f1: `0.4376`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_usage_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_usage_confusion.png`

### dataset_from_coeff

- accuracy: `0.7782`
- balanced_accuracy: `0.5000`
- macro_f1: `0.4376`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_coeff_confusion.png`

### dataset_from_usage_coeff

- accuracy: `0.7782`
- balanced_accuracy: `0.5000`
- macro_f1: `0.4376`
- confusion_csv: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_usage_coeff_confusion.csv`
- confusion_png: `outputs/disentangleNet/v31_current_verify/kfold_report_verify_mpl/dataset_from_usage_coeff_confusion.png`

