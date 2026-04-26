# V31 Full Post-hoc K-Fold Report

- checkpoint: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/best.pt`
- output_dir: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report`
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
| dataset_from_free_rep | 293 | 32 | 2 | 0.8122866894197952 | 0.6594129554655871 | 0.6809226809226809 |
| dataset_from_private_rep | 293 | 32 | 2 | 0.8873720136518771 | 0.8286437246963563 | 0.8341310277391796 |
| dataset_from_side_rep | 293 | 3 | 2 | 0.8225255972696246 | 0.649493927125506 | 0.6761054421768709 |
| dataset_from_usage | 293 | 3 | 2 | 0.7781569965870307 | 0.5 | 0.43761996161228406 |
| dataset_from_usage_coeff | 293 | 4 | 2 | 0.7406143344709898 | 0.48687584345479085 | 0.449901185770751 |
| side_from_coeff | 293 | 1 | 3 | 0.9283276450511946 | 0.9244162015246352 | 0.9271085461619076 |
| side_from_free_rep | 293 | 32 | 3 | 0.4061433447098976 | 0.37664937664937664 | 0.3145659242313517 |
| side_from_side_rep | 293 | 3 | 3 | 0.9215017064846417 | 0.9163051331726031 | 0.918058863282082 |
| side_from_usage | 293 | 3 | 3 | 0.9249146757679181 | 0.9235665018797549 | 0.9226328815170538 |
| side_from_usage_coeff | 293 | 4 | 3 | 0.9385665529010239 | 0.9370345153477683 | 0.9370345153477683 |

## Task Artifacts

### side_from_side_rep

- accuracy: `0.9215`
- balanced_accuracy: `0.9163`
- macro_f1: `0.9181`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_side_rep_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_side_rep_confusion.png`

### side_from_free_rep

- accuracy: `0.4061`
- balanced_accuracy: `0.3766`
- macro_f1: `0.3146`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_free_rep_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_free_rep_confusion.png`

### dataset_from_side_rep

- accuracy: `0.8225`
- balanced_accuracy: `0.6495`
- macro_f1: `0.6761`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_side_rep_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_side_rep_confusion.png`

### dataset_from_free_rep

- accuracy: `0.8123`
- balanced_accuracy: `0.6594`
- macro_f1: `0.6809`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_free_rep_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_free_rep_confusion.png`

### dataset_from_private_rep

- accuracy: `0.8874`
- balanced_accuracy: `0.8286`
- macro_f1: `0.8341`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_private_rep_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_private_rep_confusion.png`

### side_from_usage

- accuracy: `0.9249`
- balanced_accuracy: `0.9236`
- macro_f1: `0.9226`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_usage_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_usage_confusion.png`

### side_from_coeff

- accuracy: `0.9283`
- balanced_accuracy: `0.9244`
- macro_f1: `0.9271`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_coeff_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_coeff_confusion.png`

### side_from_usage_coeff

- accuracy: `0.9386`
- balanced_accuracy: `0.9370`
- macro_f1: `0.9370`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_usage_coeff_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/side_from_usage_coeff_confusion.png`

### dataset_from_usage

- accuracy: `0.7782`
- balanced_accuracy: `0.5000`
- macro_f1: `0.4376`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_usage_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_usage_confusion.png`

### dataset_from_coeff

- accuracy: `0.7782`
- balanced_accuracy: `0.5000`
- macro_f1: `0.4376`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_coeff_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_coeff_confusion.png`

### dataset_from_usage_coeff

- accuracy: `0.7406`
- balanced_accuracy: `0.4869`
- macro_f1: `0.4499`
- confusion_csv: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_usage_coeff_confusion.csv`
- confusion_png: `outputs/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/kfold_report/dataset_from_usage_coeff_confusion.png`

