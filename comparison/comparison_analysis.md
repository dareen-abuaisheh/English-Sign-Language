# Comparative Analysis: Three-Model Detection Comparison

## 1. Models Included

- Char Unfreeze Model (`runs/detect/outputs/unfreeze_experiments/char_unfreeze_yolov8n/results.csv`)
- Word Unfreeze Model (`runs/detect/outputs/unfreeze_experiments/word_unfreeze_yolov8n/results.csv`)
- Mohammed Model (`models/best_word_unfreeze_yolo.pt`, using the same training run history as word unfreeze)

## 2. Final Epoch Metrics

| Metric | Char Unfreeze | Word Unfreeze | Mohammed Model |
|---|---:|---:|---:|
| Precision | 0.89047 | 0.81279 | 0.81279 |
| Recall | 0.84372 | 0.81457 | 0.81457 |
| mAP50 | 0.92185 | 0.89786 | 0.89786 |
| mAP50-95 | 0.74302 | 0.59511 | 0.59511 |
| Train total loss | 1.94568 | 2.61574 | 2.61574 |
| Val total loss | 2.78261 | 3.50325 | 3.50325 |

## 3. Summary

- Best final mAP50: **Char Unfreeze** (0.92185)
- Mohammed model is now included in all comparison charts and tabular metrics.
- Confusion matrices are available for char and Mohammed model outputs in `comparison/confusion_matrices/`.
