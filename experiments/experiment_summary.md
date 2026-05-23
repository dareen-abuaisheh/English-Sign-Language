# Automated YOLOv8 ASL Letter Experiment Summary

This report was generated automatically by `run_experiments.py`.
The experiments used only the ASL letter dataset and were trained for a small number of epochs for lightweight comparison.

## Safety Notes

The experiment pipeline used a temporary folder for YOLO outputs and deleted it after completion.
The original `outputs/`, `runs/`, `models/`, main model weights, logs, CSV files, and plots were not modified by the experiment script.

## Experiment Table

| experiment_name | comparison_group | yolo_model | optimizer | image_size | epochs | final_precision | final_recall | final_map50 | final_map50_95 | final_train_box_loss | final_train_cls_loss | final_train_dfl_loss | final_val_box_loss | final_val_cls_loss | final_val_dfl_loss | training_time_seconds | inference_speed_ms | status | error_message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_yolov8n_640_auto | model comparison | yolov8n.pt | auto | 640 | 5 | 0.56429 | 0.68718 | 0.68505 | 0.53654 | 0.80588 | 2.56185 | 1.26471 | 0.89358 | 1.53544 | 1.30659 | 80.49 | 1.9445 | success | not available |
| model_yolov8s_640_auto | model comparison | yolov8s.pt | auto | 640 | 5 | 0.7665 | 0.87801 | 0.9233 | 0.75958 | 0.75345 | 1.34566 | 1.19772 | 0.78475 | 0.80892 | 1.19422 | 80.72 | 1.8925 | success | not available |
| image_size_416_yolov8n | image size comparison | yolov8n.pt | auto | 416 | 5 | 0.49547 | 0.73207 | 0.6842 | 0.54719 | 0.83906 | 2.383 | 1.18701 | 0.84903 | 1.5414 | 1.13166 | 74.49 | 1.1626 | success | not available |
| image_size_640_yolov8n | image size comparison | yolov8n.pt | auto | 640 | 5 | 0.56429 | 0.68718 | 0.68505 | 0.53654 | 0.80588 | 2.56185 | 1.26471 | 0.89358 | 1.53544 | 1.30659 | 79.44 | 0.8002 | success | not available |
| optimizer_sgd_yolov8n | optimizer comparison | yolov8n.pt | SGD | 640 | 5 | 0.5441 | 0.63787 | 0.62848 | 0.4949 | 0.88205 | 2.81552 | 1.30806 | 0.87365 | 1.64443 | 1.28887 | 78.06 | 0.872 | success | not available |
| optimizer_adamw_yolov8n | optimizer comparison | yolov8n.pt | AdamW | 640 | 5 | 0.26515 | 0.55632 | 0.33872 | 0.24599 | 1.16408 | 2.58297 | 1.48315 | 1.11854 | 2.16469 | 1.53892 | 79.75 | 0.8892 | success | not available |

## Best Configuration by mAP50

The highest final mAP50 was achieved by `model_yolov8s_640_auto` with mAP50 `0.9233`. This configuration used model `yolov8s.pt`, optimizer `auto`, and image size `640`.

## Fastest Training Configuration

The fastest successful experiment was `image_size_416_yolov8n`, with training time `74.49` seconds.

## Precision and Recall

The best final precision was produced by `model_yolov8s_640_auto` with precision `0.7665`. Precision measures how often predicted detections are correct.
The best final recall was produced by `model_yolov8s_640_auto` with recall `0.87801`. Recall measures how many real objects the model found.

## Model Size Comparison

The YOLOv8n and YOLOv8s experiments compare a smaller model with a larger model. YOLOv8n is usually faster and lighter, while YOLOv8s may improve accuracy at the cost of more computation. The best choice depends on whether the project needs maximum speed or stronger accuracy.

## Image Size Comparison

The image-size experiments compare `416` and `640`. A smaller image size usually trains faster and uses less memory. A larger image size may detect hand details better, but it normally takes more time.

## Optimizer Comparison

The optimizer experiments compare SGD and AdamW. SGD is a traditional optimizer often used for stable object detection training. AdamW can converge quickly in some cases, but its performance depends on the dataset and learning rate.

## Speed and Accuracy Tradeoff

A real-time ASL system should balance detection accuracy with inference speed. If two experiments produce similar accuracy, the faster and smaller configuration is usually more practical for webcam detection.

## Recommended Setup for Real-Time ASL Detection

For real-time ASL letter detection, the best setup should be selected by considering both mAP50 and training/inference speed. YOLOv8n is usually the safest choice for real-time use because it is lightweight. If YOLOv8s provides a clear accuracy improvement and the hardware can run it smoothly, it can be considered as an alternative.

## Comparison Plot

The combined comparison plot is saved as `comparison_plot.png`.