# Real-Time American Sign Language Detection using YOLOv8

This is a simple and educational machine learning project for detecting American Sign Language (ASL) hand signs using YOLOv8.

The project is designed for university students and beginners. The code uses a clear procedural style, simple functions, relative paths, and comments that explain the important steps.

## Project Idea

The goal is to train a YOLOv8 object detection model to recognize ASL hand signs from a custom dataset.

After training, the model can be used in two ways:

1. Detect ASL signs in a single image.
2. Detect ASL signs in real time using a webcam.

## Technologies Used

- Python
- YOLOv8 from Ultralytics
- OpenCV
- NumPy
- Custom ASL dataset

## Why YOLOv8n?

This project uses `yolov8n.pt` by default.

`YOLOv8n` means YOLOv8 nano. It is the smallest standard YOLOv8 model. It was chosen because it is lightweight, fast, and suitable for student projects, laptops, and limited GPU resources.

Larger YOLO models may give better accuracy, but they usually need more training time, more GPU memory, and stronger hardware.

## Project Pipeline

1. Prepare the ASL dataset in YOLO format.
2. Edit `dataset.yaml` if your class names are different.
3. Install the required Python packages.
4. Train the model using `train.py`.
5. Copy the best trained model to `models/best.pt`.
6. Test one image using `predict.py`.
7. Run real-time webcam detection using `webcam_detection.py`.

## Folder Structure

```text
sign-language/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── inference/
├── models/
├── outputs/
│   └── results/
├── training/
├── utils/
├── dataset.yaml
├── predict.py
├── README.md
├── requirements.txt
├── train.py
└── webcam_detection.py
```

## What Each Main File Does

### requirements.txt

Lists the Python packages needed for the project.

Install with:

```bash
pip install -r requirements.txt
```

Expected output: Python downloads and installs `ultralytics`, `opencv-python`, and `numpy`.

### dataset.yaml

Tells YOLOv8 where the dataset is located and what classes exist.

YOLOv8 uses it during training.

### train.py

Trains the YOLOv8n model on the ASL dataset.

Run with:

```bash
python train.py
```

Expected output: training logs, validation metrics, weights, plots, and result files inside `outputs/training_results/`.

### predict.py

Runs detection on one image.

Run with:

```bash
python predict.py --image inference/sample.jpg --model models/best.pt
```

Expected output: an OpenCV window showing detections and a saved image at `outputs/results/prediction_result.jpg`.

### webcam_detection.py

Runs real-time ASL detection using a webcam.

Run with:

```bash
python webcam_detection.py --model models/best.pt
```

Expected output: a webcam window with bounding boxes, class names, and confidence scores. Press `q` to quit.

## Dataset Organization

YOLO object detection datasets use separate folders for images and labels.

Use this structure:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Where Images Go

Training images go here:

```text
dataset/train/images/
```

Validation images go here:

```text
dataset/valid/images/
```

Test images go here:

```text
dataset/test/images/
```

### Where Annotation Files Go

Training label files go here:

```text
dataset/train/labels/
```

Validation label files go here:

```text
dataset/valid/labels/
```

Test label files go here:

```text
dataset/test/labels/
```

### Image and Label Names Must Match

Every image should have a matching `.txt` annotation file with the same name.

Example:

```text
dataset/train/images/A_001.jpg
dataset/train/labels/A_001.txt
```

If the image is called `A_001.jpg`, the label file must be called `A_001.txt`.

## YOLO Annotation Format

Each object in an image is written as one line in the label file:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.500 0.500 0.250 0.300
```

This means:

- `0`: class id, such as class `A`
- `0.500`: x-coordinate of the center of the box
- `0.500`: y-coordinate of the center of the box
- `0.250`: width of the box
- `0.300`: height of the box

### Normalized Coordinates

YOLO uses normalized coordinates. This means the values are between `0` and `1` instead of using raw pixels.

For example, if the image width is 1000 pixels and the box center is at x = 500 pixels, then:

```text
500 / 1000 = 0.500
```

Normalized coordinates make the labels work even if images have different sizes.

## dataset.yaml Explained

The file contains:

```yaml
path: dataset
train: train/images
val: valid/images
test: test/images
names:
  0: A
  1: B
```

Meaning:

- `path`: main dataset folder
- `train`: training image folder inside `path`
- `val`: validation image folder inside `path`
- `test`: test image folder inside `path`
- `names`: class names used by the model

To add ASL classes, add them under `names` and make sure the class ids match your label files.

Example:

```yaml
names:
  0: A
  1: B
  2: C
```

If a label file starts with `2`, YOLO will treat that object as class `C`.

## Installation

### 1. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

macOS or Linux:

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import ultralytics, cv2, numpy; print('Installation works')"
```

### 5. Check CUDA GPU Support

Run:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `True`, PyTorch can see your CUDA GPU.

If it prints `False`, training will use the CPU. CPU training works, but it is slower.

## Training

Before training, place your dataset inside the `dataset/` folder using the required YOLO structure.

Then run:

```bash
python train.py
```

The script does the following:

1. Checks for `dataset.yaml`.
2. Checks that the dataset folders exist.
3. Checks whether CUDA is available.
4. Loads the pretrained `yolov8n.pt` model.
5. Uses transfer learning to train on your ASL dataset.
6. Saves results inside `outputs/training_results/`.

### Transfer Learning Explained

Transfer learning means starting from a model that already learned useful visual features from a large dataset.

Instead of training from zero, YOLOv8 starts with pretrained knowledge and then adapts to ASL hand signs. This usually trains faster and works better for student-sized datasets.

## Training Hyperparameters

The main hyperparameters are explained directly inside `train.py` before each value is defined.

The project uses:

```python
epochs = 50
batch = 8
imgsz = 640
lr0 = 0.01
optimizer = "auto"
patience = 10
```

Short summary:

- `epochs`: how many times the model sees the full dataset
- `batch`: how many images are processed at once
- `imgsz`: image size used during training
- `lr0`: initial learning rate
- `optimizer`: method used to update model weights
- `patience`: early stopping wait time

## Model Evaluation

YOLOv8 prints and saves useful metrics after training.

### Precision

Precision answers: when the model predicts a sign, how often is it correct?

High precision means fewer false detections.

### Recall

Recall answers: out of all real signs, how many did the model find?

High recall means the model misses fewer signs.

### mAP

mAP means mean Average Precision.

It is a common object detection score. Higher mAP usually means better detection performance.

### Loss

Loss measures how wrong the model is during training.

In general, loss should decrease over time. If loss stays high, the model may not be learning well.

### Confusion Matrix

A confusion matrix shows which classes the model predicts correctly and which classes it confuses.

For example, if the model often predicts `M` when the real sign is `N`, the confusion matrix helps reveal that problem.

### Overfitting

Overfitting happens when the model performs very well on training images but poorly on validation images.

This means the model memorized the training set instead of learning general patterns.

### Underfitting

Underfitting happens when the model performs poorly on both training and validation images.

This means the model has not learned enough yet.

## Image Prediction

Place a test image inside the `inference/` folder, for example:

```text
inference/sample.jpg
```

Make sure your trained model is here:

```text
models/best.pt
```

Then run:

```bash
python predict.py --image inference/sample.jpg --model models/best.pt
```

The script will:

1. Load the trained model.
2. Read the image with OpenCV.
3. Run YOLOv8 prediction.
4. Draw bounding boxes and confidence scores.
5. Show the result using `cv2.imshow()`.
6. Save the result to `outputs/results/prediction_result.jpg`.

## Real-Time Webcam Demo

Make sure your trained model is here:

```text
models/best.pt
```

Run:

```bash
python webcam_detection.py --model models/best.pt
```

The script will:

1. Open the webcam.
2. Read video frames one by one.
3. Run YOLOv8 inference on each frame.
4. Draw bounding boxes and confidence scores.
5. Display the result using `cv2.imshow()`.
6. Quit safely when you press `q`.

If your computer has more than one webcam, try:

```bash
python webcam_detection.py --model models/best.pt --camera 1
```

## Outputs

YOLOv8 saves training files inside the folder set in `train.py`:

```text
outputs/training_results/
```

Important files may include:

- `weights/best.pt`: best trained model
- `weights/last.pt`: model from the last epoch
- `results.csv`: training metrics for each epoch
- `results.png`: training graphs
- `confusion_matrix.png`: confusion matrix
- `labels.jpg`: dataset label overview
- `train_batch*.jpg`: examples of training batches
- `val_batch*.jpg`: examples of validation predictions

For prediction, this project saves image results here:

```text
outputs/results/
```

To use the trained model with `predict.py` or `webcam_detection.py`, copy:

```text
outputs/training_results/weights/best.pt
```

into:

```text
models/best.pt
```

## Fine-Tuning Guide

### 1. How to Improve Accuracy

Use more high-quality labeled images. Good labels are very important for object detection.

Try to include different people, hand sizes, camera angles, distances, lighting conditions, and backgrounds.

### 2. How to Reduce Overfitting

Add more validation images and more variety. If training accuracy is high but validation accuracy is low, the model may be memorizing.

You can reduce epochs, use more augmentation, or collect more data.

### 3. How to Train Faster

Use YOLOv8n, reduce `imgsz`, reduce `epochs`, or use a CUDA GPU.

You can also increase `batch` if your GPU has enough memory.

### 4. How to Choose Epochs

Start with 50 epochs. If the model is still improving, try 75 or 100.

If validation performance stops improving early, use fewer epochs or rely on `patience` for early stopping.

### 5. How Batch Size Affects GPU Memory

A larger batch processes more images at once, which can be faster on a good GPU.

However, larger batches use more GPU memory. If you see an out-of-memory error, lower the batch size.

### 6. When to Change Image Size

Use a larger image size if hand signs are small or details are hard to see.

Use a smaller image size if training is too slow or your GPU memory is limited.

### 7. What to Do If Validation Loss Increases

If validation loss increases while training loss decreases, the model may be overfitting.

Try fewer epochs, more data, cleaner labels, or more varied images.

### 8. What to Do If the Model Memorizes Training Data

Add more varied images. Avoid having almost identical images in training and validation sets.

Make sure the validation set contains people, backgrounds, and lighting conditions that are different from training.

### 9. What Augmentation Helps Hand Gesture Datasets

Helpful variety includes small rotations, brightness changes, contrast changes, slight scaling, and background variety.

Avoid augmentations that make the hand sign incorrect or unrealistic.

### 10. When to Use Smaller or Larger YOLO Models

Use YOLOv8n when you need speed or have limited hardware.

Try YOLOv8s or YOLOv8m if you have a stronger GPU and need better accuracy.

### 11. How Lighting and Background Affect Predictions

Poor lighting can hide finger shapes. Busy backgrounds can confuse the model.

Collect images in different lighting conditions and backgrounds so the model learns to focus on the hand sign.

### 12. Common Beginner Mistakes

Common mistakes include:

- image files without matching label files
- label files with the wrong class id
- incorrect folder names
- using absolute paths that break on another computer
- forgetting to update `dataset.yaml`
- training with too few images
- using validation images that are too similar to training images
- expecting good webcam results from a very small dataset

## Future Improvements

Possible improvements include:

- training on more ASL signs
- collecting more images from different people
- testing larger YOLOv8 models
- saving webcam recordings
- adding a simple user interface
- adding FPS display during webcam detection
- improving predictions with better lighting and camera placement

## Limitations

This project detects visible ASL hand signs as object classes.

It does not understand full ASL grammar, sentence meaning, motion over time, or two-handed sign sequences unless the dataset and model are designed for those cases.

For best results, the dataset should be large, varied, and carefully labeled.
