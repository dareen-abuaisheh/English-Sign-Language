# Real-Time American Sign Language Detection using YOLOv8

This project now contains two independent YOLOv8 modules:

1. **Letter Detection Module**
2. **Word Detection Module**

The two modules are intentionally separate. Letter files use `letter` in their names, and word files use `word` in their names. This keeps the project simple, organized, and easier to explain in a university report.

## Technologies Used

- Python
- YOLOv8 from Ultralytics
- OpenCV
- NumPy
- Custom YOLO-format datasets

YOLOv8n is used by default because it is lightweight, fast, and suitable for student projects and limited GPU resources.

## Project Structure

```text
project/
├── letter_dataset/
│   ├── train/
│   ├── valid/
│   └── test/
├── word_dataset/
│   ├── train/
│   ├── valid/
│   └── test/
├── models/
│   ├── letter_best.pt
│   └── word_best.pt
├── train_letter.py
├── predict_letter.py
├── webcam_letter_detection.py
├── train_word.py
├── predict_word.py
├── webcam_word_detection.py
├── letter_dataset.yaml
├── word_dataset.yaml
├── outputs/
└── README.md
```

## Letter Detection vs Word Detection

### Letter Detection

Letter detection recognizes individual ASL letters such as:

```text
A, B, C, D, ... Z
```

Each letter is treated as a separate class.

Example:

```text
class 0 = A
class 1 = B
class 2 = C
```

### Word Detection

Word detection recognizes complete ASL words or signs such as:

```text
hello, thank-you, no, water, please
```

Each word is treated as a separate class.

Example:

```text
class 57 = hello
class 71 = no
class 93 = thank-you
```

Word detection is usually harder than letter detection because words may involve more complex hand shapes, positions, and sometimes motion.

## YOLO Dataset Format

Both modules use YOLO format.

Each dataset has this structure:

```text
train/images
train/labels
valid/images
valid/labels
test/images
test/labels
```

Images go inside `images/` folders.

Annotation `.txt` files go inside `labels/` folders.

Image and label names must match.

Example:

```text
letter_dataset/train/images/A_001.jpg
letter_dataset/train/labels/A_001.txt
```

YOLO annotation format:

```text
class_id x_center y_center width height
```

The coordinates are normalized, meaning values are between `0` and `1` instead of pixel values.

Example:

```text
0 0.500 0.500 0.250 0.300
```

This means class `0`, with a box centered in the image.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Check CUDA GPU support:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If it prints `True`, training can use the GPU. If it prints `False`, training will use CPU and may be slower.

## Letter Detection Module

### Train Letter Model

```bash
python3 train_letter.py
```

This uses:

```text
letter_dataset/
letter_dataset.yaml
```

Training results are saved in:

```text
outputs/letter_training_results/
```

After training, copy:

```text
outputs/letter_training_results/weights/best.pt
```

to:

```text
models/letter_best.pt
```

### Predict One Letter Image

```bash
python3 predict_letter.py --image inference/sample.jpeg --model models/letter_best.pt
```

Output is saved in:

```text
outputs/letter_predictions/letter_prediction_result.jpg
```

This script also displays the result using OpenCV.

### Run Letter Webcam Detection

```bash
python3 webcam_letter_detection.py --model models/letter_best.pt
```

Press `q` to quit safely.

## Word Detection Module

### Train Word Model

```bash
python3 train_word.py
```

This uses:

```text
word_dataset/
word_dataset.yaml
```

Training results are saved in:

```text
outputs/word_training_results/
```

After training, copy:

```text
outputs/word_training_results/weights/best.pt
```

to:

```text
models/word_best.pt
```

### Predict One Word Image

```bash
python3 predict_word.py --image inference/sample.jpeg --model models/word_best.pt
```

Output is saved in:

```text
outputs/word_predictions/word_prediction_result.jpg
```

Important: `predict_word.py` does not use `cv2.imshow()` because remote GPU servers often do not support display windows.

### Run Word Webcam Detection

```bash
python3 webcam_word_detection.py --model models/word_best.pt
```

Press `q` to quit safely.

## Training Hyperparameters

Both training scripts explain each hyperparameter before using it.

Main hyperparameters:

- `epochs`: how many times the model sees the full dataset
- `batch`: how many images are processed at once
- `imgsz`: image size used during training
- `lr0`: initial learning rate
- `optimizer`: method used to update model weights
- `patience`: early stopping setting

Increase epochs if the model is still improving.

Decrease batch size if GPU memory is low.

Increase image size if hand details are too small.

Decrease image size if training is too slow.

## Outputs

Letter training outputs:

```text
outputs/letter_training_results/
```

Word training outputs:

```text
outputs/word_training_results/
```

Letter prediction outputs:

```text
outputs/letter_predictions/
```

Word prediction outputs:

```text
outputs/word_predictions/
```

YOLO may save:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- validation prediction images

## Evaluation Metrics

YOLO reports common object detection metrics.

Precision means: when the model predicts a sign, how often is it correct?

Recall means: out of all real signs, how many did the model find?

mAP means mean Average Precision. Higher mAP usually means better detection.

Loss shows how wrong the model is during training. Loss should usually decrease.

A confusion matrix shows which classes are confused with other classes.

## Future Improvements

Possible future improvements:

- collect more images from different people
- improve lighting and background variety
- train larger YOLO models if GPU resources allow
- add a simple web interface
- add FPS display for webcam detection
- support sentence-level ASL recognition
- use video-based models for signs that require motion

## Current Limitations

The letter model detects individual letters only.

The word model detects word classes from images or frames, but it does not understand full ASL grammar.

Some ASL words may require motion, so a single-frame object detector may not fully understand every sign.

The quality of results depends strongly on dataset size, label quality, lighting, camera position, and background variety.
