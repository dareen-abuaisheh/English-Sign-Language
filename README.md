# Real-Time American Sign Language Recognition using YOLOv8

A real-time American Sign Language (ASL) recognition system built using **YOLOv8** for detecting both ASL alphabet letters and word-level gestures through webcam-based inference.

---

# Project Overview

This project uses deep learning and computer vision techniques to recognize ASL gestures in real time. The system was implemented using the **Ultralytics YOLOv8** framework and organized into two independent detection modules:

- **Letter Detection Module** — detects ASL alphabet letters (A–Z)
- **Word Detection Module** — detects ASL words and mixed gesture classes

The system supports:
- real-time webcam inference
- object detection using bounding boxes
- confidence score visualization
- GPU-accelerated training and inference

---

# Features

- Real-time ASL gesture recognition
- YOLOv8 object detection pipeline
- Letter-level and word-level detection
- Webcam-based live inference
- Transfer learning using pretrained YOLOv8 weights
- CUDA GPU acceleration
- Training visualization and evaluation plots
- Confusion matrix analysis
- Comparative analysis between models

---

# Technologies Used

- Python
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- NumPy
- pandas
- matplotlib
- scikit-learn

---

# Hardware

Training and inference were performed using:

- NVIDIA A100-SXM4-80GB GPU
- CUDA acceleration

---

# Project Structure

```text
sign-language/
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── word_dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/
│   ├── letter_best.pt
│   └── word_best.pt
│
├── outputs/
│   ├── letter_training_results/
│   ├── word_training_results/
│   └── report_plots/
│
├── comparison/
│   ├── confusion_matrices/
│   └── plots/
│
├── train_letter.py
├── train_word.py
├── predict.py
├── webcam_detection.py
└── README.md
```

---

# Datasets

## Letter Dataset

| Property | Value |
|---|---|
| Classes | 26 |
| Training Images | 1512 |
| Validation Images | 144 |
| Test Images | 72 |

---

## Mixed Letter + Word Dataset

| Property | Value |
|---|---|
| Classes | 106 |
| Training Images | 20,706 |
| Validation Images | 1,719 |
| Test Images | 925 |

---

# YOLO Annotation Format

Each label file follows the YOLO format:

```text
class_id x_center y_center width height
```

All coordinates are normalized relative to image dimensions.

---

# Model Performance

## Letter Detection Model

| Metric | Value |
|---|---|
| Precision | 0.94436 |
| Recall | 0.88544 |
| mAP50 | 0.95736 |
| mAP50-95 | 0.77851 |

---

## Mixed Dataset Detection Model

| Metric | Value |
|---|---|
| Precision | 0.96172 |
| Recall | 0.96392 |
| mAP50 | 0.97259 |
| mAP50-95 | 0.73395 |

---

# How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/dareen-abuaisheh/English-Sign-Language.git
cd English-Sign-Language
```

---

## 2. Create Virtual Environment

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Requirements

```bash
pip install ultralytics opencv-python torch torchvision torchaudio numpy pandas matplotlib scikit-learn seaborn
```

---

# Training

## Train the Letter Detection Model

```bash
python train_letter.py
```

---

## Train the Mixed Dataset / Word Detection Model

```bash
python train_word.py
```

---

# Image Prediction

Run prediction on a single image:

```bash
python predict.py --image path/to/image.jpg --model models/letter_best.pt
```

Example:

```bash
python predict.py --image dataset/test/images/sample.jpg --model models/letter_best.pt
```

---

# Real-Time Webcam Detection

Run webcam inference:

```bash
python webcam_detection.py
```

The system will:
- open the webcam
- detect ASL gestures
- display bounding boxes
- show class labels and confidence scores

---

# Comparative Analysis

The project includes a comparative analysis between:
- Letter-Only Detection Model
- Mixed Dataset Detection Model

The comparison evaluates:
- classification complexity
- prediction stability
- gesture ambiguity
- real-time usability
- model generalization

---

# Challenges and Limitations

- Lighting variability
- Gesture similarity between classes
- Background complexity
- Webcam quality limitations
- Lack of temporal modeling
- Hardware dependency for real-time inference

---

# Future Improvements

- Sentence-level recognition
- Temporal modeling using LSTM or Transformers
- Mobile deployment
- Text-to-speech integration
- Larger and more diverse datasets
- Enhanced real-time stability

---

# Authors

- Dareen Abuaisheh
- Hana Izzdeen
- Mohammed Nabulsi

---

# Course Information

**Course:** Advanced Machine Learning  
**Instructor:** Dr. Adnan Salman  
**University:** An-Najah National University  
**Date:** 23/5/2026

---

# License

This project was developed for educational and academic purposes.
