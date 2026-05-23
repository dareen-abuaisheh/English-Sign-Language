"""
train_letter.py

This file trains the LETTER detection model for American Sign Language letters.

It uses:
- letter_dataset/
- letter_dataset.yaml
- YOLOv8n as the default lightweight model

Run:
    python train_letter.py
"""

from pathlib import Path

import torch
from ultralytics import YOLO


def check_letter_dataset():
    """Check that the letter dataset files and folders exist."""

    yaml_path = Path("letter_dataset.yaml")

    if not yaml_path.exists():
        print("Error: letter_dataset.yaml was not found.")
        return False

    required_folders = [
        Path("letter_dataset/train/images"),
        Path("letter_dataset/train/labels"),
        Path("letter_dataset/valid/images"),
        Path("letter_dataset/valid/labels"),
        Path("letter_dataset/test/images"),
        Path("letter_dataset/test/labels"),
    ]

    for folder in required_folders:
        if not folder.exists():
            print(f"Error: Missing folder: {folder}")
            return False

    return True


def choose_training_device():
    """Use GPU if CUDA is available, otherwise use CPU."""

    if torch.cuda.is_available():
        print("CUDA GPU detected. Letter training will use the GPU.")
        return 0

    print("Warning: CUDA GPU was not detected.")
    print("Letter training will use CPU, which may be slow.")
    return "cpu"


def train_letter_model():
    """Train a YOLOv8n model for ASL letter detection."""

    if not check_letter_dataset():
        return

    device = choose_training_device()

    # YOLOv8n is the nano model.
    # It is small, fast, and suitable for beginner projects and limited GPUs.
    model = YOLO("yolov8n.pt")

    # epochs:
    # Number of times the model sees the whole training dataset.
    # More epochs can improve accuracy but make training slower.
    # Too many epochs can cause overfitting.
    # Increase if validation performance is still improving.
    # Decrease if training is too slow or validation gets worse.
    epochs = 50

    # batch:
    # Number of images processed at one time.
    # Larger batch can be faster on a strong GPU but uses more memory.
    # Increase if your GPU has enough memory.
    # Decrease if you get a GPU memory error.
    batch = 8

    # imgsz:
    # Image size used by YOLO during training.
    # Larger images can improve accuracy for small details but train slower.
    # Increase if signs are hard to see.
    # Decrease if training is too slow.
    imgsz = 640

    # lr0:
    # Initial learning rate.
    # Higher values learn faster but can be unstable.
    # Lower values are more stable but slower.
    # Decrease if loss jumps a lot.
    # Increase slightly if learning is too slow.
    lr0 = 0.01

    # optimizer:
    # Method used to update model weights.
    # "auto" lets Ultralytics choose a good option.
    # This is simple and beginner-friendly.
    optimizer = "auto"

    # patience:
    # Early stopping patience.
    # If validation does not improve for this many epochs, training can stop.
    # Increase to give training more time.
    # Decrease to stop sooner.
    patience = 10

    try:
        model.train(
            data="letter_dataset.yaml",
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer,
            patience=patience,
            device=device,
            project="outputs",
            name="letter_training_results",
            exist_ok=True,
        )

        print("Letter training finished.")
        print("Best weights are saved at: outputs/letter_training_results/weights/best.pt")
        print("Copy that file to: models/letter_best.pt")

    except RuntimeError as error:
        print("Letter training stopped because of a runtime error.")
        print("This may be caused by low GPU memory or dataset problems.")
        print(f"Error message: {error}")
    except Exception as error:
        print("Letter training failed.")
        print(f"Error message: {error}")


if __name__ == "__main__":
    train_letter_model()
