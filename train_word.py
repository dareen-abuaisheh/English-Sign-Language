"""
train_word.py

This file trains the WORD detection model for American Sign Language words.

In word detection, each complete word is treated as one object class.
Examples: hello, thank-you, yes, no.

It uses:
- word_dataset/
- word_dataset.yaml
- YOLOv8n as the default lightweight model

Run:
    python train_word.py
"""

from pathlib import Path

import torch
from ultralytics import YOLO


def check_word_dataset():
    """Check that the word dataset files and folders exist."""

    yaml_path = Path("word_dataset.yaml")

    if not yaml_path.exists():
        print("Error: word_dataset.yaml was not found.")
        return False

    required_folders = [
        Path("word_dataset/train/images"),
        Path("word_dataset/train/labels"),
        Path("word_dataset/valid/images"),
        Path("word_dataset/valid/labels"),
        Path("word_dataset/test/images"),
        Path("word_dataset/test/labels"),
    ]

    for folder in required_folders:
        if not folder.exists():
            print(f"Error: Missing folder: {folder}")
            print("Please organize the word dataset in YOLO format.")
            return False

    return True


def choose_training_device():
    """Use GPU if CUDA is available, otherwise use CPU."""

    if torch.cuda.is_available():
        print("CUDA GPU detected. Word training will use the GPU.")
        return 0

    print("Warning: CUDA GPU was not detected.")
    print("Word training will use CPU, which may be slow.")
    return "cpu"


def train_word_model():
    """Train a YOLOv8n model for ASL word detection."""

    if not check_word_dataset():
        return

    device = choose_training_device()

    # YOLOv8n is the nano model.
    # It is small and fast, which makes it good for a first word-detection model.
    # Larger models may improve accuracy but need more GPU memory and time.
    model = YOLO("yolov8n.pt")

    # epochs:
    # Number of times the model sees all training images.
    # More epochs can improve accuracy because the model practices more.
    # More epochs also make training slower.
    # Too many epochs can cause overfitting.
    # Increase if validation results are still improving.
    # Decrease if validation gets worse or training takes too long.
    epochs = 50

    # batch:
    # Number of images processed together in one training step.
    # Larger batch can train faster on a strong GPU.
    # Larger batch uses more GPU memory.
    # Increase if your GPU has extra memory.
    # Decrease if you get a memory error.
    batch = 8

    # imgsz:
    # Image size used during training.
    # Larger image size can improve accuracy for small hand details.
    # Larger image size makes training slower and uses more memory.
    # Increase if signs are small or difficult to see.
    # Decrease if training is too slow.
    imgsz = 640

    # lr0:
    # Initial learning rate.
    # It controls how large each learning update is.
    # Higher learning rate can train faster but may become unstable.
    # Lower learning rate is slower but often more stable.
    # Decrease if loss changes wildly.
    # Increase slightly if learning is very slow.
    lr0 = 0.01

    # optimizer:
    # Optimizer controls how model weights are updated.
    # "auto" lets Ultralytics choose a suitable optimizer.
    # This keeps the project simple for beginners.
    optimizer = "auto"

    # patience:
    # Early stopping patience.
    # If validation does not improve for this many epochs, training can stop.
    # Increase if you want to give the model more time.
    # Decrease if you want training to stop earlier.
    patience = 10

    try:
        model.train(
            data="word_dataset.yaml",
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer,
            patience=patience,
            device=device,
            project="outputs",
            name="word_training_results",
            exist_ok=True,
        )

        print("Word training finished.")
        print("Best weights are saved at: outputs/word_training_results/weights/best.pt")
        print("Copy that file to: models/word_best.pt")

    except RuntimeError as error:
        print("Word training stopped because of a runtime error.")
        print("This may be caused by low GPU memory or dataset problems.")
        print(f"Error message: {error}")
    except Exception as error:
        print("Word training failed.")
        print(f"Error message: {error}")


if __name__ == "__main__":
    train_word_model()
