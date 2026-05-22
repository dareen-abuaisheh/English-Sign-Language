"""
train.py

This file trains a YOLOv8 model for American Sign Language detection.

The code is intentionally simple and educational:
- it checks that the dataset exists
- it loads a lightweight pretrained YOLOv8n model
- it trains the model using transfer learning
- it saves the results in the outputs folder

Run:
    python train.py
"""

from pathlib import Path

import torch
from ultralytics import YOLO


def check_dataset_files(dataset_yaml_path):
    """Check that dataset.yaml and the main dataset folders exist."""

    if not dataset_yaml_path.exists():
        print("Error: dataset.yaml was not found.")
        print("Please make sure dataset.yaml exists in the project root folder.")
        return False

    dataset_folder = Path("dataset")

    if not dataset_folder.exists():
        print("Error: The dataset folder was not found.")
        print("Please create the dataset folder and add your ASL dataset.")
        return False

    required_folders = [
        Path("dataset/train/images"),
        Path("dataset/train/labels"),
        Path("dataset/valid/images"),
        Path("dataset/valid/labels"),
        Path("dataset/test/images"),
        Path("dataset/test/labels"),
    ]

    for folder in required_folders:
        if not folder.exists():
            print(f"Error: Missing folder: {folder}")
            print("Please organize the dataset using the YOLO folder structure.")
            return False

    return True


def show_device_information():
    """Show whether training will use a GPU or CPU."""

    if torch.cuda.is_available():
        print("CUDA GPU detected.")
        print("Training will use the GPU, which is usually much faster.")
        return 0

    print("Warning: CUDA GPU was not detected.")
    print("Training will use the CPU. This works, but it can be slow.")
    return "cpu"


def train_model():
    """Train YOLOv8n on the custom ASL dataset."""

    dataset_yaml_path = Path("dataset.yaml")

    if not check_dataset_files(dataset_yaml_path):
        return

    device = show_device_information()

    # Model choice:
    # YOLOv8n means YOLOv8 nano.
    # It is the smallest standard YOLOv8 model.
    # It is a good default for university projects because it trains faster,
    # needs less GPU memory, and can run in real time more easily.
    # Larger models may be more accurate but are slower and need stronger hardware.
    model = YOLO("yolov8n.pt")

    # epochs:
    # This controls how many times the model sees the full training dataset.
    # More epochs can improve accuracy because the model gets more practice.
    # Too many epochs can cause overfitting, where the model memorizes training
    # images but performs poorly on new images.
    # Increase epochs if both training and validation performance are still improving.
    # Decrease epochs if training takes too long or validation performance gets worse.
    epochs = 20

    # batch:
    # This controls how many images are processed at one time.
    # A larger batch can train faster on a strong GPU.
    # A larger batch also uses more GPU memory.
    # Increase batch size if your GPU has enough memory.
    # Decrease batch size if you get an out-of-memory error.
    batch = 8

    # imgsz:
    # This controls the image size used during training.
    # Larger images can help detect small details in hand signs.
    # Larger images are slower and use more memory.
    # Increase image size if signs are small or hard to see.
    # Decrease image size if training is too slow or memory is limited.
    imgsz = 640

    # lr0:
    # This is the initial learning rate.
    # It controls how big each learning step is.
    # A higher learning rate can train faster but may be unstable.
    # A lower learning rate is slower but can be more stable.
    # Decrease it if the loss jumps around a lot.
    # Increase it slightly if learning is extremely slow.
    lr0 = 0.01

    # optimizer:
    # This controls how the model updates its weights during training.
    # "auto" lets Ultralytics choose a good optimizer for the model and dataset.
    # This is beginner-friendly and usually works well.
    # Later, you can try "SGD" or "AdamW" for experiments.
    optimizer = "auto"

    # patience:
    # This controls early stopping.
    # If validation performance does not improve for this many epochs,
    # training stops early.
    # A higher patience gives the model more time to improve.
    # A lower patience stops training sooner and saves time.
    patience = 10

    print("Starting training...")
    print("This may take a while depending on your dataset and hardware.")

    try:
        model.train(
            data=str(dataset_yaml_path),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer,
            patience=patience,
            device=device,
            project="outputs",
            name="training_results",
            exist_ok=True,
        )

        print("Training finished.")
        print("Results were saved in: outputs/training_results")
        print("Best model is usually saved as: outputs/training_results/weights/best.pt")
        print("You can copy best.pt into the models folder for prediction.")

    except RuntimeError as error:
        print("Training stopped because of a runtime error.")
        print("Common causes are low GPU memory or an incorrect dataset.")
        print(f"Error message: {error}")
    except Exception as error:
        print("Training stopped because an unexpected error occurred.")
        print(f"Error message: {error}")


if __name__ == "__main__":
    train_model()
