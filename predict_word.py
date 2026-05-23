"""
predict_word.py

This file predicts ASL words from one image using the trained word model.

Inference means using a trained model to make predictions on new images.
This script saves the result image instead of opening a window, because remote
GPU servers often do not support cv2.imshow().

Run:
    python predict_word.py --image inference/sample.jpeg --model models/word_best.pt
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def read_arguments():
    """Read simple command-line arguments."""

    parser = argparse.ArgumentParser(description="Predict ASL words from one image.")
    parser.add_argument("--image", default="inference/sample.jpeg")
    parser.add_argument("--model", default="models/word_best.pt")
    return parser.parse_args()


def predict_word_image(image_path, model_path):
    """Load the word model, predict one image, and save the result."""

    if not model_path.exists():
        print(f"Error: Word model not found: {model_path}")
        print("Train the word model and copy best.pt to models/word_best.pt")
        return

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    try:
        model = YOLO(str(model_path))
        image = cv2.imread(str(image_path))

        if image is None:
            print("Error: OpenCV could not read the image.")
            return

        # Confidence threshold:
        # Predictions below this value are ignored.
        # Increase it to show fewer but stronger word detections.
        # Decrease it if the model misses words.
        confidence_threshold = 0.25

        results = model.predict(image, conf=confidence_threshold)
        result_image = results[0].plot()

        output_folder = Path("outputs/word_predictions")
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / "word_prediction_result.jpg"
        cv2.imwrite(str(output_path), result_image)

        print(f"Word prediction saved to: {output_path}")

    except Exception as error:
        print("Word prediction failed.")
        print(f"Error message: {error}")


if __name__ == "__main__":
    args = read_arguments()
    predict_word_image(Path(args.image), Path(args.model))
