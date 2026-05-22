"""
predict.py

This file runs ASL detection on one image using a trained YOLOv8 model.

Inference means using a trained model to make predictions on new data.
Here, the model looks at one image, finds hand signs, draws bounding boxes,
and displays the confidence score for each prediction.

Run:
    python predict.py --image inference/sample.jpg --model models/best.pt
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def read_arguments():
    """Read simple command-line arguments from the user."""

    parser = argparse.ArgumentParser(description="Run YOLOv8 ASL prediction on one image.")

    parser.add_argument(
        "--image",
        default="inference/sample.jpeg",
        help="Path to the image you want to test.",
    )

    parser.add_argument(
        "--model",
        default="runs/detect/outputs/training_results/weights/best.pt",
        help="Path to the trained YOLOv8 model file.",
    )

    return parser.parse_args()


def check_prediction_files(image_path, model_path):
    """Check that the image and model files exist before prediction."""

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Train the model first, then place best.pt inside the models folder.")
        return False

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        print("Place a test image in the inference folder or provide another path.")
        return False

    return True


def predict_image(image_path, model_path):
    """Load the trained model, run prediction, and show the result."""

    if not check_prediction_files(image_path, model_path):
        return

    try:
        print("Loading trained model...")
        model = YOLO(str(model_path))

        print("Reading image...")
        image = cv2.imread(str(image_path))

        if image is None:
            print("Error: OpenCV could not read the image.")
            print("Check that the file is a valid image format such as JPG or PNG.")
            return

        print("Running prediction...")

        # conf:
        # This is the confidence threshold.
        # Predictions below this value are ignored.
        # Increase it if the model shows too many weak or incorrect boxes.
        # Decrease it if the model misses signs that it should detect.
        confidence_threshold = 0.25

        results = model.predict(image, conf=confidence_threshold)

        # result.plot() draws bounding boxes, labels, and confidence scores.
        # It returns a normal image array that OpenCV can display.
        result_image = results[0].plot()

        output_folder = Path("outputs/results")
        output_folder.mkdir(parents=True, exist_ok=True)

        output_path = output_folder / "prediction_result.jpg"
        cv2.imwrite(str(output_path), result_image)

        print(f"Prediction image saved to: {output_path}")
        print("Press any key in the image window to close it.")

        cv2.imshow("ASL Detection Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as error:
        print("Prediction failed because an error occurred.")
        print(f"Error message: {error}")


if __name__ == "__main__":
    arguments = read_arguments()
    predict_image(Path(arguments.image), Path(arguments.model))
