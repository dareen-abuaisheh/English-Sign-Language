"""
predict_letter.py

This file predicts ASL letters from one image using the trained letter model.

Run:
    python predict_letter.py --image inference/sample.jpeg --model models/letter_best.pt
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def read_arguments():
    """Read simple command-line arguments."""

    parser = argparse.ArgumentParser(description="Predict ASL letters from one image.")
    parser.add_argument("--image", default="inference/sample.jpeg")
    parser.add_argument("--model", default="models/letter_best.pt")
    return parser.parse_args()


def predict_letter_image(image_path, model_path):
    """Load the letter model, predict one image, display it, and save it."""

    if not model_path.exists():
        print(f"Error: Letter model not found: {model_path}")
        print("Train the letter model and copy best.pt to models/letter_best.pt")
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
        # Increase it to show fewer but stronger detections.
        # Decrease it if the model misses signs.
        confidence_threshold = 0.25

        results = model.predict(image, conf=confidence_threshold)
        result_image = results[0].plot()

        output_folder = Path("outputs/letter_predictions")
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / "letter_prediction_result.jpg"
        cv2.imwrite(str(output_path), result_image)

        print(f"Letter prediction saved to: {output_path}")
        print("Press any key to close the image window.")

        cv2.imshow("ASL Letter Detection", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as error:
        print("Letter prediction failed.")
        print(f"Error message: {error}")


if __name__ == "__main__":
    args = read_arguments()
    predict_letter_image(Path(args.image), Path(args.model))
