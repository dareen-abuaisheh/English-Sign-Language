"""
webcam_detection.py

This file runs real-time ASL detection using a webcam.

The webcam gives us video frames one by one.
Each frame is sent to the YOLOv8 model.
The model predicts ASL signs, and OpenCV displays the result on the screen.

Run:
    python webcam_detection.py --model models/best.pt

Press q to quit the webcam window safely.
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def read_arguments():
    """Read simple command-line arguments from the user."""

    parser = argparse.ArgumentParser(description="Run real-time ASL detection with a webcam.")

    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Path to the trained YOLOv8 model file.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam number. Use 0 for the default webcam.",
    )

    return parser.parse_args()


def check_model_file(model_path):
    """Check that the trained model exists."""

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Train the model first, then place best.pt inside the models folder.")
        return False

    return True


def run_webcam_detection(model_path, camera_number):
    """Open the webcam and run YOLOv8 prediction on each frame."""

    if not check_model_file(model_path):
        return

    try:
        print("Loading trained model...")
        model = YOLO(str(model_path))
    except Exception as error:
        print("Error: Could not load the model.")
        print(f"Error message: {error}")
        return

    print("Opening webcam...")
    webcam = cv2.VideoCapture(camera_number)

    if not webcam.isOpened():
        print("Error: Webcam could not be opened.")
        print("Check that your webcam is connected and not being used by another app.")
        return

    # conf:
    # This is the confidence threshold for real-time prediction.
    # A higher value shows only stronger predictions.
    # A lower value shows more predictions, but some may be wrong.
    confidence_threshold = 0.25

    print("Webcam detection started.")
    print("Press q to quit.")

    while True:
        success, frame = webcam.read()

        if not success:
            print("Error: Could not read a frame from the webcam.")
            break

        try:
            # Run YOLOv8 on the current webcam frame.
            results = model.predict(frame, conf=confidence_threshold, verbose=False)

            # Draw bounding boxes, class names, and confidence scores.
            frame_with_predictions = results[0].plot()

            cv2.imshow("Real-Time ASL Detection", frame_with_predictions)

        except Exception as error:
            print("Prediction error on webcam frame.")
            print(f"Error message: {error}")
            break

        # waitKey(1) waits 1 millisecond for a key press.
        # If the user presses q, the loop stops.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()
    print("Webcam closed safely.")


if __name__ == "__main__":
    arguments = read_arguments()
    run_webcam_detection(Path(arguments.model), arguments.camera)
