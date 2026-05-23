"""
webcam_word_detection.py

This file runs real-time ASL WORD detection using a webcam.

The webcam provides frames one by one. YOLOv8 predicts the word shown in each
frame, then OpenCV draws bounding boxes, labels, and confidence scores.

Run:
    python webcam_word_detection.py --model models/word_best.pt

Press q to quit safely.
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def read_arguments():
    """Read simple command-line arguments."""

    parser = argparse.ArgumentParser(description="Run webcam ASL word detection.")
    parser.add_argument("--model", default="models/word_best.pt")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def run_word_webcam(model_path, camera_number):
    """Open webcam and run word detection on each frame."""

    if not model_path.exists():
        print(f"Error: Word model not found: {model_path}")
        return

    try:
        model = YOLO(str(model_path))
    except Exception as error:
        print("Could not load the word model.")
        print(f"Error message: {error}")
        return

    webcam = cv2.VideoCapture(camera_number)

    if not webcam.isOpened():
        print("Error: Webcam could not be opened.")
        print("Check the webcam connection or camera number.")
        return

    # Confidence threshold:
    # Higher value means fewer detections, but usually more reliable.
    # Lower value means more detections, but some may be wrong.
    confidence_threshold = 0.25

    print("Word webcam detection started. Press q to quit.")

    while True:
        success, frame = webcam.read()

        if not success:
            print("Error: Could not read from webcam.")
            break

        try:
            results = model.predict(frame, conf=confidence_threshold, verbose=False)
            frame_with_boxes = results[0].plot()
            cv2.imshow("ASL Word Detection", frame_with_boxes)
        except Exception as error:
            print("Prediction error during word webcam detection.")
            print(f"Error message: {error}")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()
    print("Word webcam closed safely.")


if __name__ == "__main__":
    args = read_arguments()
    run_word_webcam(Path(args.model), args.camera)
