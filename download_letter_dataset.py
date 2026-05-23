"""
download_letter_dataset.py

This optional helper downloads the ASL LETTER dataset from Roboflow.
It is separate from the word module and saves files into letter_dataset/.

Run:
    python download_letter_dataset.py
"""

from roboflow import Roboflow


# Roboflow account key.
# For public reports, avoid sharing this value.
API_KEY = "W9QoCQ2NXRk9L3MQUw5f"

# Keep the letter dataset separate from the word dataset.
LETTER_DATASET_LOCATION = "letter_dataset"


try:
    print("Connecting to Roboflow for the letter dataset...")
    roboflow = Roboflow(api_key=API_KEY)

    print("Opening the letter dataset project...")
    project = roboflow.workspace("david-lee-d0rhs").project("american-sign-language-letters")

    print("Downloading the letter dataset in YOLOv8 format...")
    dataset = project.version(1).download("yolov8", location=LETTER_DATASET_LOCATION)

    print("Letter dataset downloaded successfully!")
    print(f"Saved at: {dataset.location}")

except Exception as error:
    print("Letter dataset download failed.")
    print("Check your internet connection, API key, project name, and version number.")
    print(f"Error message: {error}")
