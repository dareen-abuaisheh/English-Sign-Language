from roboflow import Roboflow

# Connect to Roboflow
rf = Roboflow(api_key="W9QoCQ2NXRk9L3MQUw5f")

# Open project
project = rf.workspace("david-lee-d0rhs").project("american-sign-language-letters")

# Open dataset version
version = project.version(1)

# Download dataset in YOLOv8 format
dataset = version.download(
    "yolov8",
    location="/home/dareen/sign-language/dataset"
)

print("Dataset downloaded successfully!")