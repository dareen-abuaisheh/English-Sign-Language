# Dataset Folder

This folder contains the custom American Sign Language dataset in YOLO format.

Use this structure:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Place images inside the `images` folders.
Place matching `.txt` annotation files inside the `labels` folders.

Example:

```text
dataset/train/images/A_001.jpg
dataset/train/labels/A_001.txt
```

The image name and label name must match.
