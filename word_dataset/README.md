# Word Dataset Folder

This folder contains the ASL word dataset in YOLO format.

Each word is treated as one class, for example `hello`, `water`, `please`, or `thank-you`.

Use this structure:

```text
word_dataset/
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

Example matching image and label:

```text
word_dataset/train/images/hello_001.jpg
word_dataset/train/labels/hello_001.txt
```
