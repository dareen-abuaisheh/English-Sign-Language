# Inference Folder

This folder can store images or videos used for testing the trained model.

Example image path:

```text
inference/sample.jpg
```

Run image prediction with:

```bash
python predict.py --image inference/sample.jpg --model models/best.pt
```
