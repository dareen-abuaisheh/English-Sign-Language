# Inference Folder

This folder can store sample images or videos used for testing either module.

Letter prediction example:

```bash
python predict_letter.py --image inference/sample.jpeg --model models/letter_best.pt
```

Word prediction example:

```bash
python predict_word.py --image inference/sample.jpeg --model models/word_best.pt
```
