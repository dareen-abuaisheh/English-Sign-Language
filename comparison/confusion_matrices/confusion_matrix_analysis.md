# Confusion Matrix Analysis

## Evaluation Setup

The confusion matrices were generated using real YOLOv8 predictions from the trained best weights. The letter model was evaluated on `letter_dataset/test/images`, and the mixed letter-word model was evaluated on `word_dataset/test/images`. Predictions were matched to ground-truth boxes using an IoU threshold of `0.5` and a confidence threshold of `0.25`. The `background` row or column represents missed detections or unmatched false-positive predictions.

## Letter Detection Model

The letter confusion matrix shows how well the model separates the 26 ASL alphabet classes. Because the class space is limited to letters only, the model has lower ambiguity than the mixed model. Strong diagonal values indicate correct classification, while off-diagonal values show letters that were confused with other letters or missed.

Top observed letter-model confusions:

- `background` was predicted as `N`: 3 time(s)
- `C` was predicted as `F`: 1 time(s)
- `G` was predicted as `H`: 1 time(s)
- `I` was predicted as `D`: 1 time(s)
- `K` was predicted as `V`: 1 time(s)
- `K` was predicted as `background`: 1 time(s)
- `M` was predicted as `background`: 1 time(s)
- `N` was predicted as `A`: 1 time(s)

## Mixed Dataset / Word Detection Model

The mixed model contains both alphabet signs and word-level signs. This creates a more difficult recognition problem because the model must choose between many more classes, and some word gestures contain hand shapes that visually resemble alphabet signs. As a result, the mixed confusion matrix is expected to show more scattered off-diagonal confusion than the letter-only model.

Top observed mixed-model confusions:

- `background` was predicted as `ingredients`: 11 time(s)
- `background` was predicted as `hot`: 4 time(s)
- `background` was predicted as `straw`: 3 time(s)
- `K` was predicted as `V`: 2 time(s)
- `N` was predicted as `A`: 2 time(s)
- `french fries` was predicted as `background`: 2 time(s)
- `straw` was predicted as `background`: 2 time(s)
- `background` was predicted as `drink`: 2 time(s)

## Comparison Between Models

The letter model is easier to interpret because it focuses on a smaller and more controlled 26-class alphabet task. The mixed model is more realistic for communication use, but it is more challenging because class count, gesture similarity, and dataset diversity all increase the possibility of confusion. Similar hand poses, overlapping finger configurations, and inconsistent lighting or background conditions can cause the model to assign a visually similar but incorrect label.

## Practical Interpretation

A clean diagonal pattern means the model usually predicts the correct class. Off-diagonal cells show specific confusion cases that should be reviewed when improving the dataset. For ASL detection, confusion often happens when two signs share similar hand shape, orientation, or location in the frame. Improving lighting consistency, adding more varied examples, and balancing difficult classes can reduce these errors.

## Generated Figures

- `letter_confusion_matrix.png`
- `word_confusion_matrix.png`
