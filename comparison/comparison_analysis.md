# Comparative Analysis: Letter-Only Detection Model vs Mixed Dataset Detection Model

## 1. Introduction

This section compares the updated Letter-Only Detection Model with the Mixed Dataset Detection Model. The letter model was retrained for 50 epochs and the comparison below uses the final recorded epoch from the updated YOLO results file. The mixed model uses the existing word/mixed dataset training results.

The metric values were extracted from the real YOLO training logs:

- Letter model source: `runs/detect/outputs/letter_training_results/results.csv`
- Mixed model source: `runs/detect/outputs/word_training_results/results.csv`

## 2. Numerical Comparison

| Metric | Letter-Only Model | Mixed Letters + Words Model |
|---|---:|---:|
| Number of classes | 26 | 106 |
| Dataset size | 1728 | 23350 |
| Epochs | 50 | 48 |
| Precision | 0.93218 | 0.96172 |
| Recall | 0.89784 | 0.96392 |
| mAP50 | 0.96179 | 0.97259 |
| mAP50-95 | 0.80129 | 0.73395 |
| Final training total loss | 1.81001 | 1.45487 |
| Final validation total loss | 2.50874 | 2.52090 |

## 3. Performance Analysis

The Letter-Only Detection Model focuses on a smaller and more controlled class space containing only alphabet signs. This makes the prediction task cleaner because the detector only needs to separate 26 letter classes. The updated 50-epoch training results show the final measured performance of this focused recognition task.

The Mixed Dataset Detection Model contains both letters and word signs. Although it has a larger dataset, it also has a much larger class space. This increases classification difficulty because some ASL word gestures may contain hand shapes that visually resemble alphabet signs. For this reason, the mixed model can occasionally predict letters during word inference when a word image contains a letter-like pose.

## 4. Complexity Analysis

The letter-only model has lower ambiguity because it learns a narrower alphabet-only problem. This is useful for stable spelling-based ASL detection and educational demonstrations.

The mixed model is more complex because it increases the number of classes from 26 to 106. A larger class space makes the decision boundary harder to learn. Dataset diversity can improve generalization, but it can also increase confusion when different signs share similar hand shapes, orientations, or bounding-box positions.

## 5. Real-Time Performance Discussion

Both models use YOLOv8n, which is appropriate for real-time detection because it performs localization and classification in one efficient detection pipeline. The letter-only model is expected to provide more stable real-time predictions because it has fewer classes and lower gesture ambiguity.

The mixed model is more realistic for communication-oriented use because it supports word-level signs, but it is also more challenging in live webcam use. Real-time usability should therefore be judged not only by mAP values but also by prediction stability, confidence consistency, lighting conditions, background complexity, and similarity between gestures.

## 6. Generalization Discussion

The mixed dataset is larger and more diverse, which can help the model learn broader hand-sign patterns. However, broader diversity also increases the chance of inter-class confusion. The letter-only model is more specialized and therefore easier to control, but it is less scalable for full communication because it only supports alphabet-level recognition.

## 7. Final Analytical Conclusion

The updated comparison shows the tradeoff between a focused letter-only detector and a broader mixed letter-word detector. The letter-only model remains easier to interpret and generally cleaner for alphabet recognition, while the mixed model represents a more difficult but more realistic recognition scenario. A practical ASL system could benefit from combining both approaches with temporal modeling, stronger dataset balancing, and further real-world webcam testing.
