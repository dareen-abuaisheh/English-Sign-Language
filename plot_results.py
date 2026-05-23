"""
plot_results.py

This file creates multiple professional plots for:
1. Letter Detection Model
2. Word Detection Model

The plots are useful for:
- Reports
- Presentations
- Performance analysis
"""

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# LOAD CSV FILES
# ---------------------------------------------------

letter_results = pd.read_csv(
    "/home/dareen/sign-language/runs/detect/outputs/training_results/results.csv"
)

word_results = pd.read_csv(
    "/home/dareen/sign-language/runs/detect/outputs/word_training_results/results.csv"
)

# ---------------------------------------------------
# CREATE OUTPUT FOLDER
# ---------------------------------------------------

import os

os.makedirs("/home/dareen/sign-language/outputs/report_plots", exist_ok=True)

# ===================================================
# 1. mAP50 COMPARISON
# ===================================================

plt.figure(figsize=(10, 6))

plt.plot(
    letter_results["epoch"],
    letter_results["metrics/mAP50(B)"],
    label="Letter Detection"
)

plt.plot(
    word_results["epoch"],
    word_results["metrics/mAP50(B)"],
    label="Word Detection"
)

plt.title("mAP50 Comparison")

plt.xlabel("Epoch")

plt.ylabel("mAP50")

plt.legend()

plt.grid(True)

plt.savefig("/home/dareen/sign-language/outputs/report_plots/map50_comparison.png")

plt.close()

# ===================================================
# 2. PRECISION COMPARISON
# ===================================================

plt.figure(figsize=(10, 6))

plt.plot(
    letter_results["epoch"],
    letter_results["metrics/precision(B)"],
    label="Letter Detection"
)

plt.plot(
    word_results["epoch"],
    word_results["metrics/precision(B)"],
    label="Word Detection"
)

plt.title("Precision Comparison")

plt.xlabel("Epoch")

plt.ylabel("Precision")

plt.legend()

plt.grid(True)

plt.savefig("/home/dareen/sign-language/outputs/report_plots/precision_comparison.png")

plt.close()

# ===================================================
# 3. RECALL COMPARISON
# ===================================================

plt.figure(figsize=(10, 6))

plt.plot(
    letter_results["epoch"],
    letter_results["metrics/recall(B)"],
    label="Letter Detection"
)

plt.plot(
    word_results["epoch"],
    word_results["metrics/recall(B)"],
    label="Word Detection"
)

plt.title("Recall Comparison")

plt.xlabel("Epoch")

plt.ylabel("Recall")

plt.legend()

plt.grid(True)

plt.savefig("/home/dareen/sign-language/outputs/report_plots/recall_comparison.png")

plt.close()

# ===================================================
# 4. TRAINING LOSS COMPARISON
# ===================================================

plt.figure(figsize=(10, 6))

plt.plot(
    letter_results["epoch"],
    letter_results["train/cls_loss"],
    label="Letter Train Loss"
)

plt.plot(
    word_results["epoch"],
    word_results["train/cls_loss"],
    label="Word Train Loss"
)

plt.title("Training Classification Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("/home/dareen/sign-language/outputs/report_plots/train_loss_comparison.png")

plt.close()

# ===================================================
# 5. VALIDATION LOSS COMPARISON
# ===================================================

plt.figure(figsize=(10, 6))

plt.plot(
    letter_results["epoch"],
    letter_results["val/cls_loss"],
    label="Letter Validation Loss"
)

plt.plot(
    word_results["epoch"],
    word_results["val/cls_loss"],
    label="Word Validation Loss"
)

plt.title("Validation Classification Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("/home/dareen/sign-language/outputs/report_plots/validation_loss_comparison.png")

plt.close()

# ===================================================
# 6. FINAL MODEL COMPARISON BAR CHART
# ===================================================

letter_map = letter_results["metrics/mAP50(B)"].iloc[-1]
word_map = word_results["metrics/mAP50(B)"].iloc[-1]

letter_precision = letter_results["metrics/precision(B)"].iloc[-1]
word_precision = word_results["metrics/precision(B)"].iloc[-1]

letter_recall = letter_results["metrics/recall(B)"].iloc[-1]
word_recall = word_results["metrics/recall(B)"].iloc[-1]

metrics = ["mAP50", "Precision", "Recall"]

letter_values = [
    letter_map,
    letter_precision,
    letter_recall
]

word_values = [
    word_map,
    word_precision,
    word_recall
]

x = range(len(metrics))

plt.figure(figsize=(10, 6))

plt.bar(
    [i - 0.2 for i in x],
    letter_values,
    width=0.4,
    label="Letter Detection"
)

plt.bar(
    [i + 0.2 for i in x],
    word_values,
    width=0.4,
    label="Word Detection"
)

plt.xticks(x, metrics)

plt.ylabel("Score")

plt.title("Final Model Performance Comparison")

plt.legend()

plt.grid(True)

plt.savefig("/home/dareen/sign-language/outputs/report_plots/final_comparison_bar_chart.png")

plt.close()

# ---------------------------------------------------
# DONE
# ---------------------------------------------------

print("All report plots generated successfully.")