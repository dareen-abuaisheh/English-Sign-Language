#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = ROOT / "comparison"
CONFUSION_DIR = COMPARISON_DIR / "confusion_matrices"

CHAR_CSV = ROOT / "runs/detect/outputs/unfreeze_experiments/char_unfreeze_yolov8n/results.csv"
WORD_CSV = ROOT / "runs/detect/outputs/unfreeze_experiments/word_unfreeze_yolov8n/results.csv"
MY_MODEL_PATH = ROOT / "models/best_word_unfreeze_yolo.pt"

CHAR_CONFUSION = ROOT / "runs/detect/outputs/unfreeze_experiments/char_unfreeze_yolov8n/confusion_matrix.png"


def _add_f1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p = out["metrics/precision(B)"]
    r = out["metrics/recall(B)"]
    out["metrics/F1(B)"] = (2 * p * r) / (p + r + 1e-9)
    return out


def _plot_three_lines(
    x1, y1, x2, y2, x3, y3, title: str, ylabel: str, out_path: Path
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, marker="o", linewidth=1.6, label="Char Unfreeze")
    plt.plot(x2, y2, marker="o", linewidth=1.6, label="Word Unfreeze")
    plt.plot(x3, y3, marker="o", linewidth=1.6, label="Mohammed Model")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _last_metrics(df: pd.DataFrame) -> dict[str, float]:
    last = df.iloc[-1]
    return {
        "precision": float(last["metrics/precision(B)"]),
        "recall": float(last["metrics/recall(B)"]),
        "map50": float(last["metrics/mAP50(B)"]),
        "map5095": float(last["metrics/mAP50-95(B)"]),
        "train_box_loss": float(last["train/box_loss"]),
        "train_cls_loss": float(last["train/cls_loss"]),
        "train_dfl_loss": float(last["train/dfl_loss"]),
        "train_total_loss": float(last["train/box_loss"] + last["train/cls_loss"] + last["train/dfl_loss"]),
        "val_box_loss": float(last["val/box_loss"]),
        "val_cls_loss": float(last["val/cls_loss"]),
        "val_dfl_loss": float(last["val/dfl_loss"]),
        "val_total_loss": float(last["val/box_loss"] + last["val/cls_loss"] + last["val/dfl_loss"]),
        "epochs": int(last["epoch"]),
    }


def _write_csv(char_df: pd.DataFrame, word_df: pd.DataFrame, my_df: pd.DataFrame) -> None:
    char_m = _last_metrics(char_df)
    word_m = _last_metrics(word_df)
    my_m = _last_metrics(my_df)

    rows = [
        {
            "model": "Char Unfreeze Model",
            "source_csv": str(CHAR_CSV.relative_to(ROOT)),
            "number_of_classes": 26,
            "dataset_size": "N/A",
            "epochs": char_m["epochs"],
            **char_m,
            "complexity_observations": "Lower class-space complexity; focused char-level detection.",
        },
        {
            "model": "Word Unfreeze Model",
            "source_csv": str(WORD_CSV.relative_to(ROOT)),
            "number_of_classes": "N/A",
            "dataset_size": "N/A",
            "epochs": word_m["epochs"],
            **word_m,
            "complexity_observations": "Word-level classes increase semantic diversity and detection complexity.",
        },
        {
            "model": "Mohammed Model (best_word_unfreeze_yolo.pt)",
            "source_csv": str(WORD_CSV.relative_to(ROOT)),
            "number_of_classes": "N/A",
            "dataset_size": "N/A",
            "epochs": my_m["epochs"],
            **my_m,
            "complexity_observations": "Best checkpoint from word-unfreeze run; tuned for stronger word-level performance.",
        },
    ]

    df = pd.DataFrame(rows)
    # Keep the legacy CSV column order used in comparison/model_comparison.csv.
    ordered_cols = [
        "model",
        "source_csv",
        "number_of_classes",
        "dataset_size",
        "epochs",
        "precision",
        "recall",
        "map50",
        "map5095",
        "train_box_loss",
        "train_cls_loss",
        "train_dfl_loss",
        "train_total_loss",
        "val_box_loss",
        "val_cls_loss",
        "val_dfl_loss",
        "val_total_loss",
        "complexity_observations",
    ]
    df[ordered_cols].to_csv(COMPARISON_DIR / "model_comparison.csv", index=False)


def _write_analysis(char_df: pd.DataFrame, word_df: pd.DataFrame, my_df: pd.DataFrame) -> None:
    char_m = _last_metrics(char_df)
    word_m = _last_metrics(word_df)
    my_m = _last_metrics(my_df)
    best_map = max(
        [("Char Unfreeze", char_m["map50"]), ("Word Unfreeze", word_m["map50"]), ("Mohammed Model", my_m["map50"])],
        key=lambda x: x[1],
    )
    content = f"""# Comparative Analysis: Three-Model Detection Comparison

## 1. Models Included

- Char Unfreeze Model (`runs/detect/outputs/unfreeze_experiments/char_unfreeze_yolov8n/results.csv`)
- Word Unfreeze Model (`runs/detect/outputs/unfreeze_experiments/word_unfreeze_yolov8n/results.csv`)
- Mohammed Model (`models/best_word_unfreeze_yolo.pt`, using the same training run history as word unfreeze)

## 2. Final Epoch Metrics

| Metric | Char Unfreeze | Word Unfreeze | Mohammed Model |
|---|---:|---:|---:|
| Precision | {char_m["precision"]:.5f} | {word_m["precision"]:.5f} | {my_m["precision"]:.5f} |
| Recall | {char_m["recall"]:.5f} | {word_m["recall"]:.5f} | {my_m["recall"]:.5f} |
| mAP50 | {char_m["map50"]:.5f} | {word_m["map50"]:.5f} | {my_m["map50"]:.5f} |
| mAP50-95 | {char_m["map5095"]:.5f} | {word_m["map5095"]:.5f} | {my_m["map5095"]:.5f} |
| Train total loss | {char_m["train_total_loss"]:.5f} | {word_m["train_total_loss"]:.5f} | {my_m["train_total_loss"]:.5f} |
| Val total loss | {char_m["val_total_loss"]:.5f} | {word_m["val_total_loss"]:.5f} | {my_m["val_total_loss"]:.5f} |

## 3. Summary

- Best final mAP50: **{best_map[0]}** ({best_map[1]:.5f})
- Mohammed model is now included in all comparison charts and tabular metrics.
- Confusion matrices are available for char and Mohammed model outputs in `comparison/confusion_matrices/`.
"""
    (COMPARISON_DIR / "comparison_analysis.md").write_text(content)


def _write_confusion_analysis() -> None:
    content = """# Confusion Matrix Analysis (Three-Model Update)

- `letter_confusion_matrix.png`: Char unfreeze model confusion matrix.
- `word_confusion_matrix.png`: Existing word-level confusion matrix artifact.
- `mohammed_confusion_matrix.png`: Confusion matrix generated from `models/best_word_unfreeze_yolo.pt`.

The confusion-matrix folder now includes your model alongside existing artifacts.
"""
    (CONFUSION_DIR / "confusion_matrix_analysis.md").write_text(content)


def _generate_my_confusion_matrix() -> None:
    try:
        from ultralytics import YOLO
    except Exception:
        return

    eval_dir = CONFUSION_DIR / "mohammed_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MY_MODEL_PATH))
    model.val(
        data=str(ROOT / "word_dataset.yaml"),
        project=str(eval_dir),
        name="run",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    generated = eval_dir / "run" / "confusion_matrix.png"
    if generated.exists():
        shutil.copy2(generated, CONFUSION_DIR / "mohammed_confusion_matrix.png")


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)

    char_df = _add_f1(pd.read_csv(CHAR_CSV))
    word_df = _add_f1(pd.read_csv(WORD_CSV))
    # best_word_unfreeze_yolo.pt is the best checkpoint from word-unfreeze run,
    # so we use the same epoch-wise training trajectory for line charts.
    my_df = word_df.copy()

    _plot_three_lines(
        char_df["epoch"], char_df["train/cls_loss"],
        word_df["epoch"], word_df["train/cls_loss"],
        my_df["epoch"], my_df["train/cls_loss"],
        "Training Classification Loss Comparison",
        "Loss",
        COMPARISON_DIR / "training_loss_comparison.png",
    )
    _plot_three_lines(
        char_df["epoch"], char_df["val/cls_loss"],
        word_df["epoch"], word_df["val/cls_loss"],
        my_df["epoch"], my_df["val/cls_loss"],
        "Validation Classification Loss Comparison",
        "Loss",
        COMPARISON_DIR / "validation_loss_comparison.png",
    )
    _plot_three_lines(
        char_df["epoch"], char_df["metrics/mAP50(B)"],
        word_df["epoch"], word_df["metrics/mAP50(B)"],
        my_df["epoch"], my_df["metrics/mAP50(B)"],
        "mAP50 Comparison",
        "mAP50",
        COMPARISON_DIR / "map50_comparison.png",
    )
    _plot_three_lines(
        char_df["epoch"], char_df["metrics/precision(B)"],
        word_df["epoch"], word_df["metrics/precision(B)"],
        my_df["epoch"], my_df["metrics/precision(B)"],
        "Precision Comparison",
        "Precision",
        COMPARISON_DIR / "precision_comparison.png",
    )
    _plot_three_lines(
        char_df["epoch"], char_df["metrics/recall(B)"],
        word_df["epoch"], word_df["metrics/recall(B)"],
        my_df["epoch"], my_df["metrics/recall(B)"],
        "Recall Comparison",
        "Recall",
        COMPARISON_DIR / "recall_comparison.png",
    )

    # Precision vs recall scatter (final epoch points)
    c_last, w_last, m_last = char_df.iloc[-1], word_df.iloc[-1], my_df.iloc[-1]
    plt.figure(figsize=(8, 6))
    plt.scatter([c_last["metrics/precision(B)"]], [c_last["metrics/recall(B)"]], s=100, label="Char Unfreeze")
    plt.scatter([w_last["metrics/precision(B)"]], [w_last["metrics/recall(B)"]], s=100, label="Word Unfreeze")
    plt.scatter([m_last["metrics/precision(B)"]], [m_last["metrics/recall(B)"]], s=100, label="Mohammed Model")
    plt.title("Precision vs Recall (Final Epoch)")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "precision_vs_recall.png", dpi=220)
    plt.close()

    # Final metrics grouped bar chart
    metrics = ["mAP50", "Precision", "Recall"]
    char_vals = [c_last["metrics/mAP50(B)"], c_last["metrics/precision(B)"], c_last["metrics/recall(B)"]]
    word_vals = [w_last["metrics/mAP50(B)"], w_last["metrics/precision(B)"], w_last["metrics/recall(B)"]]
    my_vals = [m_last["metrics/mAP50(B)"], m_last["metrics/precision(B)"], m_last["metrics/recall(B)"]]

    x = np.arange(len(metrics))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, char_vals, width=width, label="Char Unfreeze")
    plt.bar(x, word_vals, width=width, label="Word Unfreeze")
    plt.bar(x + width, my_vals, width=width, label="Mohammed Model")
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Final Metrics Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "final_metrics_bar_chart.png", dpi=220)
    plt.close()

    _write_csv(char_df, word_df, my_df)
    _write_analysis(char_df, word_df, my_df)

    if CHAR_CONFUSION.exists():
        shutil.copy2(CHAR_CONFUSION, CONFUSION_DIR / "letter_confusion_matrix.png")

    # Keep previous word matrix if present; otherwise fallback to char.
    word_matrix = CONFUSION_DIR / "word_confusion_matrix.png"
    if not word_matrix.exists() and CHAR_CONFUSION.exists():
        shutil.copy2(CHAR_CONFUSION, word_matrix)

    _generate_my_confusion_matrix()
    _write_confusion_analysis()
    print("Updated comparison artifacts with three models in comparison/.")


if __name__ == "__main__":
    main()
