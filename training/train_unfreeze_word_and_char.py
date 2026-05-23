#!/usr/bin/env python3
"""
Run two YOLOv8 transfer-learning experiments:
1) Character-level ASL dataset (dataset.yaml)
2) Word-level ASL dataset (dataset_word_level/data.yaml)

Both runs use partial freezing (freeze=10), which keeps early backbone layers fixed
and unfreezes later layers + detection head.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ultralytics import YOLO


PROJECT_DIR = Path("outputs/unfreeze_experiments")
CHARTS_DIR = PROJECT_DIR / "charts"
MODELS_DIR = Path("models")


def train_one(data_yaml: str, run_name: str, epochs: int) -> Path:
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=32,
        device=0 if torch.cuda.is_available() else "cpu",
        optimizer="AdamW",
        lr0=0.003,
        lrf=0.01,
        freeze=10,
        warmup_epochs=3,
        cos_lr=True,
        patience=20,
        pretrained=True,
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=True,
        workers=8,
        plots=True,
    )
    return Path(results.save_dir)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of these columns: {candidates}")


def make_charts(char_csv: Path, word_csv: Path) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    char_df = pd.read_csv(char_csv)
    word_df = pd.read_csv(word_csv)

    epoch_col_char = _find_col(char_df, ["epoch"])
    epoch_col_word = _find_col(word_df, ["epoch"])

    train_loss_col_char = _find_col(
        char_df, ["train/box_loss", "train/dfl_loss", "train/cls_loss"]
    )
    train_loss_col_word = _find_col(
        word_df, ["train/box_loss", "train/dfl_loss", "train/cls_loss"]
    )
    f1_col_char = _find_col(char_df, ["metrics/F1(B)", "metrics/F1"])
    f1_col_word = _find_col(word_df, ["metrics/F1(B)", "metrics/F1"])
    acc_col_char = _find_col(char_df, ["metrics/mAP50(B)", "metrics/mAP50"])
    acc_col_word = _find_col(word_df, ["metrics/mAP50(B)", "metrics/mAP50"])

    # Train loss
    plt.figure(figsize=(9, 5))
    plt.plot(char_df[epoch_col_char], char_df[train_loss_col_char], label="char-level")
    plt.plot(word_df[epoch_col_word], word_df[train_loss_col_word], label="word-level")
    plt.title("Training Loss (Box Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "train_loss_comparison.png", dpi=200)
    plt.close()

    # F1
    plt.figure(figsize=(9, 5))
    plt.plot(char_df[epoch_col_char], char_df[f1_col_char], label="char-level")
    plt.plot(word_df[epoch_col_word], word_df[f1_col_word], label="word-level")
    plt.title("F1 Score Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "f1_comparison.png", dpi=200)
    plt.close()

    # "Accuracy" proxy (mAP50 for object detection)
    plt.figure(figsize=(9, 5))
    plt.plot(char_df[epoch_col_char], char_df[acc_col_char], label="char-level")
    plt.plot(word_df[epoch_col_word], word_df[acc_col_word], label="word-level")
    plt.title("Accuracy Proxy Over Epochs (mAP50)")
    plt.xlabel("Epoch")
    plt.ylabel("mAP50")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "accuracy_map50_comparison.png", dpi=200)
    plt.close()


def summarize(csv_path: Path, label: str) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    f1_col = _find_col(df, ["metrics/F1(B)", "metrics/F1"])
    acc_col = _find_col(df, ["metrics/mAP50(B)", "metrics/mAP50"])
    loss_col = _find_col(df, ["train/box_loss", "train/dfl_loss", "train/cls_loss"])

    first = df.iloc[0]
    last = df.iloc[-1]
    out = {
        "label": label,
        "f1_start": float(first[f1_col]),
        "f1_end": float(last[f1_col]),
        "map50_start": float(first[acc_col]),
        "map50_end": float(last[acc_col]),
        "loss_start": float(first[loss_col]),
        "loss_end": float(last[loss_col]),
    }
    out["f1_delta"] = out["f1_end"] - out["f1_start"]
    out["map50_delta"] = out["map50_end"] - out["map50_start"]
    out["loss_delta"] = out["loss_end"] - out["loss_start"]
    return out


def main() -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    char_dir = train_one("dataset.yaml", "char_unfreeze_yolov8n", epochs=20)
    word_dir = train_one("dataset_word_level/data.yaml", "word_unfreeze_yolov8n", epochs=20)

    char_csv = char_dir / "results.csv"
    word_csv = word_dir / "results.csv"
    make_charts(char_csv, word_csv)

    best_word_src = word_dir / "weights" / "best.pt"
    best_word_dst = MODELS_DIR / "best_word_unfreeze_yolo.pt"
    shutil.copy2(best_word_src, best_word_dst)

    char_stats = summarize(char_csv, "char-level")
    word_stats = summarize(word_csv, "word-level")

    report_lines = [
        "Unfreeze YOLO Training Summary",
        f"char run dir: {char_dir}",
        f"word run dir: {word_dir}",
        f"best word model: {best_word_dst}",
        "",
        f"char F1: {char_stats['f1_start']:.4f} -> {char_stats['f1_end']:.4f} (delta {char_stats['f1_delta']:+.4f})",
        f"char mAP50: {char_stats['map50_start']:.4f} -> {char_stats['map50_end']:.4f} (delta {char_stats['map50_delta']:+.4f})",
        f"char train loss: {char_stats['loss_start']:.4f} -> {char_stats['loss_end']:.4f} (delta {char_stats['loss_delta']:+.4f})",
        "",
        f"word F1: {word_stats['f1_start']:.4f} -> {word_stats['f1_end']:.4f} (delta {word_stats['f1_delta']:+.4f})",
        f"word mAP50: {word_stats['map50_start']:.4f} -> {word_stats['map50_end']:.4f} (delta {word_stats['map50_delta']:+.4f})",
        f"word train loss: {word_stats['loss_start']:.4f} -> {word_stats['loss_end']:.4f} (delta {word_stats['loss_delta']:+.4f})",
        "",
        f"charts: {CHARTS_DIR}",
    ]
    report_path = PROJECT_DIR / "summary.txt"
    report_path.write_text("\n".join(report_lines))
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
