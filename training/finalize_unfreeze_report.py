#!/usr/bin/env python3
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd


CHAR_DIR = Path("runs/detect/outputs/unfreeze_experiments/char_unfreeze_yolov8n")
WORD_DIR = Path("runs/detect/outputs/unfreeze_experiments/word_unfreeze_yolov8n")
OUT_DIR = Path("outputs/unfreeze_experiments/charts")
SUMMARY_PATH = Path("outputs/unfreeze_experiments/summary.txt")
BEST_WORD_SRC = WORD_DIR / "weights" / "best.pt"
BEST_WORD_DST = Path("models/best_word_unfreeze_yolo.pt")


def make_plot(char_df: pd.DataFrame, word_df: pd.DataFrame, col: str, ylabel: str, title: str, out_name: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(char_df["epoch"], char_df[col], label="char-level", marker="o", linewidth=1.5)
    plt.plot(word_df["epoch"], word_df[col], label="word-level", marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=200)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_WORD_DST.parent.mkdir(parents=True, exist_ok=True)

    char_df = pd.read_csv(CHAR_DIR / "results.csv")
    word_df = pd.read_csv(WORD_DIR / "results.csv")

    # User asked for train loss, F1, and accuracy improvement.
    # YOLO CSV does not expose F1 directly, so we compute it from precision/recall.
    for df in (char_df, word_df):
        p = df["metrics/precision(B)"]
        r = df["metrics/recall(B)"]
        df["metrics/F1(B)"] = (2 * p * r) / (p + r + 1e-9)

    make_plot(
        char_df,
        word_df,
        "train/box_loss",
        "train/box_loss",
        "Train Loss Improvement Over Epochs",
        "train_loss_comparison.png",
    )
    make_plot(
        char_df,
        word_df,
        "metrics/F1(B)",
        "F1",
        "F1 Improvement Over Epochs",
        "f1_comparison.png",
    )
    make_plot(
        char_df,
        word_df,
        "metrics/mAP50(B)",
        "mAP50",
        "Accuracy Proxy Improvement Over Epochs (mAP50)",
        "accuracy_map50_comparison.png",
    )

    shutil.copy2(BEST_WORD_SRC, BEST_WORD_DST)

    def line(df: pd.DataFrame, label: str) -> str:
        f1_start = float(df.iloc[0]["metrics/F1(B)"])
        f1_end = float(df.iloc[-1]["metrics/F1(B)"])
        acc_start = float(df.iloc[0]["metrics/mAP50(B)"])
        acc_end = float(df.iloc[-1]["metrics/mAP50(B)"])
        loss_start = float(df.iloc[0]["train/box_loss"])
        loss_end = float(df.iloc[-1]["train/box_loss"])
        return (
            f"{label}: "
            f"loss {loss_start:.4f}->{loss_end:.4f} ({loss_end-loss_start:+.4f}), "
            f"F1 {f1_start:.4f}->{f1_end:.4f} ({f1_end-f1_start:+.4f}), "
            f"mAP50 {acc_start:.4f}->{acc_end:.4f} ({acc_end-acc_start:+.4f})"
        )

    summary = "\n".join(
        [
            "Unfreeze YOLO Summary",
            line(char_df, "char-level"),
            line(word_df, "word-level"),
            f"best word checkpoint: {BEST_WORD_DST}",
            f"charts: {OUT_DIR}",
            "",
            "Note: word-level run was manually stopped after collecting meaningful trend data.",
        ]
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
