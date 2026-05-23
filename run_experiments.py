"""
run_experiments.py

Automated experiment pipeline for the YOLOv8 ASL LETTER detection project.

This script runs small controlled experiments using only the letter dataset.
It is designed to protect the original project results.

Important safety rules followed by this script:
- It does NOT write into the original outputs/ folder.
- It does NOT write into the original runs/ folder.
- It does NOT modify models/ or any existing best.pt files.
- It stores YOLO experiment outputs in a temporary folder.
- It deletes the temporary folder at the end.
- It keeps only three final experiment files:
    1. experiment_summary.md
    2. comparison_results.csv
    3. comparison_plot.png

Run:
    python run_experiments.py
"""

import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ultralytics import YOLO


# -----------------------------
# Important project paths
# -----------------------------

# The script uses only the letter dataset.
DATASET_YAML = Path("letter_dataset.yaml")
LETTER_DATASET_FOLDER = Path("letter_dataset")

# Final output files. These are the only files kept after the script finishes.
SUMMARY_FILE = Path("experiment_summary.md")
RESULTS_FILE = Path("comparison_results.csv")
PLOT_FILE = Path("comparison_plot.png")

# Temporary folder for all YOLO experiment outputs.
# This folder is deleted automatically at the end.
TEMP_EXPERIMENT_FOLDER = Path("_temporary_experiment_outputs")
YOLO_FALLBACK_TEMP_FOLDER = Path("runs/detect/_temporary_experiment_outputs")
TEMP_MODEL_DOWNLOADS = [Path("yolov8s.pt")]


# -----------------------------
# Experiment settings
# -----------------------------

# epochs:
# This controls how many times YOLO sees the training dataset.
# For the main model, larger values such as 20 or 50 can be used.
# For experiments, we use a small value to keep testing fast.
# More epochs may improve accuracy but take longer.
# Fewer epochs are faster but may not reach the best possible result.
EXPERIMENT_EPOCHS = 5

# batch:
# This controls how many images are processed at one time.
# A larger batch can train faster on a strong GPU but uses more GPU memory.
# A smaller batch is safer on limited hardware but may train more slowly.
EXPERIMENT_BATCH = 8

# patience:
# This controls early stopping.
# If the validation score does not improve for this many epochs, training can stop.
# Since these are short experiments, this value is kept simple.
EXPERIMENT_PATIENCE = 3

# learning rate:
# This controls how large each training update is.
# A higher learning rate can train faster but may be unstable.
# A lower learning rate can be more stable but slower.
EXPERIMENT_LEARNING_RATE = 0.01


# Each dictionary below describes one controlled experiment.
# The experiments compare model size, image size, and optimizer choice.
EXPERIMENTS = [
    {
        "experiment_name": "model_yolov8n_640_auto",
        "comparison_group": "model comparison",
        "model_file": "yolov8n.pt",
        "optimizer": "auto",
        "imgsz": 640,
    },
    {
        "experiment_name": "model_yolov8s_640_auto",
        "comparison_group": "model comparison",
        "model_file": "yolov8s.pt",
        "optimizer": "auto",
        "imgsz": 640,
    },
    {
        "experiment_name": "image_size_416_yolov8n",
        "comparison_group": "image size comparison",
        "model_file": "yolov8n.pt",
        "optimizer": "auto",
        "imgsz": 416,
    },
    {
        "experiment_name": "image_size_640_yolov8n",
        "comparison_group": "image size comparison",
        "model_file": "yolov8n.pt",
        "optimizer": "auto",
        "imgsz": 640,
    },
    {
        "experiment_name": "optimizer_sgd_yolov8n",
        "comparison_group": "optimizer comparison",
        "model_file": "yolov8n.pt",
        "optimizer": "SGD",
        "imgsz": 640,
    },
    {
        "experiment_name": "optimizer_adamw_yolov8n",
        "comparison_group": "optimizer comparison",
        "model_file": "yolov8n.pt",
        "optimizer": "AdamW",
        "imgsz": 640,
    },
]


# -----------------------------
# Basic safety checks
# -----------------------------


def check_project_files():
    """Check that the required letter dataset files exist before training."""

    if not DATASET_YAML.exists():
        print("Error: letter_dataset.yaml was not found.")
        return False

    if not LETTER_DATASET_FOLDER.exists():
        print("Error: letter_dataset folder was not found.")
        return False

    required_folders = [
        Path("letter_dataset/train/images"),
        Path("letter_dataset/train/labels"),
        Path("letter_dataset/valid/images"),
        Path("letter_dataset/valid/labels"),
    ]

    for folder in required_folders:
        if not folder.exists():
            print(f"Error: Missing dataset folder: {folder}")
            return False

    return True


def choose_device():
    """Use CUDA GPU if available, otherwise use CPU."""

    if torch.cuda.is_available():
        print("CUDA GPU detected. Experiments will use GPU device 0.")
        return 0

    print("Warning: CUDA GPU was not detected. Experiments will run on CPU.")
    print("CPU experiments may be slow.")
    return "cpu"


def clean_temporary_folder():
    """Delete temporary experiment files without touching original project folders."""

    if TEMP_EXPERIMENT_FOLDER.exists():
        shutil.rmtree(TEMP_EXPERIMENT_FOLDER)

    # Ultralytics may place relative project paths under runs/detect.
    # Remove that fallback temporary folder too, without touching original runs.
    if YOLO_FALLBACK_TEMP_FOLDER.exists():
        shutil.rmtree(YOLO_FALLBACK_TEMP_FOLDER)

    # YOLO may download yolov8s.pt for the model-comparison experiment.
    # Treat it as temporary so the project is restored after experiments.
    for model_file in TEMP_MODEL_DOWNLOADS:
        if model_file.exists():
            model_file.unlink()


# -----------------------------
# Training and metric extraction
# -----------------------------


def run_one_experiment(experiment, device):
    """Train one short YOLO experiment and return its final metrics."""

    experiment_name = experiment["experiment_name"]
    model_file = experiment["model_file"]
    optimizer = experiment["optimizer"]
    image_size = experiment["imgsz"]

    print("\n" + "=" * 70)
    print(f"Starting experiment: {experiment_name}")
    print(f"Model: {model_file}")
    print(f"Optimizer: {optimizer}")
    print(f"Image size: {image_size}")
    print("=" * 70)

    start_time = time.time()

    try:
        model = YOLO(model_file)

        # All YOLO outputs go into the temporary folder.
        # This protects the original outputs/ and runs/ folders.
        model.train(
            data=str(DATASET_YAML),
            epochs=EXPERIMENT_EPOCHS,
            batch=EXPERIMENT_BATCH,
            imgsz=image_size,
            lr0=EXPERIMENT_LEARNING_RATE,
            optimizer=optimizer,
            patience=EXPERIMENT_PATIENCE,
            device=device,
            project=str(TEMP_EXPERIMENT_FOLDER.resolve()),
            name=experiment_name,
            exist_ok=True,
            plots=False,
            save=False,
            verbose=False,
        )

        training_time_seconds = time.time() - start_time
        results_csv = TEMP_EXPERIMENT_FOLDER / experiment_name / "results.csv"

        if not results_csv.exists():
            print(f"Warning: results.csv was not found for {experiment_name}.")
            return make_failed_result(experiment, training_time_seconds, "missing results.csv")

        inference_speed_ms = measure_inference_speed(model, experiment, device)
        result = extract_final_metrics(
            experiment,
            results_csv,
            training_time_seconds,
            inference_speed_ms,
        )
        result["status"] = "success"
        result["error_message"] = ""
        return result

    except Exception as error:
        training_time_seconds = time.time() - start_time
        print(f"Experiment failed: {experiment_name}")
        print(f"Error message: {error}")
        return make_failed_result(experiment, training_time_seconds, str(error))


def measure_inference_speed(model, experiment, device):
    """Run a validation pass and return YOLO inference speed in milliseconds."""

    try:
        metrics = model.val(
            data=str(DATASET_YAML),
            imgsz=experiment["imgsz"],
            device=device,
            project=str(TEMP_EXPERIMENT_FOLDER.resolve()),
            name=experiment["experiment_name"] + "_validation",
            exist_ok=True,
            plots=False,
            verbose=False,
        )

        # Ultralytics reports speed as milliseconds per image.
        return round(float(metrics.speed.get("inference", 0.0)), 4)

    except Exception as error:
        print("Warning: Could not measure inference speed.")
        print(f"Error message: {error}")
        return ""


def extract_final_metrics(experiment, results_csv, training_time_seconds, inference_speed_ms):
    """Read YOLO results.csv and collect the final epoch metrics."""

    data = pd.read_csv(results_csv)

    # Remove extra spaces from column names because YOLO sometimes pads them.
    data.columns = [column.strip() for column in data.columns]

    final_row = data.iloc[-1]

    return {
        "experiment_name": experiment["experiment_name"],
        "comparison_group": experiment["comparison_group"],
        "yolo_model": experiment["model_file"],
        "optimizer": experiment["optimizer"],
        "image_size": experiment["imgsz"],
        "epochs": EXPERIMENT_EPOCHS,
        "final_precision": float(final_row["metrics/precision(B)"]),
        "final_recall": float(final_row["metrics/recall(B)"]),
        "final_map50": float(final_row["metrics/mAP50(B)"]),
        "final_map50_95": float(final_row["metrics/mAP50-95(B)"]),
        "final_train_box_loss": float(final_row["train/box_loss"]),
        "final_train_cls_loss": float(final_row["train/cls_loss"]),
        "final_train_dfl_loss": float(final_row["train/dfl_loss"]),
        "final_val_box_loss": float(final_row["val/box_loss"]),
        "final_val_cls_loss": float(final_row["val/cls_loss"]),
        "final_val_dfl_loss": float(final_row["val/dfl_loss"]),
        "training_time_seconds": round(training_time_seconds, 2),
        "inference_speed_ms": inference_speed_ms,
    }


def make_failed_result(experiment, training_time_seconds, error_message):
    """Create a row for a failed experiment so the report still explains what happened."""

    return {
        "experiment_name": experiment["experiment_name"],
        "comparison_group": experiment["comparison_group"],
        "yolo_model": experiment["model_file"],
        "optimizer": experiment["optimizer"],
        "image_size": experiment["imgsz"],
        "epochs": EXPERIMENT_EPOCHS,
        "final_precision": "",
        "final_recall": "",
        "final_map50": "",
        "final_map50_95": "",
        "final_train_box_loss": "",
        "final_train_cls_loss": "",
        "final_train_dfl_loss": "",
        "final_val_box_loss": "",
        "final_val_cls_loss": "",
        "final_val_dfl_loss": "",
        "training_time_seconds": round(training_time_seconds, 2),
        "inference_speed_ms": "",
        "status": "failed",
        "error_message": error_message,
    }


# -----------------------------
# Reporting and plotting
# -----------------------------


def save_results_table(results):
    """Save all experiment results into one CSV table."""

    table = pd.DataFrame(results)
    table.to_csv(RESULTS_FILE, index=False)
    return table


def make_comparison_plot(table):
    """Create one figure containing all comparison plots."""

    successful = table[table["status"] == "success"].copy()

    if successful.empty:
        print("No successful experiments. Creating an empty comparison plot.")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No successful experiments", ha="center", va="center")
        plt.axis("off")
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        return

    names = successful["experiment_name"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("YOLOv8 ASL Letter Experiment Comparison", fontsize=16)

    axes[0, 0].bar(names, successful["final_map50"])
    axes[0, 0].set_title("mAP50 Comparison")
    axes[0, 0].set_ylabel("mAP50")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(names, successful["final_precision"])
    axes[0, 1].set_title("Precision Comparison")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].bar(names, successful["final_recall"])
    axes[1, 0].set_title("Recall Comparison")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].bar(names, successful["training_time_seconds"])
    axes[1, 1].set_title("Training Time Comparison")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()


def value_or_not_available(value):
    """Format missing report values in a readable way."""

    if pd.isna(value) or value == "":
        return "not available"
    return value


def make_markdown_table(table):
    """Create a simple markdown table without extra dependencies."""

    columns = list(table.columns)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in table.iterrows():
        values = [str(value_or_not_available(row[column])) for column in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def find_best_row(table, column):
    """Find the best successful experiment for one numeric metric."""

    successful = table[table["status"] == "success"].copy()

    if successful.empty:
        return None

    successful[column] = pd.to_numeric(successful[column], errors="coerce")
    successful = successful.dropna(subset=[column])

    if successful.empty:
        return None

    return successful.loc[successful[column].idxmax()]


def find_fastest_row(table):
    """Find the successful experiment with the shortest training time."""

    successful = table[table["status"] == "success"].copy()

    if successful.empty:
        return None

    successful["training_time_seconds"] = pd.to_numeric(
        successful["training_time_seconds"], errors="coerce"
    )
    successful = successful.dropna(subset=["training_time_seconds"])

    if successful.empty:
        return None

    return successful.loc[successful["training_time_seconds"].idxmin()]


def create_markdown_report(table):
    """Generate the final markdown experiment summary."""

    best_map = find_best_row(table, "final_map50")
    best_precision = find_best_row(table, "final_precision")
    best_recall = find_best_row(table, "final_recall")
    fastest = find_fastest_row(table)

    lines = []
    lines.append("# Automated YOLOv8 ASL Letter Experiment Summary")
    lines.append("")
    lines.append("This report was generated automatically by `run_experiments.py`.")
    lines.append("The experiments used only the ASL letter dataset and were trained for a small number of epochs for lightweight comparison.")
    lines.append("")
    lines.append("## Safety Notes")
    lines.append("")
    lines.append("The experiment pipeline used a temporary folder for YOLO outputs and deleted it after completion.")
    lines.append("The original `outputs/`, `runs/`, `models/`, main model weights, logs, CSV files, and plots were not modified by the experiment script.")
    lines.append("")
    lines.append("## Experiment Table")
    lines.append("")
    lines.append(make_markdown_table(table))
    lines.append("")
    lines.append("## Best Configuration by mAP50")
    lines.append("")

    if best_map is not None:
        lines.append(
            f"The highest final mAP50 was achieved by `{best_map['experiment_name']}` "
            f"with mAP50 `{best_map['final_map50']}`. This configuration used "
            f"model `{best_map['yolo_model']}`, optimizer `{best_map['optimizer']}`, "
            f"and image size `{best_map['image_size']}`."
        )
    else:
        lines.append("No successful experiment was available for mAP50 comparison.")

    lines.append("")
    lines.append("## Fastest Training Configuration")
    lines.append("")

    if fastest is not None:
        lines.append(
            f"The fastest successful experiment was `{fastest['experiment_name']}`, "
            f"with training time `{fastest['training_time_seconds']}` seconds."
        )
    else:
        lines.append("No successful experiment was available for training-time comparison.")

    lines.append("")
    lines.append("## Precision and Recall")
    lines.append("")

    if best_precision is not None:
        lines.append(
            f"The best final precision was produced by `{best_precision['experiment_name']}` "
            f"with precision `{best_precision['final_precision']}`. Precision measures how often predicted detections are correct."
        )
    else:
        lines.append("Precision comparison was not available.")

    if best_recall is not None:
        lines.append(
            f"The best final recall was produced by `{best_recall['experiment_name']}` "
            f"with recall `{best_recall['final_recall']}`. Recall measures how many real objects the model found."
        )
    else:
        lines.append("Recall comparison was not available.")

    lines.append("")
    lines.append("## Model Size Comparison")
    lines.append("")
    lines.append("The YOLOv8n and YOLOv8s experiments compare a smaller model with a larger model. YOLOv8n is usually faster and lighter, while YOLOv8s may improve accuracy at the cost of more computation. The best choice depends on whether the project needs maximum speed or stronger accuracy.")
    lines.append("")
    lines.append("## Image Size Comparison")
    lines.append("")
    lines.append("The image-size experiments compare `416` and `640`. A smaller image size usually trains faster and uses less memory. A larger image size may detect hand details better, but it normally takes more time.")
    lines.append("")
    lines.append("## Optimizer Comparison")
    lines.append("")
    lines.append("The optimizer experiments compare SGD and AdamW. SGD is a traditional optimizer often used for stable object detection training. AdamW can converge quickly in some cases, but its performance depends on the dataset and learning rate.")
    lines.append("")
    lines.append("## Speed and Accuracy Tradeoff")
    lines.append("")
    lines.append("A real-time ASL system should balance detection accuracy with inference speed. If two experiments produce similar accuracy, the faster and smaller configuration is usually more practical for webcam detection.")
    lines.append("")
    lines.append("## Recommended Setup for Real-Time ASL Detection")
    lines.append("")

    if best_map is not None and fastest is not None:
        lines.append("For real-time ASL letter detection, the best setup should be selected by considering both mAP50 and training/inference speed. YOLOv8n is usually the safest choice for real-time use because it is lightweight. If YOLOv8s provides a clear accuracy improvement and the hardware can run it smoothly, it can be considered as an alternative.")
    else:
        lines.append("A recommendation could not be generated because no successful experiment metrics were available.")

    lines.append("")
    lines.append("## Comparison Plot")
    lines.append("")
    lines.append("The combined comparison plot is saved as `comparison_plot.png`.")

    SUMMARY_FILE.write_text("\n".join(lines))


# -----------------------------
# Main program
# -----------------------------


def main():
    """Run all experiments, create final reports, and clean temporary files."""

    print("Starting automated YOLOv8 ASL letter experiments...")

    if not check_project_files():
        print("Project checks failed. Experiments were not started.")
        return

    device = choose_device()

    # Start from a clean temporary experiment folder.
    # This does not touch original outputs/, runs/, or models/.
    clean_temporary_folder()
    TEMP_EXPERIMENT_FOLDER.mkdir(parents=True, exist_ok=True)

    results = []

    try:
        for experiment in EXPERIMENTS:
            result = run_one_experiment(experiment, device)
            results.append(result)

        table = save_results_table(results)
        make_comparison_plot(table)
        create_markdown_report(table)

        print("\nExperiments finished.")
        print(f"Saved: {SUMMARY_FILE}")
        print(f"Saved: {RESULTS_FILE}")
        print(f"Saved: {PLOT_FILE}")

    finally:
        # Always clean temporary experiment outputs, even if an experiment fails.
        clean_temporary_folder()
        print("Temporary experiment folders were removed.")


if __name__ == "__main__":
    main()
