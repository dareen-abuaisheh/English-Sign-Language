#!/usr/bin/env python3
"""
Download and validate the Roboflow word-level ASL dataset.

Example:
  export ROBOFLOW_API_KEY="your_key"
  python download_word_dataset.py --download --test
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

from roboflow import Roboflow


DEFAULT_WORKSPACE = "majorproject-25tao"
DEFAULT_PROJECT = "american-sign-language-v36cz"
DEFAULT_LOCATION = "word_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and test Roboflow word-level ASL dataset."
    )
    parser.add_argument(
        "--api-key",
        default="W9QoCQ2NXRk9L3MQUw5f",
        help="Roboflow API key. A default key is already included.",
    )
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Roboflow dataset version. If omitted, latest version is used.",
    )
    parser.add_argument(
        "--format",
        default="yolov8",
        help="Dataset export format (default: yolov8).",
    )
    parser.add_argument(
        "--location",
        default=DEFAULT_LOCATION,
        help="Where to download the dataset.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from Roboflow.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Validate dataset structure and labels.",
    )
    return parser.parse_args()


def download_dataset(
    api_key: str,
    workspace: str,
    project_name: str,
    version: int | None,
    export_format: str,
    location: str,
) -> Path:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)

    if version is None:
        versions = project.versions()
        if not versions:
            raise RuntimeError("No versions found in project.")
        version_obj = versions[-1]
    else:
        version_obj = project.version(version)

    dataset = version_obj.download(export_format, location=location)
    return Path(dataset.location).resolve()


def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in images_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]


def validate_split(root: Path, split: str) -> Dict[str, int]:
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing split folders for '{split}' at {root}")

    images = list_images(images_dir)
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    missing_labels = 0
    bad_lines = 0
    empty_labels = 0

    for img in images:
        label_file = labels_dir / f"{img.stem}.txt"
        if not label_file.exists():
            missing_labels += 1
            continue

        lines = [line.strip() for line in label_file.read_text().splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
            continue

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                bad_lines += 1
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
                if class_id < 0 or any(c < 0 or c > 1 for c in coords):
                    bad_lines += 1
            except ValueError:
                bad_lines += 1

    return {
        "images": len(images),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "bad_lines": bad_lines,
    }


def find_dataset_yaml(root: Path) -> Path | None:
    candidates = [
        root / "data.yaml",
        root / "dataset.yaml",
        root.parent / "data.yaml",
        root.parent / "dataset.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_validation(root: Path) -> None:
    yaml_path = find_dataset_yaml(root)
    if yaml_path is not None:
        print(f"Found dataset config: {yaml_path}")
    else:
        print("No data.yaml/dataset.yaml found. Continuing with folder/label validation only.")

    totals: Dict[str, Dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        totals[split] = validate_split(root, split)

    print("\nValidation summary:")
    for split, stats in totals.items():
        print(
            f"  {split}: images={stats['images']}, "
            f"missing_labels={stats['missing_labels']}, "
            f"empty_labels={stats['empty_labels']}, "
            f"bad_lines={stats['bad_lines']}"
        )

    issues: List[Tuple[str, str, int]] = []
    for split, stats in totals.items():
        for key in ("missing_labels", "empty_labels", "bad_lines"):
            if stats[key] > 0:
                issues.append((split, key, stats[key]))

    if issues:
        issue_text = ", ".join(f"{s}:{k}={v}" for s, k, v in issues)
        raise RuntimeError(f"Dataset validation failed: {issue_text}")

    print("\nDataset validation passed.")


def main() -> None:
    args = parse_args()

    if not args.download and not args.test:
        raise SystemExit("Use at least one flag: --download and/or --test")

    dataset_root = Path(args.location).resolve()

    if args.download:
        if not args.api_key:
            raise SystemExit("Missing API key. Set ROBOFLOW_API_KEY or pass --api-key.")
        dataset_root = download_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project_name=args.project,
            version=args.version,
            export_format=args.format,
            location=args.location,
        )
        print(f"Downloaded dataset to: {dataset_root}")

    if args.test:
        run_validation(dataset_root)


if __name__ == "__main__":
    main()