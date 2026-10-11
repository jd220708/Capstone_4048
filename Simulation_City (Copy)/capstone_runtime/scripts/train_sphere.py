#!/usr/bin/env python3
"""
Pink sphere detector - local training on an RTX 4060.

Capstone: UAV gimbal target reacquisition.

Usage:
    source ~/yolo_env/bin/activate
    python3 train_sphere.py

Everything you need to edit is in the CONFIG block below.
"""

import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ============================== CONFIG =======================================

# Folder holding your Label Studio export.
# Must contain: images/  labels/  classes.txt
DATASET = Path("/home/felipe/sphere_data")

# Where the split dataset gets written (created fresh each run).
WORKDIR = Path.home() / "sphere_training"

# Fraction of SEQUENCES (not images) used for training.
TRAIN_PCT = 0.8

MODEL    = "yolo11s.pt"   # yolo11n.pt is faster, less accurate
EPOCHS   = 80
IMG_SIZE = 960            # see note at the bottom of this file
BATCH    = -1             # -1 lets Ultralytics pick a batch size for your VRAM
SEED     = 0

# Also train a nano model afterwards, for an accuracy-vs-speed comparison.
TRAIN_NANO_TOO = True

# =============================================================================


def sequence_key(filename: str) -> str:
    """Which recording does this frame belong to?

    'flight3_0142.jpg' -> 'flight3'
    Adjust this if your filenames follow a different convention.
    """
    stem = Path(filename).stem
    m = re.match(r"^(.*)[_-]\d+$", stem)
    return m.group(1) if m else stem


def build_split():
    img_dir = DATASET / "images"
    lbl_dir = DATASET / "labels"

    if not img_dir.is_dir():
        sys.exit(f"No images/ folder in {DATASET}")
    if not lbl_dir.is_dir():
        sys.exit(f"No labels/ folder in {DATASET}")

    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        sys.exit(f"No image files found in {img_dir}")

    groups = defaultdict(list)
    for p in images:
        groups[sequence_key(p.name)].append(p)

    print(f"{len(images)} images in {len(groups)} sequence(s):")
    for k, v in sorted(groups.items()):
        print(f"   {k:<30} {len(v):>5} frames")

    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)

    if len(keys) == 1:
        print()
        print("*** WARNING ***")
        print("Only one sequence detected, so train and val will be split")
        print("randomly. If these frames came from one continuous recording,")
        print("consecutive frames are near-identical and your validation score")
        print("will be optimistic. Rename files per recording run")
        print("(run1_0001.jpg, run2_0001.jpg, ...) to fix this.")
        print()
        only = list(groups[keys[0]])
        random.Random(SEED).shuffle(only)
        cut = int(len(only) * TRAIN_PCT)
        split = {"train": only[:cut], "validation": only[cut:]}
    else:
        cut = max(1, round(len(keys) * TRAIN_PCT))
        train_keys, val_keys = keys[:cut], keys[cut:]
        if not val_keys:
            train_keys, val_keys = keys[:-1], keys[-1:]
        split = {
            "train":      [p for k in train_keys for p in groups[k]],
            "validation": [p for k in val_keys   for p in groups[k]],
        }
        print(f"\ntrain sequences: {train_keys}")
        print(f"val   sequences: {val_keys}")

    data_root = WORKDIR / "data"
    shutil.rmtree(data_root, ignore_errors=True)

    missing = 0
    for subset, files in split.items():
        (data_root / subset / "images").mkdir(parents=True, exist_ok=True)
        (data_root / subset / "labels").mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy(img, data_root / subset / "images" / img.name)
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, data_root / subset / "labels" / lbl.name)
            else:
                missing += 1

    print(f"\ntrain: {len(split['train'])} images")
    print(f"val:   {len(split['validation'])} images")
    if missing:
        print(f"\n{missing} image(s) had no matching label file.")
        print("A few is fine (genuinely empty frames).")
        print("Many means the filenames don't line up between images/ and labels/.")

    return data_root


def write_yaml(data_root: Path) -> Path:
    import yaml

    classes_txt = DATASET / "classes.txt"
    if not classes_txt.exists():
        sys.exit(f"classes.txt not found at {classes_txt}")

    classes = [c.strip() for c in classes_txt.read_text().splitlines() if c.strip()]
    if len(classes) != 1:
        print(f"Note: {len(classes)} classes found - expected 1 for a single target.")

    yaml_path = WORKDIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path":  str(data_root),
            "train": "train/images",
            "val":   "validation/images",
            "nc":    len(classes),
            "names": classes,
        }, f, sort_keys=False)

    print(f"\nWrote {yaml_path}")
    print(yaml_path.read_text())
    return yaml_path


def train(yaml_path: Path, model: str, run_name: str):
    from ultralytics import YOLO

    print(f"\n{'=' * 70}\nTraining {model} -> {run_name}\n{'=' * 70}\n")

    YOLO(model).train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        seed=SEED,
        device=0,
        project=str(WORKDIR / "runs"),
        name=run_name,
        exist_ok=True,
        patience=20,
        # Colour augmentation turned down: the target is defined mostly by
        # being pink, so heavy saturation/brightness jitter trains the model
        # to ignore its single most useful feature.
        hsv_s=0.3,
        hsv_v=0.2,
        fliplr=0.5,
        flipud=0.0,
        degrees=5,
    )


def compare():
    """Print final metrics side by side, for the report."""
    import csv

    rows = []
    for name in ["sphere_s", "sphere_n"]:
        path = WORKDIR / "runs" / name / "results.csv"
        if not path.exists():
            continue
        with open(path) as f:
            last = list(csv.DictReader(f))[-1]
        last = {k.strip(): v for k, v in last.items()}

        def g(key):
            try:
                return f"{float(last[key]):.4f}"
            except (KeyError, ValueError, TypeError):
                return "-"

        rows.append((
            name,
            last.get("epoch", "-"),
            g("metrics/precision(B)"),
            g("metrics/recall(B)"),
            g("metrics/mAP50(B)"),
            g("metrics/mAP50-95(B)"),
        ))

    if not rows:
        return

    print(f"\n{'=' * 70}\nRESULTS\n{'=' * 70}")
    hdr = ("model", "epochs", "precision", "recall", "mAP50", "mAP50-95")
    print("{:<12}{:<9}{:<12}{:<10}{:<10}{:<10}".format(*hdr))
    for r in rows:
        print("{:<12}{:<9}{:<12}{:<10}{:<10}{:<10}".format(*r))

    print(f"\nWeights and curves are in {WORKDIR / 'runs'}")
    print("Keep results.png, PR_curve.png and results.csv - they belong in your report.")


def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available - this will train on CPU and be very slow.")
            print("Check that nvidia-smi works and that torch was built with CUDA.\n")
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    except ImportError:
        sys.exit("torch not installed. Run: pip install ultralytics")

    data_root = build_split()
    yaml_path = write_yaml(data_root)

    train(yaml_path, MODEL, "sphere_s")
    if TRAIN_NANO_TOO:
        train(yaml_path, "yolo11n.pt", "sphere_n")

    compare()


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# NOTE ON IMG_SIZE
#
# Your project measures reacquisition after the target is lost, so how the
# detector behaves on small, distant spheres is part of your result - not a
# detail. A sphere 15 px across in a 1280-wide frame becomes ~7 px at
# imgsz=640, which YOLO struggles with. 960 keeps it near 11.
#
# If you hit CUDA out-of-memory, drop BATCH to 8 first, then 4, before
# reducing IMG_SIZE. Lowering image size costs you exactly the detections
# your research question depends on.
# -----------------------------------------------------------------------------
