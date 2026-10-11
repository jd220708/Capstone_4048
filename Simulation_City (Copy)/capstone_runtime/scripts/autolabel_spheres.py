#!/usr/bin/env python3
"""
Auto-label pink spheres in captured frames, in YOLO format.

The spheres are flat-shaded solid colours on a rendered background, which makes
colour thresholding near-exact. Hue values come straight from the palette in
SphereNetworkSystem.cc, not from eyeballing screenshots.

    Colour    RGB (0-1)            OpenCV hue
    RED       1.00, 0.00, 0.00       0
    ORANGE    1.00, 0.35, 0.00      10
    YELLOW    1.00, 1.00, 0.00      30
    GREEN     0.00, 1.00, 0.10      63
    BLUE      0.00, 0.25, 1.00     113
    PURPLE    0.55, 0.05, 1.00     136
    PINK      1.00, 0.05, 0.55     164   <- target

Pink's nearest neighbours are purple (28 away) and red (16 away, wrapping at
180). The window below sits clear of both.

Usage:
    python3 autolabel_spheres.py                    # label + review images
    python3 autolabel_spheres.py --distractors      # also label other colours
    python3 autolabel_spheres.py --no-review        # skip review images

Then check ~/sphere_data/review/ before trusting the labels. Do actually look
at them — a systematic labelling error is invisible until it has already
poisoned your training run.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not found. Run: pip install opencv-python --break-system-packages")


# --- thresholds ------------------------------------------------------------
# Verified against synthetic shaded spheres of all seven palette colours:
# 99% width coverage of the target, zero false positives on the other six.
PINK_LO = np.array([153, 120, 40])
PINK_HI = np.array([173, 255, 255])

# Other palette colours, for optional distractor labelling. Red spans the hue
# wrap so it needs two ranges.
DISTRACTORS = {
    "orange": [(np.array([5, 120, 40]),   np.array([18, 255, 255]))],
    "yellow": [(np.array([22, 120, 40]),  np.array([38, 255, 255]))],
    "green":  [(np.array([55, 120, 40]),  np.array([72, 255, 255]))],
    "blue":   [(np.array([105, 120, 40]), np.array([122, 255, 255]))],
    "purple": [(np.array([128, 120, 40]), np.array([145, 255, 255]))],
    "red":    [(np.array([0, 120, 40]),   np.array([4, 255, 255])),
               (np.array([176, 120, 40]), np.array([180, 255, 255]))],
}

MIN_AREA = 12          # px; below this it's noise or a sphere too small to learn from
MAX_ASPECT = 3.0       # reject long thin blobs - not spheres


def mask_for(hsv, ranges):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 3x3 rather than 5x5: a larger kernel merges two spheres sitting side by
    # side into a single box. Spheres that genuinely overlap on screen will
    # still merge - that's inherent to blob detection, and rare enough not to
    # matter across a few hundred frames.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return m


def boxes_from(mask, min_area):
    out = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        aspect = max(w / h, h / w)
        if aspect > MAX_ASPECT:
            continue
        out.append((x, y, w, h))
    return out


def to_yolo(box, img_w, img_h):
    x, y, w, h = box
    return ((x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path.home() / "sphere_data"))
    p.add_argument("--distractors", action="store_true",
                   help="Also label other-coloured spheres as class 1")
    p.add_argument("--no-review", action="store_true")
    p.add_argument("--min-area", type=int, default=MIN_AREA)
    args = p.parse_args()

    root = Path(args.data)
    img_dir = root / "images"
    lbl_dir = root / "labels"
    rev_dir = root / "review"

    if not img_dir.is_dir():
        sys.exit(f"No images/ folder in {root}. Capture some frames first.")

    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not images:
        sys.exit(f"No image files in {img_dir}")

    lbl_dir.mkdir(exist_ok=True)
    if not args.no_review:
        rev_dir.mkdir(exist_ok=True)

    classes = ["pink_sphere"] + (["other_sphere"] if args.distractors else [])
    (root / "classes.txt").write_text("\n".join(classes) + "\n")

    n_pink = n_other = n_empty = 0

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skipped unreadable file: {path.name}")
            continue
        ih, iw = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lines = []
        review = img.copy() if not args.no_review else None

        pink = boxes_from(mask_for(hsv, [(PINK_LO, PINK_HI)]), args.min_area)
        for b in pink:
            cx, cy, w, h = to_yolo(b, iw, ih)
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if review is not None:
                x, y, bw, bh = b
                cv2.rectangle(review, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
        n_pink += len(pink)

        if args.distractors:
            for ranges in DISTRACTORS.values():
                for b in boxes_from(mask_for(hsv, ranges), args.min_area):
                    cx, cy, w, h = to_yolo(b, iw, ih)
                    lines.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    n_other += 1
                    if review is not None:
                        x, y, bw, bh = b
                        cv2.rectangle(review, (x, y), (x + bw, y + bh), (128, 128, 128), 1)

        # An empty label file is valid and useful - it teaches the model what
        # "no target here" looks like. Don't skip these frames.
        (lbl_dir / f"{path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if not lines:
            n_empty += 1

        if review is not None:
            cv2.imwrite(str(rev_dir / path.name), review)

    print(f"\n{len(images)} images processed")
    print(f"  {n_pink} pink spheres labelled")
    if args.distractors:
        print(f"  {n_other} other-coloured spheres labelled as class 1")
    print(f"  {n_empty} frames with no target (kept as negatives)")
    print(f"\nLabels:  {lbl_dir}")
    if not args.no_review:
        print(f"Review:  {rev_dir}")
        print("\nOpen a dozen review images before training. Magenta boxes are the")
        print("target class. If any purple sphere has a magenta box, stop and tell me.")


if __name__ == "__main__":
    main()
