#!/usr/bin/env python3
"""Analyze YOLO-style label files across multiple directories.

Usage examples:
    python analyze_label_differences.py
    python analyze_label_differences.py --out report.csv --grid-out top5_grid.jpg

The script will produce a CSV with per-file box counts, a short
summary printed to stdout, and an optional visualization grid.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import random


DEFAULT_DIRS = [
    "/data1/home/ypliu/DIODE/syn_data_result_test/day_02_04_2026_time_01_42_32_res512_finetune/coco/labels/train2014",
    "/data1/home/ypliu/DIODE/syn_data_result_test/day_02_04_2026_time_02_12_06_res512_finetune/coco/labels/train2014",
    "/data1/home/ypliu/DIODE/augmented_data/mosaic/labels",
]

# VOC 20 class names (can be customized)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    random.seed(42)
    colors = []
    for _ in range(num_classes):
        colors.append((
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ))
    return colors


def list_label_files(directory: str, recursive: bool = False) -> List[str]:
    files = []
    if recursive:
        for root, _, filenames in os.walk(directory):
            for fn in filenames:
                files.append(os.path.relpath(os.path.join(root, fn), directory))
    else:
        for fn in os.listdir(directory):
            path = os.path.join(directory, fn)
            if os.path.isfile(path):
                files.append(fn)
    return sorted(files)


def count_boxes_in_file(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def load_yolo_labels(label_path: str) -> np.ndarray:
    if not os.path.exists(label_path):
        return np.zeros((0, 6))

    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    labels.append([class_id, x, y, w, h, conf])
    return np.array(labels) if labels else np.zeros((0, 6))


def draw_boxes(
    image: np.ndarray,
    labels: np.ndarray,
    colors: List[Tuple[int, int, int]],
    class_names: List[str] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    for label in labels:
        class_id = int(label[0])
        x_center, y_center, width, height = label[1:5]
        conf = label[5]
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        color = colors[class_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if class_names and class_id < len(class_names):
            label_text = f"{class_names[class_id]}: {conf:.2f}"
        else:
            label_text = f"Class {class_id}: {conf:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            img,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            -1,
        )
        cv2.putText(
            img,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
        )

    return img


def infer_image_dir(label_dir: str) -> str:
    if "/labels" in label_dir:
        return label_dir.replace("/labels", "/images", 1)
    if label_dir.endswith("labels"):
        return label_dir[:-6] + "images"
    return label_dir


def find_image_path(image_dir: str, rel_label_path: str) -> str:
    subdir = os.path.dirname(rel_label_path)
    stem = os.path.splitext(os.path.basename(rel_label_path))[0]
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    for ext in exts + [e.upper() for e in exts]:
        candidate = os.path.join(image_dir, subdir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return ""


def render_cell(
    image_path: str,
    label_path: str,
    title: str,
    cell_size: Tuple[int, int],
    class_names: List[str],
) -> np.ndarray:
    width, height = cell_size
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((height, width, 3), dtype=np.uint8)
        labels = load_yolo_labels(label_path)
        max_class_id = int(labels[:, 0].max()) if len(labels) > 0 else 0
        num_colors = max(max_class_id + 1, 80)
        colors = generate_colors(num_colors)
        img = draw_boxes(img, labels, colors, class_names)
    else:
        img = np.zeros((height, width, 3), dtype=np.uint8)
    img = cv2.resize(img, (width, height))

    cv2.rectangle(img, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def analyze(dirs: List[str], recursive: bool = False) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    # mapping: filename -> {dir -> count}
    per_file: Dict[str, Dict[str, int]] = defaultdict(dict)
    totals: Dict[str, int] = {d: 0 for d in dirs}

    # collect file lists per dir
    files_per_dir: Dict[str, List[str]] = {}
    for d in dirs:
        files_per_dir[d] = list_label_files(d, recursive=recursive)

    # union of filenames (relative names)
    all_files = set()
    for fl in files_per_dir.values():
        all_files.update(fl)

    for fname in sorted(all_files):
        for d in dirs:
            file_path = os.path.join(d, fname)
            if os.path.isfile(file_path):
                cnt = count_boxes_in_file(file_path)
            else:
                cnt = 0
            per_file[fname][d] = cnt
            totals[d] += cnt

    return per_file, totals


def write_csv(out_path: str, per_file: Dict[str, Dict[str, int]], dirs: List[str]):
    header = ["filename"] + dirs + ["max", "min", "diff_max_min"]
    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(header)
        for fname in sorted(per_file.keys()):
            counts = [per_file[fname].get(d, 0) for d in dirs]
            mx = max(counts) if counts else 0
            mn = min(counts) if counts else 0
            row = [fname] + counts + [mx, mn, mx - mn]
            writer.writerow(row)


def print_summary(per_file: Dict[str, Dict[str, int]], totals: Dict[str, int], dirs: List[str]):
    print("Summary:")
    for d in dirs:
        print(f"- {d}: total boxes = {totals[d]}")

    # pairwise total differences
    print("\nPairwise total differences (abs):")
    max_total_diff = -1
    max_total_pair = ("", "")
    for i, a in enumerate(dirs):
        for j, b in enumerate(dirs):
            if j <= i:
                continue
            diff = abs(totals[a] - totals[b])
            if diff > max_total_diff:
                max_total_diff = diff
                max_total_pair = (a, b)
            print(f"- {a}  vs  {b}: {diff}")
    if max_total_diff >= 0:
        print(f"\nMax total difference: {max_total_diff} ({max_total_pair[0]} vs {max_total_pair[1]})")

    # per-file with large differences (non-zero)
    print("\nFiles with differences (max-min > 0):")
    any_diff = False
    max_file_diff = -1
    max_file_names: List[str] = []
    for fname in sorted(per_file.keys()):
        counts = [per_file[fname].get(d, 0) for d in dirs]
        mx = max(counts)
        mn = min(counts)
        if mx - mn > 0:
            any_diff = True
            print(f"- {fname}: counts={counts}, diff={mx-mn}")
        if mx - mn > max_file_diff:
            max_file_diff = mx - mn
            max_file_names = [fname]
        elif mx - mn == max_file_diff and max_file_diff >= 0:
            max_file_names.append(fname)
    if not any_diff:
        print("- No per-file differences found.")
    if max_file_diff >= 0:
        print(f"\nMax per-file difference: {max_file_diff}")
        if max_file_names:
            print("Files with max difference:")
            for n in max_file_names:
                print(f"- {n}")


def save_topk_grid(
    per_file: Dict[str, Dict[str, int]],
    dirs: List[str],
    out_path: str,
    top_k: int = 5,
    col_gap: int = 24,
    bg_color: Tuple[int, int, int] = (245, 245, 245),
):
    if not per_file:
        print("No files to visualize.")
        return

    diffs = []
    for fname, counts_map in per_file.items():
        counts = [counts_map.get(d, 0) for d in dirs]
        diff = max(counts) - min(counts) if counts else 0
        diffs.append((diff, fname, counts))

    diffs.sort(key=lambda x: x[0], reverse=True)
    top_items = diffs[:top_k]

    image_dirs = [infer_image_dir(d) for d in dirs]
    cell_size = (512, 512)
    class_names = VOC_CLASSES

    rows = []
    for diff, fname, counts in top_items:
        row_imgs = []
        for d, img_dir in zip(dirs, image_dirs):
            label_path = os.path.join(d, fname)
            image_path = find_image_path(img_dir, fname)
            title = f"{os.path.basename(d)} | {os.path.basename(fname)} | n={counts[dirs.index(d)]}"
            row_imgs.append(render_cell(image_path, label_path, title, cell_size, class_names))

        if len(row_imgs) > 1 and col_gap > 0:
            gap = np.full((cell_size[1], col_gap, 3), bg_color, dtype=np.uint8)
            row = row_imgs[0]
            for img in row_imgs[1:]:
                row = cv2.hconcat([row, gap, img])
        else:
            row = cv2.hconcat(row_imgs)

        cv2.rectangle(row, (0, 0), (row.shape[1], 30), (50, 50, 50), -1)
        cv2.putText(row, f"diff={diff} | {fname}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        rows.append(row)

    grid = cv2.vconcat(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, grid)
    print(f"Saved top-{top_k} grid to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare YOLO label box counts across directories")
    p.add_argument("dirs", nargs="*", help="Label directories to compare (order preserved)")
    p.add_argument("--out", default="label_compare_report.csv", help="CSV output path")
    p.add_argument("--recursive", action="store_true", help="Recursively walk directories and match relative paths")
    p.add_argument("--grid-out", default="top5_diff_grid.jpg", help="Output image path for top-k grid")
    p.add_argument("--top-k", type=int, default=5, help="Top-K files to visualize by max-min diff")
    return p.parse_args()


def main():
    args = parse_args()
    dirs = args.dirs if args.dirs else DEFAULT_DIRS
    # validate directories
    for d in dirs:
        if not os.path.isdir(d):
            print(f"Error: not a directory: {d}")
            return

    per_file, totals = analyze(dirs, recursive=args.recursive)
    write_csv(args.out, per_file, dirs)
    print(f"Wrote CSV report to: {args.out}")
    print_summary(per_file, totals, dirs)
    save_topk_grid(per_file, dirs, args.grid_out, top_k=args.top_k)


if __name__ == "__main__":
    main()
