#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

def merge_coco(root_dir: Path, out_dir: Path) -> None:
    """Merge multiple coco folders under root_dir into out_dir.

    Expects structure like:
      root_dir/<batch_or_run>/coco/images/train2014/*.png
      root_dir/<batch_or_run>/coco/labels/train2014/*.txt
    """
    out_images = out_dir / "images" / "train2014"
    out_labels = out_dir / "labels" / "train2014"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    coco_dirs = sorted(root_dir.glob("**/coco"))
    if not coco_dirs:
        print(f"No coco directories found under {root_dir}")
        return

    merged = 0
    for coco_dir in coco_dirs:
        images_dir = coco_dir / "images" / "train2014"
        labels_dir = coco_dir / "labels" / "train2014"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        prefix = coco_dir.parent.name
        for img_path in images_dir.glob("*.png"):
            label_path = labels_dir / (img_path.stem + ".txt")

            # prefix filenames to avoid collisions
            new_stem = f"{prefix}_{img_path.stem}"
            out_img = out_images / (new_stem + img_path.suffix)
            out_lbl = out_labels / (new_stem + ".txt")

            shutil.copy2(img_path, out_img)
            if label_path.exists():
                shutil.copy2(label_path, out_lbl)
            else:
                out_lbl.write_text("")

            merged += 1

    print(f"Merged {merged} images into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge coco folders under a root into a single coco dataset.")
    parser.add_argument("--root", type=str, default="/data1/home/ypliu/DIODE/syn_data_result", help="Root directory containing multiple coco folders")
    parser.add_argument("--out", type=str, default="/data1/home/ypliu/merged_coco_syn_v2", help="Output directory for merged coco dataset")
    args = parser.parse_args()

    root_dir = Path(args.root)
    out_dir = Path(args.out)

    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir}")

    merge_coco(root_dir, out_dir)


if __name__ == "__main__":
    main()
