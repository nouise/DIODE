#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

def find_label_path(image_path: Path) -> Path:
    # Common pattern: /images/.../xxx.jpg -> /labels/.../xxx.txt
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        return label_path
    except ValueError:
        # Fallback: same directory with .txt
        return image_path.with_suffix(".txt")

def unique_dest(dest_dir: Path, filename: str) -> Path:
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    i = 1
    while True:
        cand = dest_dir / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def build_coco_from_list(list_path: Path, out_dir: Path, split: str) -> None:
    images_out = out_dir / "images" / split
    labels_out = out_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    with list_path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    copied = 0
    for line in lines:
        img_path = Path(line)
        if not img_path.exists():
            print(f"Skip missing image: {img_path}")
            continue

        label_path = find_label_path(img_path)

        dest_img = unique_dest(images_out, img_path.name)
        dest_lbl = labels_out / (dest_img.stem + ".txt")

        shutil.copy2(img_path, dest_img)
        if label_path.exists():
            shutil.copy2(label_path, dest_lbl)
        else:
            dest_lbl.write_text("")

        copied += 1

    print(f"Copied {copied} images into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a COCO-style dataset from a list of image paths.")
    parser.add_argument("--list", default="/data1/home/ypliu/DIODE/augmented_data/random_164.txt", help="Path to txt list of image paths")
    parser.add_argument("--out", default="/data1/home/ypliu/voc0203_random", help="Output directory for COCO-style dataset")
    parser.add_argument("--split", default="train164", help="Split name, default: train2014")
    args = parser.parse_args()

    list_path = Path(args.list)
    out_dir = Path(args.out)
    if not list_path.exists():
        raise SystemExit(f"List file not found: {list_path}")

    build_coco_from_list(list_path, out_dir, args.split)


if __name__ == "__main__":
    main()
