#!/usr/bin/env python3
"""Convert Pascal VOC XML annotations to YOLO txt labels and generate train/val lists.

Usage examples:
  python convert_voc_to_yolo.py --images-root /data1/home/ypliu/yolov3/datasets/voc/images \
      --annotations-root /data1/home/ypliu/yolov3/datasets/voc/Annotations --out-data-dir data/voc

If annotations are already in YOLO txt form (labels folder), the script will skip XML conversion and only generate lists.
"""
import argparse
import os
import sys
import glob
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'
]


def xml_to_yolo(xml_path, names_map):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    if size is None:
        return []
    w = float(size.find('width').text)
    h = float(size.find('height').text)
    out_lines = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is not None and difficult.text == '1':
            continue
        cls = obj.find('name').text
        if cls not in names_map:
            continue
        cls_id = names_map[cls]
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        ymin = float(xmlbox.find('ymin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymax = float(xmlbox.find('ymax').text)
        # convert to x_center y_center w h (normalized)
        x = (xmin + xmax) / 2.0 / w
        y = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        out_lines.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
    return out_lines


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def find_xml_for_image(img_path, annotations_root):
    base = os.path.splitext(os.path.basename(img_path))[0]
    # Common places
    cand = os.path.join(annotations_root, base + '.xml')
    if os.path.exists(cand):
        return cand
    # search recursively
    for root, _, files in os.walk(annotations_root):
        if base + '.xml' in files:
            return os.path.join(root, base + '.xml')
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images-root', default='/data1/home/ypliu/yolov3/datasets/voc/images')
    p.add_argument('--annotations-root', default='/data1/home/ypliu/yolov3/datasets/voc/Annotations')
    p.add_argument('--out-data-dir', default='data/voc')
    p.add_argument('--names-file', default='data/voc.names')
    p.add_argument('--classes', type=int, default=20)
    args = p.parse_args()

    images_root = args.images_root
    ann_root = args.annotations_root
    out_dir = args.out_data_dir

    names_map = {n: i for i, n in enumerate(VOC_CLASSES)}

    # prepare output dirs
    images_train = os.path.join(out_dir, 'images', 'train')
    images_val = os.path.join(out_dir, 'images', 'val')
    labels_train = os.path.join(out_dir, 'labels', 'train')
    labels_val = os.path.join(out_dir, 'labels', 'val')
    ensure_dir(labels_train)
    ensure_dir(labels_val)
    ensure_dir(os.path.join(out_dir, 'images', 'train'))
    ensure_dir(os.path.join(out_dir, 'images', 'val'))

    # discover images
    train_dir = os.path.join(images_root, 'train')
    val_dir = os.path.join(images_root, 'val')
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print('Expected images directory with train/ and val/ subfolders under', images_root)
        sys.exit(1)

    image_exts = ('*.jpg', '*.jpeg', '*.png')
    train_imgs = []
    val_imgs = []
    for ext in image_exts:
        train_imgs.extend(sorted(glob.glob(os.path.join(train_dir, ext))))
        val_imgs.extend(sorted(glob.glob(os.path.join(val_dir, ext))))

    # Convert XML -> YOLO labels if XML annotations exist
    converted = 0
    for img_list, lbl_dir in ((train_imgs, labels_train), (val_imgs, labels_val)):
        for img in img_list:
            base = os.path.splitext(os.path.basename(img))[0]
            out_lbl = os.path.join(lbl_dir, base + '.txt')
            # prefer existing YOLO label in a path mirroring images (labels may already exist)
            # check image sibling labels first
            sibling_lbl = os.path.join(os.path.dirname(img), '..', 'labels', os.path.basename(img).replace(os.path.splitext(img)[1], '.txt'))
            if os.path.exists(out_lbl):
                continue
            xmlp = find_xml_for_image(img, ann_root)
            if xmlp:
                lines = xml_to_yolo(xmlp, names_map)
                with open(out_lbl, 'w') as f:
                    f.write('\n'.join(lines) + ('\n' if lines else ''))
                converted += 1

    # Write train.txt and val.txt listing absolute paths to images
    out_train_list = os.path.join(out_dir, 'train.txt')
    out_val_list = os.path.join(out_dir, 'val.txt')
    with open(out_train_list, 'w') as f:
        for pth in train_imgs:
            f.write(os.path.abspath(pth) + '\n')
    with open(out_val_list, 'w') as f:
        for pth in val_imgs:
            f.write(os.path.abspath(pth) + '\n')

    # also create expected files for yolov3-master repo layout
    repo_data_dir = os.path.join('data', 'voc')
    ensure_dir(repo_data_dir)
    # copy train/val lists to repo data/voc/train.txt & val.txt
    import shutil
    shutil.copy(out_train_list, os.path.join(repo_data_dir, 'train.txt'))
    shutil.copy(out_val_list, os.path.join(repo_data_dir, 'val.txt'))

    # create names file if missing
    if not os.path.exists(args.names_file):
        ensure_dir(os.path.dirname(args.names_file) or '.')
        with open(args.names_file, 'w') as f:
            f.write('\n'.join(VOC_CLASSES) + '\n')

    print(f'Found {len(train_imgs)} train images and {len(val_imgs)} val images.')
    print(f'Converted {converted} XML annotations to YOLO txt labels (if XMLs were found).')
    print('Wrote:', os.path.join(repo_data_dir, 'train.txt'), os.path.join(repo_data_dir, 'val.txt'))
    print('Names file at', args.names_file)


if __name__ == '__main__':
    main()
