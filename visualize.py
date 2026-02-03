"""
Visualization Script for YOLOv3 Data Augmentation

This script visualizes YOLO format labels by drawing bounding boxes on images.
Useful for validating augmentation results.

Usage:
    python visualize.py --image <path> --label <path> --output <path>
    python visualize.py --dir <directory> --output <output_dir>
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from typing import List, Tuple
import random


# COCO class names for reference (can be customized)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for each class."""
    random.seed(42)  # Fixed seed for consistent colors
    colors = []
    for _ in range(num_classes):
        colors.append((
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
    return colors


def load_yolo_labels(label_path: str) -> np.ndarray:
    """Load YOLO format labels."""
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return np.zeros((0, 6))
    
    labels = []
    with open(label_path, 'r') as f:
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


def draw_boxes(image: np.ndarray, 
               labels: np.ndarray, 
               colors: List[Tuple[int, int, int]],
               class_names: List[str] = None,
               thickness: int = 2,
               font_scale: float = 0.5) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (BGR format)
        labels: YOLO labels of shape (N, 6) [class, x, y, w, h, conf]
        colors: List of BGR colors for each class
        class_names: Optional list of class names
        thickness: Box line thickness
        font_scale: Font scale for text
    
    Returns:
        Image with drawn boxes
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    for label in labels:
        class_id = int(label[0])
        x_center, y_center, width, height = label[1:5]
        conf = label[5]
        
        # Convert normalized coords to pixel coords
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        # Clip to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        if class_names and class_id < len(class_names):
            label_text = f"{class_names[class_id]}: {conf:.2f}"
        else:
            label_text = f"Class {class_id}: {conf:.2f}"
        
        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(img, 
                     (x1, y1 - text_h - baseline - 5), 
                     (x1 + text_w, y1), 
                     color, -1)
        
        # Draw label text
        cv2.putText(img, label_text, 
                   (x1, y1 - baseline - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), 1)
    
    return img


def visualize_single(image_path: str, 
                    label_path: str, 
                    output_path: str,
                    class_names: List[str] = None,
                    show: bool = False):
    """
    Visualize a single image with its labels.
    
    Args:
        image_path: Path to image file
        label_path: Path to label file
        output_path: Path to save visualization
        class_names: Optional list of class names
        show: Whether to display the image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Load labels
    labels = load_yolo_labels(label_path)
    
    # Generate colors
    max_class_id = int(labels[:, 0].max()) if len(labels) > 0 else 0
    num_colors = max(max_class_id + 1, 80)
    colors = generate_colors(num_colors)
    
    # Draw boxes
    img_with_boxes = draw_boxes(img, labels, colors, class_names)
    
    # Add info text
    info_text = f"Objects: {len(labels)} | Image: {os.path.basename(image_path)}"
    cv2.putText(img_with_boxes, info_text, 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 255, 0), 2)
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_with_boxes)
    print(f"Saved visualization to: {output_path}")
    
    # Show if requested
    if show:
        cv2.imshow('Visualization', img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_directory(image_dir: str, 
                        label_dir: str, 
                        output_dir: str,
                        class_names: List[str] = None,
                        max_images: int = None):
    """
    Visualize all images in a directory.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        output_dir: Directory to save visualizations
        class_names: Optional list of class names
        max_images: Maximum number of images to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images")
    
    for i, img_path in enumerate(image_files):
        # Find corresponding label
        label_name = img_path.stem + '.txt'
        label_path = Path(label_dir) / label_name
        
        # Output path
        output_path = Path(output_dir) / f"{img_path.stem}_viz.jpg"
        
        print(f"[{i+1}/{len(image_files)}] Processing {img_path.name}...")
        visualize_single(str(img_path), str(label_path), str(output_path), class_names)
    
    print(f"\nDone! Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize YOLOv3 labels on images')
    
    # Single image mode
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--label', type=str, help='Path to single label file')
    
    # Directory mode
    parser.add_argument('--image-dir', type=str, help='Directory containing images')
    parser.add_argument('--label-dir', type=str, help='Directory containing labels')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process')
    
    # Common arguments
    parser.add_argument('--output', type=str, required=True, 
                       help='Output path (file for single image, directory for batch)')
    parser.add_argument('--show', action='store_true', 
                       help='Display the image (single image mode only)')
    parser.add_argument('--classes', type=str, 
                       help='Path to class names file (one per line)')
    
    args = parser.parse_args()
    
    # Load class names if provided
    class_names = None
    if args.classes and os.path.exists(args.classes):
        with open(args.classes, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = COCO_CLASSES
    
    # Single image mode
    if args.image and args.label:
        visualize_single(args.image, args.label, args.output, class_names, args.show)
    
    # Directory mode
    elif args.image_dir:
        label_dir = args.label_dir if args.label_dir else args.image_dir.replace('images', 'labels')
        visualize_directory(args.image_dir, label_dir, args.output, class_names, args.max_images)
    
    else:
        parser.print_help()
        print("\nError: Specify either --image/--label or --image-dir")


if __name__ == "__main__":
    main()
