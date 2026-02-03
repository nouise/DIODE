"""
YOLOv3 Data Augmentation Module

This module provides three data augmentation techniques for YOLOv3 format datasets:
1. MosaicAugmentation: 2x2 grid mosaic merge
2. CopyPasteAugmentation: ROI extraction and copy-paste with collision detection
3. SyntheticLabelGenerator: Random synthetic label generation

Data Format (YOLO):
- Each label line: [class_id, x_center, y_center, width, height, confidence]
- All coordinates are normalized to 0-1
- Target canvas size: 512 x 512 pixels
"""

import cv2
import numpy as np
import random
import os
from pathlib import Path
from typing import List, Tuple, Optional


# ============================================================================
# Utility Functions
# ============================================================================

def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box format from [x_center, y_center, width, height] to [x1, y1, x2, y2].
    
    CRITICAL: Input may contain [class, x, y, w, h, conf]. 
    This function only operates on columns 1:5 (x, y, w, h).
    
    Args:
        x: Array of shape (N, 6) with format [class, x, y, w, h, conf]
           or (N, 4) with format [x, y, w, h]
    
    Returns:
        Array of shape (N, 4) with format [x1, y1, x2, y2]
    """
    if x.shape[1] == 6:
        # Extract only coordinate columns (indices 1:5)
        coords = x[:, 1:5].copy()
    elif x.shape[1] == 4:
        coords = x.copy()
    else:
        raise ValueError(f"Expected input shape (N, 4) or (N, 6), got {x.shape}")
    
    y = np.zeros_like(coords)
    y[:, 0] = coords[:, 0] - coords[:, 2] / 2  # x1 = x_center - width/2
    y[:, 1] = coords[:, 1] - coords[:, 3] / 2  # y1 = y_center - height/2
    y[:, 2] = coords[:, 0] + coords[:, 2] / 2  # x2 = x_center + width/2
    y[:, 3] = coords[:, 1] + coords[:, 3] / 2  # y2 = y_center + height/2
    
    return y


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box format from [x1, y1, x2, y2] to [x_center, y_center, width, height].
    
    Args:
        x: Array of shape (N, 4) with format [x1, y1, x2, y2]
    
    Returns:
        Array of shape (N, 4) with format [x_center, y_center, width, height]
    """
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x_center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y_center
    y[:, 2] = x[:, 2] - x[:, 0]         # width
    y[:, 3] = x[:, 3] - x[:, 1]         # height
    
    return y


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU (Intersection over Union) between two sets of boxes.
    
    Args:
        boxes1: Array of shape (N, 4) with format [x1, y1, x2, y2]
        boxes2: Array of shape (M, 4) with format [x1, y1, x2, y2]
    
    Returns:
        IoU matrix of shape (N, M)
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    # Broadcast to get intersection coordinates
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2] - left-top
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2] - right-bottom
    
    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou


def load_yolo_labels(label_path: str) -> np.ndarray:
    """
    Load YOLO format labels from file.
    
    Args:
        label_path: Path to label file
    
    Returns:
        Array of shape (N, 6) with format [class, x, y, w, h, conf]
        Returns empty array if file doesn't exist
    """
    if not os.path.exists(label_path):
        return np.zeros((0, 6))
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    # Handle both 5-column (no conf) and 6-column (with conf) formats
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    labels.append([class_id, x, y, w, h, conf])
    
    return np.array(labels) if labels else np.zeros((0, 6))


def save_yolo_labels(label_path: str, labels: np.ndarray):
    """
    Save YOLO format labels to file.
    
    Args:
        label_path: Path to save label file
        labels: Array of shape (N, 6) with format [class, x, y, w, h, conf]
    """
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    
    with open(label_path, 'w') as f:
        for label in labels:
            class_id = int(label[0])
            x, y, w, h, conf = label[1:]
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")


def img_path_to_label_path(img_path: str) -> str:
    """
    Convert image path to label path by replacing 'images' with 'labels' and extension with '.txt'.
    
    Args:
        img_path: Path to image file
    
    Returns:
        Path to corresponding label file
    """
    # Replace 'images' with 'labels' and change extension to .txt
    label_path = img_path.replace('images', 'labels')
    label_path = os.path.splitext(label_path)[0] + '.txt'
    return label_path


# ============================================================================
# Mosaic Augmentation (2x2 Grid)
# ============================================================================

class MosaicAugmentation:
    """
    2x2 Mosaic Augmentation for YOLOv3.
    
    Merges 4 images into a 512x512 canvas in a fixed 2x2 grid pattern.
    Each image is resized to 256x256 and placed in one of the four quadrants.
    """
    
    def __init__(self, output_size: int = 512):
        """
        Args:
            output_size: Size of output mosaic image (default: 512)
        """
        self.output_size = output_size
        self.grid_size = output_size // 2  # 256 for 512x512 output
    
    def create_mosaic(self, 
                     img_paths: List[str], 
                     output_img_path: str,
                     output_label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a 2x2 mosaic from 4 images.
        
        Args:
            img_paths: List of 4 image paths
            output_img_path: Path to save output image
            output_label_path: Path to save output labels
        
        Returns:
            Tuple of (mosaic_image, mosaic_labels)
        """
        if len(img_paths) != 4:
            raise ValueError(f"Expected 4 images, got {len(img_paths)}")
        
        # Create blank canvas
        mosaic_img = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        mosaic_labels = []
        
        # Define quadrant positions: [top-left, top-right, bottom-left, bottom-right]
        positions = [
            (0, 0),           # Top-left
            (self.grid_size, 0),           # Top-right
            (0, self.grid_size),           # Bottom-left
            (self.grid_size, self.grid_size)  # Bottom-right
        ]
        
        for i, (img_path, (offset_x, offset_y)) in enumerate(zip(img_paths, positions)):
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}, using blank")
                img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            
            # Resize to grid_size x grid_size
            img_resized = cv2.resize(img, (self.grid_size, self.grid_size))
            
            # Place on canvas
            mosaic_img[offset_y:offset_y + self.grid_size, 
                      offset_x:offset_x + self.grid_size] = img_resized
            
            # Load and transform labels
            label_path = img_path_to_label_path(img_path)
            labels = load_yolo_labels(label_path)
            
            if len(labels) > 0:
                # Transform labels for this quadrant
                transformed_labels = self._transform_labels(
                    labels, offset_x, offset_y
                )
                mosaic_labels.append(transformed_labels)
        
        # Concatenate all labels
        if mosaic_labels:
            mosaic_labels = np.vstack(mosaic_labels)
        else:
            mosaic_labels = np.zeros((0, 6))
        
        # Save outputs
        cv2.imwrite(output_img_path, mosaic_img)
        save_yolo_labels(output_label_path, mosaic_labels)
        
        return mosaic_img, mosaic_labels
    
    def _transform_labels(self, 
                         labels: np.ndarray, 
                         offset_x: int, 
                         offset_y: int) -> np.ndarray:
        """
        Transform labels for a specific quadrant position.
        
        Formula:
            new_w = old_w * 0.5
            new_h = old_h * 0.5
            new_x = (old_x * 0.5) + (offset_x / output_size)
            new_y = (old_y * 0.5) + (offset_y / output_size)
        
        Args:
            labels: Original labels (N, 6)
            offset_x: X offset in pixels
            offset_y: Y offset in pixels
        
        Returns:
            Transformed labels (N, 6)
        """
        if len(labels) == 0:
            return labels
        
        transformed = labels.copy()
        
        # Scale factor for size
        scale = 0.5
        
        # Transform coordinates (columns 1:5 are x, y, w, h)
        transformed[:, 3] = labels[:, 3] * scale  # new_w = old_w * 0.5
        transformed[:, 4] = labels[:, 4] * scale  # new_h = old_h * 0.5
        transformed[:, 1] = (labels[:, 1] * scale) + (offset_x / self.output_size)  # new_x
        transformed[:, 2] = (labels[:, 2] * scale) + (offset_y / self.output_size)  # new_y
        
        return transformed


# ============================================================================
# Copy-Paste Augmentation with Collision Detection
# ============================================================================

class CopyPasteAugmentation:
    """
    Copy-Paste Augmentation for YOLOv3.
    
    Extracts objects from source images and pastes them onto a random background
    with collision detection to avoid overlapping objects.
    """
    
    def __init__(self, 
                 output_size: int = 512,
                 iou_threshold: float = 0.1,
                 max_attempts: int = 50):
        """
        Args:
            output_size: Size of output image (default: 512)
            iou_threshold: Maximum allowed IoU with existing boxes (default: 0.1)
            max_attempts: Maximum attempts to find non-overlapping position (default: 50)
        """
        self.output_size = output_size
        self.iou_threshold = iou_threshold
        self.max_attempts = max_attempts
    
    def create_copy_paste(self,
                         source_img_paths: List[str],
                         background_img_path: str,
                         output_img_path: str,
                         output_label_path: str,
                         objects_per_image: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a copy-paste augmented image.
        
        Args:
            source_img_paths: List of source image paths to extract objects from
            background_img_path: Path to background image (will be resized to output_size)
            output_img_path: Path to save output image
            output_label_path: Path to save output labels
            objects_per_image: Number of objects to paste (None = all available)
        
        Returns:
            Tuple of (output_image, output_labels)
        """
        # Load and resize background
        bg_img = cv2.imread(background_img_path)
        if bg_img is None:
            print(f"Warning: Could not load background {background_img_path}, using black canvas")
            canvas = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        else:
            canvas = cv2.resize(bg_img, (self.output_size, self.output_size))
        
        # Track placed boxes in pixel coordinates [x1, y1, x2, y2]
        placed_boxes_pixel = []
        output_labels = []
        
        # Collect all objects from source images
        all_objects = []
        for src_path in source_img_paths:
            objects = self._extract_objects(src_path)
            all_objects.extend(objects)
        
        if not all_objects:
            print("Warning: No objects found in source images")
            cv2.imwrite(output_img_path, canvas)
            save_yolo_labels(output_label_path, np.zeros((0, 6)))
            return canvas, np.zeros((0, 6))
        
        # Limit number of objects if specified
        if objects_per_image is not None:
            random.shuffle(all_objects)
            all_objects = all_objects[:objects_per_image]
        
        # Paste each object onto canvas
        for obj_img, class_id, conf in all_objects:
            success, bbox = self._paste_object(
                canvas, obj_img, placed_boxes_pixel
            )
            
            if success:
                # Convert pixel bbox to normalized YOLO format
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2 / self.output_size
                y_center = (y1 + y2) / 2 / self.output_size
                width = (x2 - x1) / self.output_size
                height = (y2 - y1) / self.output_size
                
                output_labels.append([class_id, x_center, y_center, width, height, conf])
        
        # Save outputs
        output_labels = np.array(output_labels) if output_labels else np.zeros((0, 6))
        cv2.imwrite(output_img_path, canvas)
        save_yolo_labels(output_label_path, output_labels)
        
        return canvas, output_labels
    
    def _extract_objects(self, img_path: str) -> List[Tuple[np.ndarray, int, float]]:
        """
        Extract all objects from an image based on its labels.
        
        Args:
            img_path: Path to source image
        
        Returns:
            List of tuples (cropped_object_img, class_id, confidence)
        """
        img = cv2.imread(img_path)
        if img is None:
            return []
        
        h, w = img.shape[:2]
        label_path = img_path_to_label_path(img_path)
        labels = load_yolo_labels(label_path)
        
        objects = []
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
            
            # Extract ROI
            if x2 > x1 and y2 > y1:
                roi = img[y1:y2, x1:x2].copy()
                objects.append((roi, class_id, conf))
        
        return objects
    
    def _paste_object(self,
                     canvas: np.ndarray,
                     obj_img: np.ndarray,
                     placed_boxes: List[List[int]]) -> Tuple[bool, Optional[List[int]]]:
        """
        Attempt to paste an object onto canvas without collision.
        
        Args:
            canvas: Canvas image to paste onto (modified in-place)
            obj_img: Object image to paste
            placed_boxes: List of already placed boxes [x1, y1, x2, y2]
        
        Returns:
            Tuple of (success, bbox) where bbox is [x1, y1, x2, y2] or None
        """
        obj_h, obj_w = obj_img.shape[:2]
        
        # Try multiple random positions
        for _ in range(self.max_attempts):
            # Random position ensuring object stays within bounds
            max_x = self.output_size - obj_w
            max_y = self.output_size - obj_h
            
            if max_x <= 0 or max_y <= 0:
                # Object too large for canvas
                return False, None
            
            x1 = random.randint(0, max_x)
            y1 = random.randint(0, max_y)
            x2 = x1 + obj_w
            y2 = y1 + obj_h
            
            new_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            
            # Check collision with existing boxes
            if placed_boxes:
                existing_boxes = np.array(placed_boxes, dtype=np.float32)
                ious = box_iou(new_box, existing_boxes)
                
                if np.any(ious > self.iou_threshold):
                    continue  # Collision detected, try again
            
            # No collision, paste the object
            canvas[y1:y2, x1:x2] = obj_img
            placed_boxes.append([x1, y1, x2, y2])
            
            return True, [x1, y1, x2, y2]
        
        # Failed to find non-overlapping position
        return False, None


# ============================================================================
# Synthetic Label Generator
# ============================================================================

class SyntheticLabelGenerator:
    """
    Generate synthetic YOLO format labels without images.
    
    Useful for anchor matching tests or stress testing.
    """
    
    def __init__(self,
                 num_classes: int = 80,
                 min_boxes: int = 4,
                 max_boxes: int = 9,
                 min_size: float = 0.05,
                 max_size: float = 0.3,
                 min_conf: float = 0.5,
                 max_conf: float = 1.0):
        """
        Args:
            num_classes: Number of object classes
            min_boxes: Minimum number of boxes per image
            max_boxes: Maximum number of boxes per image
            min_size: Minimum box size (normalized, default: 0.05)
            max_size: Maximum box size (normalized, default: 0.3)
            min_conf: Minimum confidence score
            max_conf: Maximum confidence score
        """
        self.num_classes = num_classes
        self.min_boxes = min_boxes
        self.max_boxes = max_boxes
        self.min_size = min_size
        self.max_size = max_size
        self.min_conf = min_conf
        self.max_conf = max_conf
    
    def generate_labels(self, output_label_path: str) -> np.ndarray:
        """
        Generate random synthetic labels.
        
        Args:
            output_label_path: Path to save generated labels
        
        Returns:
            Generated labels array of shape (N, 6)
        """
        num_boxes = random.randint(self.min_boxes, self.max_boxes)
        labels = []
        
        for _ in range(num_boxes):
            # Random class
            class_id = random.randint(0, self.num_classes - 1)
            
            # Random size
            w = random.uniform(self.min_size, self.max_size)
            h = random.uniform(self.min_size, self.max_size)
            
            # Random center ensuring box stays within bounds
            # Constraint: x - w/2 >= 0 and x + w/2 <= 1
            x = random.uniform(w / 2, 1 - w / 2)
            y = random.uniform(h / 2, 1 - h / 2)
            
            # Random confidence
            conf = random.uniform(self.min_conf, self.max_conf)
            
            labels.append([class_id, x, y, w, h, conf])
        
        labels = np.array(labels)
        save_yolo_labels(output_label_path, labels)
        
        return labels
    
    def generate_batch(self, 
                      output_dir: str, 
                      num_samples: int,
                      prefix: str = "synthetic") -> List[str]:
        """
        Generate a batch of synthetic labels.
        
        Args:
            output_dir: Directory to save labels
            num_samples: Number of label files to generate
            prefix: Filename prefix
        
        Returns:
            List of generated label file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        label_paths = []
        for i in range(num_samples):
            label_path = os.path.join(output_dir, f"{prefix}_{i:06d}.txt")
            self.generate_labels(label_path)
            label_paths.append(label_path)
        
        return label_paths


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("YOLOv3 Data Augmentation Module")
    print("=" * 60)
    print("\nAvailable augmentation techniques:")
    print("1. MosaicAugmentation - 2x2 grid mosaic merge")
    print("2. CopyPasteAugmentation - Object copy-paste with collision detection")
    print("3. SyntheticLabelGenerator - Random synthetic label generation")
    print("\nUse the classes above in your training pipeline or see examples in demo script.")
