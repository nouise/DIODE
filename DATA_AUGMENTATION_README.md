# YOLOv3 Data Augmentation Module

A comprehensive Python module for YOLOv3 data augmentation with three powerful techniques: Mosaic, Copy-Paste, and Synthetic Label Generation.

## Features

### 1. **Mosaic Augmentation (2x2 Grid)**
- Merges 4 images into a 512×512 canvas
- Fixed quadrant placement for predictable results
- Automatic label transformation with precise coordinate mapping
- Maintains YOLO format (6-column: class, x, y, w, h, conf)

### 2. **Copy-Paste Augmentation**
- Extracts objects from source images based on labels
- Pastes onto random background images
- **Collision Detection**: IoU-based overlap avoidance (configurable threshold)
- **Boundary Protection**: Ensures objects stay within canvas
- Configurable retry attempts for placement

### 3. **Synthetic Label Generator**
- Generates random YOLO labels without images
- Useful for anchor matching tests and stress testing
- Configurable: number of boxes, size range, confidence range
- Automatic boundary constraint validation

## Installation

Required dependencies:
```bash
pip install opencv-python numpy
```

## File Structure

```
DIODE/
├── data_augmentation.py      # Main module with all augmentation classes
├── visualize.py               # Visualization script for validation
├── example_augmentation.py    # Usage examples and demo
└── random_328.txt            # Sample image list file
```

## Quick Start

### Basic Usage

```python
from data_augmentation import (
    MosaicAugmentation,
    CopyPasteAugmentation,
    SyntheticLabelGenerator
)

# 1. Mosaic Augmentation
mosaic = MosaicAugmentation(output_size=512)
mosaic_img, mosaic_labels = mosaic.create_mosaic(
    img_paths=['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
    output_img_path='output/mosaic.jpg',
    output_label_path='output/mosaic.txt'
)

# 2. Copy-Paste Augmentation
copypaste = CopyPasteAugmentation(
    output_size=512,
    iou_threshold=0.1,
    max_attempts=50
)
cp_img, cp_labels = copypaste.create_copy_paste(
    source_img_paths=['src1.jpg', 'src2.jpg'],
    background_img_path='background.jpg',
    output_img_path='output/copypaste.jpg',
    output_label_path='output/copypaste.txt'
)

# 3. Synthetic Labels
label_gen = SyntheticLabelGenerator(
    num_classes=80,
    min_boxes=4,
    max_boxes=9,
    min_size=0.05,
    max_size=0.3
)
labels = label_gen.generate_labels('output/synthetic.txt')
```

### Run Example Script

```bash
# Generate augmented data from image list
python example_augmentation.py

# Visualize results
python visualize.py --image-dir augmented_data/mosaic/images --output viz_mosaic
python visualize.py --image-dir augmented_data/copypaste/images --output viz_copypaste
```

## Input Format

### Image List File
A text file containing image paths (one per line):
```
/path/to/images/image1.jpg
/path/to/images/image2.jpg
/path/to/images/image3.jpg
...
```

### Label Format (YOLO)
Each label file contains one object per line:
```
class_id x_center y_center width height confidence
0 0.5 0.5 0.3 0.4 0.95
1 0.2 0.3 0.1 0.2 0.88
```
- All coordinates normalized to [0, 1]
- Labels stored in `labels/` directory (parallel to `images/`)

## API Reference

### MosaicAugmentation

```python
mosaic = MosaicAugmentation(output_size=512)
```

**Parameters:**
- `output_size` (int): Output canvas size (default: 512)

**Methods:**
- `create_mosaic(img_paths, output_img_path, output_label_path)`
  - Requires exactly 4 image paths
  - Returns: (mosaic_image, mosaic_labels)

### CopyPasteAugmentation

```python
copypaste = CopyPasteAugmentation(
    output_size=512,
    iou_threshold=0.1,
    max_attempts=50
)
```

**Parameters:**
- `output_size` (int): Output canvas size (default: 512)
- `iou_threshold` (float): Max allowed IoU with existing boxes (default: 0.1)
- `max_attempts` (int): Max retries for placement (default: 50)

**Methods:**
- `create_copy_paste(source_img_paths, background_img_path, output_img_path, output_label_path, objects_per_image=None)`
  - `source_img_paths`: Images to extract objects from
  - `background_img_path`: Background image (randomly selected from dataset)
  - `objects_per_image`: Limit number of pasted objects (None = all)
  - Returns: (output_image, output_labels)

### SyntheticLabelGenerator

```python
label_gen = SyntheticLabelGenerator(
    num_classes=80,
    min_boxes=4,
    max_boxes=9,
    min_size=0.05,
    max_size=0.3,
    min_conf=0.5,
    max_conf=1.0
)
```

**Parameters:**
- `num_classes` (int): Number of object classes (default: 80)
- `min_boxes` (int): Min boxes per image (default: 4)
- `max_boxes` (int): Max boxes per image (default: 9)
- `min_size` (float): Min normalized box size (default: 0.05)
- `max_size` (float): Max normalized box size (default: 0.3)
- `min_conf` (float): Min confidence (default: 0.5)
- `max_conf` (float): Max confidence (default: 1.0)

**Methods:**
- `generate_labels(output_label_path)` - Generate single label file
- `generate_batch(output_dir, num_samples, prefix)` - Generate batch

## Visualization

```bash
# Visualize single image
python visualize.py --image path/to/image.jpg --label path/to/label.txt --output output.jpg --show

# Visualize directory
python visualize.py --image-dir path/to/images --output viz_output --max-images 50

# With custom class names
python visualize.py --image-dir images --output viz --classes coco.names
```

## Technical Details

### Coordinate Transformation (Mosaic)

For each quadrant in the 2×2 grid:
```python
new_w = old_w * 0.5
new_h = old_h * 0.5
new_x = (old_x * 0.5) + (offset_x / 512)
new_y = (old_y * 0.5) + (offset_y / 512)
```

Quadrant offsets:
- Top-left: (0, 0)
- Top-right: (256, 0)
- Bottom-left: (0, 256)
- Bottom-right: (256, 256)

### Collision Detection (Copy-Paste)

1. Extract ROI from source using YOLO coordinates
2. Generate random position on canvas
3. Check IoU with all existing boxes
4. If IoU > threshold, retry up to `max_attempts`
5. Ensure object stays within [0, 512] bounds

### Data Format Compliance

All functions handle 6-column YOLO format:
```
[class_id, x_center, y_center, width, height, confidence]
```

**Critical**: Coordinate operations (xywh↔xyxy) only modify columns 1:5, preserving class and confidence.

## Validation

### Acceptance Criteria
✅ Boxes precisely wrap objects (no offset)  
✅ Mosaic quadrant placement correct  
✅ Copy-paste collision avoidance works  
✅ Synthetic labels stay within bounds  
✅ All outputs in valid YOLO format  

### Visual Verification
Use `visualize.py` to verify:
- Boxes align with objects
- No overlapping boxes in copy-paste
- All boxes within canvas boundaries

## Example Output

After running `example_augmentation.py`:
```
augmented_data/
├── mosaic/
│   ├── images/
│   │   ├── mosaic_0000.jpg
│   │   └── ...
│   └── labels/
│       ├── mosaic_0000.txt
│       └── ...
├── copypaste/
│   ├── images/
│   └── labels/
└── synthetic/
    └── labels/
```

## License

See [LICENSE](LICENSE) file in the project root.

## Contributing

This module is designed for YOLOv3 training data augmentation. Contributions welcome for:
- Additional augmentation techniques
- Performance optimizations
- Bug fixes and validation improvements
