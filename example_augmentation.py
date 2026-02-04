"""
Example Usage Script for YOLOv3 Data Augmentation Module

This script demonstrates how to use the three augmentation techniques:
1. Mosaic Augmentation (2x2 grid)
2. Copy-Paste Augmentation
3. Synthetic Label Generation

The script reads image paths from a text file (like random_328.txt) and 
generates augmented data.
"""

import os
import random
from pathlib import Path
from data_augmentation import (
    MosaicAugmentation,
    CopyPasteAugmentation,
    SyntheticLabelGenerator,
    img_path_to_label_path
)


def load_image_list(txt_file: str) -> list:
    """Load image paths from text file."""
    with open(txt_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


def example_mosaic_augmentation(image_list: list, output_dir: str, num_samples: int = 10):
    """
    Example: Create mosaic augmented images.
    
    Args:
        image_list: List of image paths
        output_dir: Directory to save outputs
        num_samples: Number of mosaic images to generate
    """
    print("\n" + "="*60)
    print("Example 1: Mosaic Augmentation (2x2 Grid)")
    print("="*60)
    
    # Create output directories
    img_output_dir = os.path.join(output_dir, 'mosaic', 'images')
    label_output_dir = os.path.join(output_dir, 'mosaic', 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    
    # Initialize mosaic augmentation
    mosaic_aug = MosaicAugmentation(output_size=512)
    
    # Generate mosaic images
    for i in range(num_samples):
        # Randomly select 4 images
        selected_imgs = random.sample(image_list, 4)
        
        # Define output paths
        output_img = os.path.join(img_output_dir, f'mosaic_{i:04d}.jpg')
        output_label = os.path.join(label_output_dir, f'mosaic_{i:04d}.txt')
        
        try:
            # Create mosaic
            mosaic_img, mosaic_labels = mosaic_aug.create_mosaic(
                selected_imgs, output_img, output_label
            )
            
            print(f"[{i+1}/{num_samples}] Created mosaic: {output_img}")
            print(f"  - Source images: {len(selected_imgs)}")
            print(f"  - Total objects: {len(mosaic_labels)}")
        except Exception as e:
            print(f"[{i+1}/{num_samples}] Error creating mosaic: {e}")
    
    print(f"\nMosaic images saved to: {img_output_dir}")
    print(f"Mosaic labels saved to: {label_output_dir}")


def example_copypaste_augmentation(image_list: list, output_dir: str, num_samples: int = 10):
    """
    Example: Create copy-paste augmented images.
    
    Args:
        image_list: List of image paths
        output_dir: Directory to save outputs
        num_samples: Number of copy-paste images to generate
    """
    print("\n" + "="*60)
    print("Example 2: Copy-Paste Augmentation")
    print("="*60)
    
    # Create output directories
    img_output_dir = os.path.join(output_dir, 'copypaste', 'images')
    label_output_dir = os.path.join(output_dir, 'copypaste', 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    
    # Initialize copy-paste augmentation
    copypaste_aug = CopyPasteAugmentation(
        output_size=512,
        iou_threshold=0.1,
        max_attempts=50
    )
    
    # Generate copy-paste images
    for i in range(num_samples):
        # Randomly select source images (for extracting objects)
        num_sources = random.randint(2, 5)
        source_imgs = random.sample(image_list, num_sources)
        
        # Randomly select background image
        background_img = random.choice(image_list)
        
        # Define output paths
        output_img = os.path.join(img_output_dir, f'copypaste_{i:04d}.jpg')
        output_label = os.path.join(label_output_dir, f'copypaste_{i:04d}.txt')
        
        try:
            # Create copy-paste image
            cp_img, cp_labels = copypaste_aug.create_copy_paste(
                source_img_paths=source_imgs,
                background_img_path=background_img,
                output_img_path=output_img,
                output_label_path=output_label,
                objects_per_image=None  # Paste all available objects
            )
            
            print(f"[{i+1}/{num_samples}] Created copy-paste: {output_img}")
            print(f"  - Source images: {len(source_imgs)}")
            print(f"  - Objects pasted: {len(cp_labels)}")
        except Exception as e:
            print(f"[{i+1}/{num_samples}] Error creating copy-paste: {e}")
    
    print(f"\nCopy-paste images saved to: {img_output_dir}")
    print(f"Copy-paste labels saved to: {label_output_dir}")


def example_synthetic_labels(output_dir: str, num_samples: int = 100):
    """
    Example: Generate synthetic labels.
    
    Args:
        output_dir: Directory to save outputs
        num_samples: Number of synthetic label files to generate
    """
    print("\n" + "="*60)
    print("Example 3: Synthetic Label Generation")
    print("="*60)
    
    # Create output directory
    label_output_dir = os.path.join(output_dir, 'synthetic', 'labels')
    os.makedirs(label_output_dir, exist_ok=True)
    
    # Initialize synthetic label generator
    label_gen = SyntheticLabelGenerator(
        num_classes=80,
        min_boxes=4,
        max_boxes=9,
        min_size=0.05,
        max_size=0.3,
        min_conf=0.5,
        max_conf=1.0
    )
    
    # Generate labels
    label_paths = label_gen.generate_batch(
        output_dir=label_output_dir,
        num_samples=num_samples,
        prefix="synthetic"
    )
    
    print(f"Generated {len(label_paths)} synthetic label files")
    
    # Statistics
    total_boxes = 0
    for label_path in label_paths:
        with open(label_path, 'r') as f:
            num_boxes = sum(1 for line in f if line.strip())
            total_boxes += num_boxes
    
    avg_boxes = total_boxes / len(label_paths)
    print(f"  - Total boxes: {total_boxes}")
    print(f"  - Average boxes per file: {avg_boxes:.2f}")
    print(f"\nSynthetic labels saved to: {label_output_dir}")


def main():
    """Main demonstration script."""
    print("\n" + "="*70)
    print("YOLOv3 Data Augmentation Module - Usage Examples")
    print("="*70)
    
    # Configuration
    IMAGE_LIST_FILE = "/data1/home/ypliu/DIODE/augmented_data/selected_voc_train_164*4.txt"  # Path to text file with image paths
    OUTPUT_DIR = "augmented_data"       # Output directory for augmented data
    
    # Check if image list file exists
    if not os.path.exists(IMAGE_LIST_FILE):
        print(f"\nError: Image list file not found: {IMAGE_LIST_FILE}")
        print("Please provide a text file with image paths (one per line)")
        print("Example format:")
        print("  /path/to/image1.jpg")
        print("  /path/to/image2.jpg")
        print("  ...")
        return
    
    # Load image list
    print(f"\nLoading image list from: {IMAGE_LIST_FILE}")
    image_list = load_image_list(IMAGE_LIST_FILE)
    print(f"Loaded {len(image_list)} image paths")
    
    if len(image_list) < 4:
        print("\nError: Need at least 4 images for mosaic augmentation")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run examples
    try:
        # Example 1: Mosaic Augmentation
        example_mosaic_augmentation(
            image_list=image_list,
            output_dir=OUTPUT_DIR,
            num_samples=164
        )
        
        # # Example 2: Copy-Paste Augmentation
        # example_copypaste_augmentation(
        #     image_list=image_list,
        #     output_dir=OUTPUT_DIR,
        #     num_samples=10
        # )
        
        # # Example 3: Synthetic Labels
        # example_synthetic_labels(
        #     output_dir=OUTPUT_DIR,
        #     num_samples=100
        # )
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nTo visualize the results, run:")
        print(f"  python visualize.py --image-dir {OUTPUT_DIR}/mosaic/images --output viz_mosaic")
        print(f"  python visualize.py --image-dir {OUTPUT_DIR}/copypaste/images --output viz_copypaste")
        
    except Exception as e:
        print(f"\nError during augmentation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
