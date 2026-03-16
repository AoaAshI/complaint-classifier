"""
Image Dataset Integrity Checker

Scans a directory recursively for corrupted image files and removes them.
"""

import os
from PIL import Image

def verify_and_remove_corrupted_images(directory):
    """
    Scan the specified directory for corrupted image files and remove them.
    
    Args:
        directory (str): Path to the root directory containing images.
    """
    corrupted_images = []
    total_checked = 0

    print(f"Scanning '{directory}' for corrupted images...")

    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                path = os.path.join(root, file)
                total_checked += 1

                try:
                    with Image.open(path) as img:
                        img.verify()  # Validate image integrity
                except (IOError, SyntaxError, OSError) as e:
                    print(f"Corrupted image detected: {path}")
                    corrupted_images.append(path)

    # Summary of scan results
    print(f"\nScan Summary:")
    print(f"Total images checked: {total_checked}")
    print(f"Corrupted images found: {len(corrupted_images)}")

    # Remove corrupted images if any
    if corrupted_images:
        print("\nRemoving corrupted images...")
        removed_count = 0
        for path in corrupted_images:
            try:
                os.remove(path)
                print(f"Removed: {path}")
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {path}: {e}")

        print(f"\nCleanup complete. Removed {removed_count} corrupted images.")
    else:
        print("\nNo corrupted images found.")

if __name__ == '__main__':
    dataset_directory = 'data/raw/image_dataset'
    verify_and_remove_corrupted_images(dataset_directory)
