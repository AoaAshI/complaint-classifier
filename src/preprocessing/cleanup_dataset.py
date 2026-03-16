"""
Aggressive Image Dataset Cleaner

Removes corrupted and empty image files to prevent training errors.
"""

import os
from PIL import Image

def remove_corrupted_and_empty_images(path):
    """
    Scan directory recursively to detect and remove empty or corrupted images.

    Args:
        path (str): Root directory containing images to clean.
    """
    corrupted_count = 0
    empty_count = 0
    total_files = 0

    print(f"Starting aggressive cleanup of '{path}'")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    for subdir, _, files in os.walk(path):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue

            filename = os.path.join(subdir, file)
            total_files += 1

            try:
                # Remove empty files
                if os.path.getsize(filename) == 0:
                    os.remove(filename)
                    empty_count += 1
                    print(f"Removed empty file: {filename}")
                    continue

                # Verify and force load image to detect corruption
                with Image.open(filename) as img:
                    img.verify()
                with Image.open(filename) as img:
                    img.load()

            except Exception as e:
                # Remove problematic files
                try:
                    os.remove(filename)
                    corrupted_count += 1
                    print(f"Removed corrupted file: {filename} - Reason: {e}")
                except Exception as removal_error:
                    print(f"Failed to remove {filename}: {removal_error}")

    # Summary of cleanup results
    print(f"\nCleanup Results:")
    print(f"Total image files processed: {total_files}")
    print(f"Empty files removed: {empty_count}")
    print(f"Corrupted files removed: {corrupted_count}")
    print(f"Clean files remaining: {total_files - empty_count - corrupted_count}")

if __name__ == '__main__':
    data_folder = 'data/raw/image_dataset'
    remove_corrupted_and_empty_images(data_folder)
