import os
import sys
import glob
import hashlib
import random
from datetime import datetime, timezone
from PIL import Image

# Force UTF-8 output for Windows console emoji support
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- PATH CONFIGURATION ---
SUPPORT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SUPPORT_DIR)

SOURCE_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "AIDER")
PROCESSED_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "aider_processed_images")

# The tracking file to prevent processing duplicates across multiple runs
TRACKER_FILE = os.path.join(PROJECT_ROOT, "data", "aider_processed_tracker.txt")

# Create output directories if they don't exist
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)

# Image standardization parameters (matching xBD pipeline conventions)
MAX_EDGE = 1280  # Cap longest edge for VRAM safety on T4 GPUs
MIN_EDGE = 256   # Minimum resolution gate
JPEG_QUALITY = 95


def load_processed_history():
    """Loads the set of already processed image base names from the tracker."""
    if not os.path.exists(TRACKER_FILE):
        return set()
    processed = set()
    with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('|')
                base = parts[0].strip()
                processed.add(base)
    return processed


def append_to_history(original, output, md5_hash, status):
    """Appends a processed image record to the tracker file."""
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"{original} | {output} | {md5_hash} | {timestamp} | {status}"
    with open(TRACKER_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def standardize_image(input_path, output_path, max_edge=MAX_EDGE):
    """
    Standardizes an image for VLM fine-tuning:
    - Skips naturally grayscale imagery.
    - Converts RGBA/P/etc. to RGB.
    - Caps longest edge to max_edge (preserving aspect ratio).
    - Rejects images below MIN_EDGE after scaling.
    - Saves as optimized JPEG.
    - Verifies output integrity.
    - Computes MD5 hash for deduplication audits.
    """
    try:
        with Image.open(input_path) as img:
            # Skip naturally grayscale imagery
            if img.mode == 'L':
                return False, "skipped_grayscale", "NONE"

            # Drop alpha/multispectral channels
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            # Dynamically downscale if it exceeds max_edge, preserving aspect ratio
            if max(width, height) > max_edge:
                scaling_factor = max_edge / float(max(width, height))
                new_size = (int(width * scaling_factor), int(height * scaling_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Minimum resolution gate
            if min(img.size) < MIN_EDGE:
                return False, "skipped_too_small", "NONE"

            # Save cleanly as JPEG
            img.save(output_path, "JPEG", quality=JPEG_QUALITY, optimize=True)

            # Integrity verification
            Image.open(output_path).verify()

            # Compute MD5 hash
            with open(output_path, "rb") as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()

            return True, "success", md5_hash

    except Exception as e:
        print(f"  ❌ Error processing {input_path}: {e}")
        return False, "failed_error", "NONE"


def main():
    print(f"🔍 Scanning {SOURCE_IMG_DIR} for AIDER images...")

    # Recursively grab all images from class subdirectories
    search_pattern = os.path.join(SOURCE_IMG_DIR, "**", "*.*")
    all_files = glob.glob(search_pattern, recursive=True)

    # Filter to valid image extensions
    img_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not img_files:
        print("No images found! Check that data/AIDER/ contains class subdirectories.")
        return

    # Shuffle to prevent sequential bias (deterministic seed for reproducibility)
    random.seed(42)
    random.shuffle(img_files)

    print(f"Found {len(img_files)} images across AIDER classes.")

    # Load tracker history
    processed_history = load_processed_history()
    print(f"Already processed in previous runs: {len(processed_history)}")

    successful = 0
    skipped_existing = 0
    failed = 0

    for img_path in img_files:
        # Extract the class folder name and original filename
        parent_dir = os.path.basename(os.path.dirname(img_path))
        filename = os.path.basename(img_path)

        # Build a flattened output name: class_filename.jpg
        # AIDER filenames already contain the class prefix, so check to avoid duplication
        if parent_dir not in filename:
            out_name = f"{parent_dir}_{os.path.splitext(filename)[0]}.jpg"
        else:
            out_name = f"{os.path.splitext(filename)[0]}.jpg"

        # Check tracker to skip already processed
        if out_name in processed_history:
            skipped_existing += 1
            continue

        output_path = os.path.join(PROCESSED_IMG_DIR, out_name)

        success, status, md5_hash = standardize_image(img_path, output_path)

        # Log to tracker immediately (resume-safe)
        append_to_history(out_name, out_name if success else "NONE", md5_hash, status)
        processed_history.add(out_name)

        if success:
            successful += 1
            if successful % 100 == 0:
                print(f"  [{successful}] Processed: {out_name}")
        else:
            failed += 1
            if status != "skipped_grayscale" and status != "skipped_too_small":
                print(f"  ⚠️  {status}: {out_name}")

    # Final Report
    print(f"\n{'='*60}")
    print(f"📊 AIDER STANDARDIZATION REPORT")
    print(f"{'='*60}")
    print(f"  Total scanned         : {len(img_files)}")
    print(f"  Already processed     : {skipped_existing}")
    print(f"  Newly processed       : {successful}")
    print(f"  Failed / Skipped      : {failed}")
    print(f"  Output directory      : {PROCESSED_IMG_DIR}")
    print(f"  Tracker file          : {TRACKER_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
