#!/usr/bin/env python3
"""
BMP to Filtered+Normalized HDF5 Pipeline

This script recursively finds all BMP images in each subfolder of a root directory (default: ./image),
applies non-local means filtering, then normalizes them using the 'peak_align' method (peak shift)
with the default reference image, and saves the results as an HDF5 file per subfolder.
No intermediate TIFFs are created.

Usage:
    python bmp_to_filtered_normalized_hdf5.py --input_root image \
        --reference_image 3mM-0628-4-mix_20250628213544658_17_o.tiff \
        --h 6.0 --template_size 3 --search_size 10

"""
import os
import sys
import h5py
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import concurrent.futures

# Import filtering and normalization logic
from nonlocal_means_filter import apply_nonlocal_means
from image_normalization import ImageNormalizer

# Global variable for per-process ImageNormalizer
normalizer_global = None

def worker_initializer(reference_image_path):
    global normalizer_global
    from image_normalization import ImageNormalizer
    normalizer_global = ImageNormalizer(reference_image_path)

def parse_timestamp_from_filename(filename: str) -> datetime:
    import re
    timestamp_pattern = r'_(\d{17})_'
    match = re.search(timestamp_pattern, filename)
    if not match:
        raise ValueError(f"Could not extract timestamp from filename: {filename}")
    timestamp_str = match.group(1)
    year = int(timestamp_str[0:4])
    month = int(timestamp_str[4:6])
    day = int(timestamp_str[6:8])
    hour = int(timestamp_str[8:10])
    minute = int(timestamp_str[10:12])
    second = int(timestamp_str[12:14])
    millisecond = int(timestamp_str[14:17])
    microsecond = millisecond * 1000
    return datetime(year, month, day, hour, minute, second, microsecond)

def find_bmp_files(directory: Path):
    return sorted(directory.glob("*.bmp"))

def find_bmp_files_recursive(folder_path: Path):
    """
    Recursively find all BMP files in a folder and its subfolders.
    Returns a list of Path objects.
    """
    return sorted(folder_path.rglob("*.bmp"))

def find_folders_with_bmps(input_root: Path):
    """
    Recursively find all folders under input_root that contain at least one BMP file (not counting subfolders).
    Returns a list of Path objects (folders).
    """
    folders = []
    for folder in input_root.rglob(""):
        if folder.is_dir():
            bmp_files = list(folder.glob("*.bmp"))
            if bmp_files:
                folders.append(folder)
    return folders

def find_all_bmp_folders_and_bmps(input_root: Path):
    """
    Returns a list of (folder_path, bmp_paths) for every folder under input_root that needs processing.
    Applies timestamp filtering for folders with >19 BMPs.
    Skips folders with existing HDF5.
    """
    folder_bmp_list = []
    for folder in input_root.rglob(""):
        if folder.is_dir():
            expected_h5_file = folder / f"{folder.name}_filtered_normalized_timeseries.h5"
            if expected_h5_file.exists():
                print(f"[DEBUG] Skipping folder {folder.name}: HDF5 file already exists")
                continue
            bmp_files = list(folder.glob("*.bmp"))
            if len(bmp_files) > 19:
                print(f"[DEBUG] Folder {folder.name} has {len(bmp_files)} BMPs, filtering to first 3 hours")
                bmp_with_timestamps = []
                for bmp_path in bmp_files:
                    try:
                        timestamp = parse_timestamp_from_filename(bmp_path.name)
                        bmp_with_timestamps.append((bmp_path, timestamp))
                    except Exception as e:
                        print(f"[DEBUG] Could not parse timestamp from {bmp_path.name}: {e}")
                        continue
                if not bmp_with_timestamps:
                    print(f"[DEBUG] No valid timestamps found in {folder.name}, skipping folder")
                    continue
                bmp_with_timestamps.sort(key=lambda x: x[1])
                earliest_timestamp = bmp_with_timestamps[0][1]
                three_hours_later = earliest_timestamp + timedelta(seconds=10800)
                filtered_bmps = [bmp_path for bmp_path, timestamp in bmp_with_timestamps if timestamp <= three_hours_later]
                print(f"[DEBUG] Filtered from {len(bmp_files)} to {len(filtered_bmps)} BMPs (first 3 hours)")
                bmp_files = filtered_bmps
            if bmp_files:
                folder_bmp_list.append((folder, bmp_files))
    return folder_bmp_list

def load_and_sort_images(image_tuples):
    # image_tuples: list of (img_array, timestamp, filename)
    image_tuples = [t for t in image_tuples if t[0] is not None]
    if not image_tuples:
        raise ValueError("No images could be loaded successfully")
    image_tuples.sort(key=lambda x: x[1])
    sorted_images = [item[0] for item in image_tuples]
    sorted_timestamps = [item[1] for item in image_tuples]
    sorted_filenames = [item[2] for item in image_tuples]
    return sorted_images, sorted_timestamps, sorted_filenames

def create_hdf5_file(folder_path: Path, images, timestamps, filenames):
    folder_name = folder_path.name
    output_file = folder_path / f"{folder_name}_filtered_normalized_timeseries.h5"
    print(f"[DEBUG] Attempting to save HDF5 to: {output_file.resolve()}")
    earliest_time = timestamps[0]
    relative_times = [(ts - earliest_time).total_seconds() for ts in timestamps]
    image_stack = np.stack(images, axis=0)
    try:
        with h5py.File(str(output_file), 'w') as h5f:
            h5f.create_dataset(
                'images',
                data=image_stack,
                compression='gzip',
                compression_opts=3,
                chunks=(1, image_stack.shape[1], image_stack.shape[2])
            )
            h5f.create_dataset('times', data=np.array(relative_times))
            timestamp_strings = [ts.isoformat() for ts in timestamps]
            dt = h5py.special_dtype(vlen=str)
            h5f.create_dataset('timestamps', data=timestamp_strings, dtype=dt)
            h5f.create_dataset('filenames', data=filenames, dtype=dt)
            h5f.attrs['folder_name'] = folder_name
            h5f.attrs['num_images'] = len(images)
            h5f.attrs['image_height'] = image_stack.shape[1]
            h5f.attrs['image_width'] = image_stack.shape[2]
            h5f.attrs['earliest_timestamp'] = earliest_time.isoformat()
            h5f.attrs['latest_timestamp'] = timestamps[-1].isoformat()
            h5f.attrs['total_duration_seconds'] = relative_times[-1]
            h5f.attrs['creation_time'] = datetime.now().isoformat()
            h5f.attrs['data_type'] = str(image_stack.dtype)
            h5f.attrs['description'] = f"Time series of filtered+normalized precipitation images from {folder_name} folder"
        print(f"✓ HDF5 file created: {output_file.resolve()}")
    except Exception as e:
        print(f"[ERROR] Failed to save HDF5 to: {output_file.resolve()}")
        print(f"[ERROR] Exception: {e}")
    return output_file

def process_single_bmp(args):
    bmp_path, h, template_size, search_size = args
    try:
        from PIL import Image
        import numpy as np
        from nonlocal_means_filter import apply_nonlocal_means
        import re
        from datetime import datetime
        global normalizer_global
        def parse_timestamp_from_filename(filename: str) -> datetime:
            timestamp_pattern = r'_(\d{17})_'
            match = re.search(timestamp_pattern, filename)
            if not match:
                raise ValueError(f"Could not extract timestamp from filename: {filename}")
            timestamp_str = match.group(1)
            year = int(timestamp_str[0:4])
            month = int(timestamp_str[4:6])
            day = int(timestamp_str[6:8])
            hour = int(timestamp_str[8:10])
            minute = int(timestamp_str[10:12])
            second = int(timestamp_str[12:14])
            millisecond = int(timestamp_str[14:17])
            microsecond = millisecond * 1000
            return datetime(year, month, day, hour, minute, second, microsecond)
        with Image.open(bmp_path) as img:
            img = img.convert('L')
            img_array = np.array(img)
        filtered = apply_nonlocal_means(img_array, h=h, template_window_size=template_size, search_window_size=search_size)
        normalized = normalizer_global._peak_alignment(filtered)
        timestamp = parse_timestamp_from_filename(bmp_path.name)
        return (normalized, timestamp, bmp_path.name, None)
    except Exception as e:
        return (None, None, bmp_path.name, str(e))

def process_folder(folder_path: Path, normalizer: ImageNormalizer, h=6.0, template_size=3, search_size=10, max_workers=None):
    bmp_files = find_bmp_files(folder_path)
    if not bmp_files:
        print(f"No BMP files found in {folder_path}")
        return False
    print(f"Found {len(bmp_files)} BMP(s) in {folder_path.name}")

    args_list = [(bmp_path, h, template_size, search_size) for bmp_path in bmp_files]
    image_tuples = []
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_bmp, args_list))
        for normalized, timestamp, fname, err in results:
            if normalized is not None:
                image_tuples.append((normalized, timestamp, fname))
            elif err:
                errors.append((fname, err))
    for fname, err in errors:
        print(f"✗ Error processing {fname}: {err}")
    if not image_tuples:
        print(f"No images processed successfully in {folder_path.name}")
        return False
    images, timestamps, filenames = load_and_sort_images(image_tuples)
    create_hdf5_file(folder_path, images, timestamps, filenames)
    return True

def main():
    parser = argparse.ArgumentParser(description="BMP to Filtered+Normalized HDF5 Pipeline")
    parser.add_argument('--input_root', default='image', help='Root directory containing subfolders of BMP images')
    parser.add_argument('--reference_image', default='/Users/hm114/Desktop/Precipitation_Data_test/image/2025_0628/3mM-0628-4-mix/3mM-0628-4-mix_20250628213544658_17_AfterFPN.bmp', help='Reference image for normalization (should be accessible from working dir)')
    parser.add_argument('--h', type=float, default=6.0, help='Non-local means filter strength (default: 6.0)')
    parser.add_argument('--template_size', type=int, default=3, help='Template window size (default: 3)')
    parser.add_argument('--search_size', type=int, default=10, help='Search window size (default: 10)')
    parser.add_argument('--max_workers', type=int, default=None, help='Number of parallel workers (set to number of high-performance cores)')
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        print(f"Input root directory does not exist: {input_root}")
        sys.exit(1)
    print(f"[DEBUG] Input root: {input_root.resolve()}")

    # Gather all folders and their BMPs to process
    folder_bmp_list = find_all_bmp_folders_and_bmps(input_root)
    if not folder_bmp_list:
        print("No BMP files found in any folder.")
        sys.exit(0)
    print(f"Found {sum(len(bmps) for _, bmps in folder_bmp_list)} BMP(s) in total across {len(folder_bmp_list)} folder(s).")

    for folder, bmp_files in folder_bmp_list:
        print(f"[DEBUG] Processing folder: {folder.resolve()} with {len(bmp_files)} BMPs")
        parallel_args = [(bmp_path, args.h, args.template_size, args.search_size) for bmp_path in bmp_files]
        image_tuples = []
        errors = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers, initializer=worker_initializer, initargs=(args.reference_image,)) as executor:
            with tqdm(total=len(parallel_args), desc=f"Processing BMPs in {folder.name}", unit="image") as pbar:
                for res in executor.map(process_single_bmp, parallel_args):
                    normalized, timestamp, fname, err = res
                    if normalized is not None:
                        image_tuples.append((normalized, timestamp, fname))
                    elif err:
                        errors.append((fname, err))
                    pbar.update(1)
        for fname, err in errors:
            print(f"✗ Error processing {fname}: {err}")
        if not image_tuples:
            print(f"No images processed successfully in {folder.name}")
            continue
        images, timestamps, filenames = load_and_sort_images(image_tuples)
        create_hdf5_file(folder, images, timestamps, filenames)
        # Explicitly free memory
        del image_tuples, images, timestamps, filenames

if __name__ == "__main__":
    main() 