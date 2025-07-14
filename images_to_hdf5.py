#!/usr/bin/env python3
"""
Images to HDF5 Converter for Precipitation Data

This script consolidates filtered TIFF images from each folder into a single HDF5 file.
Images are sorted by timestamp and stored with relative time information.
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import re
from datetime import datetime
from typing import List, Tuple, Dict
import time

# Configure PIL for large scientific images
Image.MAX_IMAGE_PIXELS = None


def parse_timestamp_from_filename(filename: str) -> datetime:
    """
    Extract timestamp from filename.
    
    Expected format: *_YYYYMMDDHHMMSSZZZ_*
    Example: 3mM-0704-12-mix_20250704183005267_1_AfterFPN_nlm_filtered.tiff
    
    Args:
        filename: Name of the file containing timestamp
        
    Returns:
        datetime object representing the timestamp
    """
    # Look for timestamp pattern: YYYYMMDDHHMMSSZZZ (17 digits)
    timestamp_pattern = r'_(\d{17})_'
    match = re.search(timestamp_pattern, filename)
    
    if not match:
        raise ValueError(f"Could not extract timestamp from filename: {filename}")
    
    timestamp_str = match.group(1)
    
    # Parse: YYYYMMDDHHMMSSZZZ
    year = int(timestamp_str[0:4])
    month = int(timestamp_str[4:6])
    day = int(timestamp_str[6:8])
    hour = int(timestamp_str[8:10])
    minute = int(timestamp_str[10:12])
    second = int(timestamp_str[12:14])
    millisecond = int(timestamp_str[14:17])
    
    # Convert milliseconds to microseconds for datetime
    microsecond = millisecond * 1000
    
    return datetime(year, month, day, hour, minute, second, microsecond)


def find_filtered_images(directory: Path) -> List[Path]:
    """
    Find all filtered TIFF images in a directory, avoiding multiply-filtered files.
    
    Args:
        directory: Directory to search for filtered images
        
    Returns:
        List of Path objects to filtered TIFF files
    """
    filtered_files = []
    
    # Look for files ending in exactly "_nlm_filtered.tiff" to avoid multiply-filtered files
    for tiff_file in directory.glob("*_nlm_filtered.tiff"):
        # Make sure it doesn't have multiple "_nlm_filtered" suffixes
        if tiff_file.name.count("_nlm_filtered") == 1:
            filtered_files.append(tiff_file)
    
    return filtered_files


def load_and_sort_images(image_paths: List[Path]) -> Tuple[List[np.ndarray], List[datetime], List[str]]:
    """
    Load images and sort them by timestamp.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        Tuple of (sorted_images, sorted_timestamps, sorted_filenames)
    """
    image_data = []
    
    print(f"Loading {len(image_paths)} images...")
    
    for img_path in image_paths:
        try:
            # Extract timestamp
            timestamp = parse_timestamp_from_filename(img_path.name)
            
            # Load image
            with Image.open(img_path) as img:
                img_array = np.array(img)
                
            image_data.append((img_array, timestamp, img_path.name))
            print(f"✓ Loaded: {img_path.name} ({timestamp})")
            
        except Exception as e:
            print(f"✗ Error loading {img_path.name}: {e}")
            continue
    
    if not image_data:
        raise ValueError("No images could be loaded successfully")
    
    # Sort by timestamp
    image_data.sort(key=lambda x: x[1])
    
    # Separate sorted data
    sorted_images = [item[0] for item in image_data]
    sorted_timestamps = [item[1] for item in image_data]
    sorted_filenames = [item[2] for item in image_data]
    
    return sorted_images, sorted_timestamps, sorted_filenames


def create_hdf5_file(folder_path: Path, images: List[np.ndarray], 
                    timestamps: List[datetime], filenames: List[str]) -> Path:
    """
    Create HDF5 file with image stack and metadata.
    
    Args:
        folder_path: Path to the folder containing the images
        images: List of image arrays
        timestamps: List of timestamps
        filenames: List of original filenames
        
    Returns:
        Path to created HDF5 file
    """
    folder_name = folder_path.name
    output_file = folder_path / f"{folder_name}_filtered_timeseries.h5"
    
    print(f"\nCreating HDF5 file: {output_file}")
    
    # Calculate relative times (seconds from earliest timestamp)
    earliest_time = timestamps[0]
    relative_times = []
    
    for ts in timestamps:
        time_diff = ts - earliest_time
        # Convert to seconds with fractional part for milliseconds
        relative_seconds = time_diff.total_seconds()
        relative_times.append(relative_seconds)
    
    # Stack images into 3D array (time, height, width)
    image_stack = np.stack(images, axis=0)
    
    print(f"Image stack shape: {image_stack.shape}")
    print(f"Data type: {image_stack.dtype}")
    print(f"Time range: 0 to {relative_times[-1]:.3f} seconds")
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as h5f:
        # Store image stack
        h5f.create_dataset('images', data=image_stack, 
                          compression='gzip', compression_opts=6,
                          chunks=True)
        
        # Store relative times
        h5f.create_dataset('times', data=np.array(relative_times))
        
        # Store absolute timestamps as strings for reference
        timestamp_strings = [ts.isoformat() for ts in timestamps]
        dt = h5py.special_dtype(vlen=str)
        h5f.create_dataset('timestamps', data=timestamp_strings, dtype=dt)
        
        # Store original filenames
        h5f.create_dataset('filenames', data=filenames, dtype=dt)
        
        # Add metadata as attributes
        h5f.attrs['folder_name'] = folder_name
        h5f.attrs['num_images'] = len(images)
        h5f.attrs['image_height'] = image_stack.shape[1]
        h5f.attrs['image_width'] = image_stack.shape[2]
        h5f.attrs['earliest_timestamp'] = earliest_time.isoformat()
        h5f.attrs['latest_timestamp'] = timestamps[-1].isoformat()
        h5f.attrs['total_duration_seconds'] = relative_times[-1]
        h5f.attrs['creation_time'] = datetime.now().isoformat()
        h5f.attrs['data_type'] = str(image_stack.dtype)
        
        # Add description
        h5f.attrs['description'] = f"Time series of filtered precipitation images from {folder_name} folder"
    
    print(f"✓ HDF5 file created successfully: {output_file}")
    return output_file


def process_folder(folder_path: Path, force: bool = False) -> bool:
    """
    Process a single folder to create HDF5 file.
    
    Args:
        folder_path: Path to folder containing filtered images
        force: Force overwrite existing HDF5 file
        
    Returns:
        True if successful, False otherwise
    """
    folder_name = folder_path.name
    output_file = folder_path / f"{folder_name}_filtered_timeseries.h5"
    
    # Check if output already exists
    if output_file.exists() and not force:
        print(f"⏭ Skipping {folder_name}: HDF5 file already exists (use --force to overwrite)")
        return True
    
    print(f"\n{'='*60}")
    print(f"Processing folder: {folder_name}")
    print(f"{'='*60}")
    
    # Find filtered images
    image_paths = find_filtered_images(folder_path)
    
    if not image_paths:
        print(f"No filtered images found in {folder_name}")
        return False
    
    print(f"Found {len(image_paths)} filtered image(s)")
    
    try:
        # Load and sort images
        images, timestamps, filenames = load_and_sort_images(image_paths)
        
        # Create HDF5 file
        create_hdf5_file(folder_path, images, timestamps, filenames)
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {folder_name}: {e}")
        return False


def main():
    """Main function to handle command-line arguments and coordinate processing."""
    parser = argparse.ArgumentParser(
        description="Convert filtered TIFF images to HDF5 time series files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python images_to_hdf5.py                    # Process all folders with filtered images
  python images_to_hdf5.py 3mM 6mM            # Process specific folders
  python images_to_hdf5.py --force            # Force overwrite existing HDF5 files
  python images_to_hdf5.py --dry-run          # Show what would be processed

Output HDF5 Structure:
  /images                    # 3D array (time, height, width)
  /times                     # 1D array of relative times (seconds from earliest)
  /timestamps                # 1D array of absolute timestamp strings
  /filenames                 # 1D array of original filenames
  
  Attributes:
    folder_name              # Name of source folder
    num_images               # Number of images in time series
    image_height, image_width # Image dimensions
    earliest_timestamp       # ISO format timestamp of first image
    latest_timestamp         # ISO format timestamp of last image
    total_duration_seconds   # Total time span of the series
        """
    )
    
    parser.add_argument(
        "folders",
        nargs="*",
        help="Folders to process (default: all folders containing filtered images)"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force overwrite existing HDF5 files"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without creating files"
    )
    
    args = parser.parse_args()
    
    # Determine folders to process
    if args.folders:
        # Process specified folders
        folder_paths = []
        for folder_name in args.folders:
            folder_path = Path(folder_name)
            if not folder_path.exists():
                print(f"Warning: Folder {folder_name} does not exist, skipping")
                continue
            if not folder_path.is_dir():
                print(f"Warning: {folder_name} is not a directory, skipping")
                continue
            folder_paths.append(folder_path)
    else:
        # Find all folders containing filtered images
        current_dir = Path(".")
        folder_paths = []
        
        for item in current_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
                filtered_images = find_filtered_images(item)
                if filtered_images:
                    folder_paths.append(item)
    
    if not folder_paths:
        print("No folders with filtered images found!")
        return
    
    print(f"Found {len(folder_paths)} folder(s) to process:")
    for folder_path in folder_paths:
        filtered_count = len(find_filtered_images(folder_path))
        print(f"  - {folder_path.name}: {filtered_count} filtered image(s)")
    
    if args.dry_run:
        print("\nDry run complete - no files would be created")
        return
    
    # Process each folder
    success_count = 0
    error_count = 0
    
    start_time = time.time()
    
    for folder_path in folder_paths:
        if process_folder(folder_path, args.force):
            success_count += 1
        else:
            error_count += 1
    
    # Print summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Processing complete:")
    print(f"  - Successfully processed: {success_count} folder(s)")
    print(f"  - Errors: {error_count} folder(s)")
    print(f"  - Total time: {duration:.1f} seconds")
    
    if success_count > 0:
        print(f"\nHDF5 files created:")
        for folder_path in folder_paths:
            hdf5_file = folder_path / f"{folder_path.name}_filtered_timeseries.h5"
            if hdf5_file.exists():
                file_size = hdf5_file.stat().st_size / (1024**2)  # MB
                print(f"  - {hdf5_file}: {file_size:.1f} MB")


if __name__ == "__main__":
    main() 