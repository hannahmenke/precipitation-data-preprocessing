#!/usr/bin/env python3
"""
Non-Local Means Filter for Precipitation Data

This script applies non-local means filtering to .tiff files for denoising
while preserving image details and edges. It's particularly effective for
scientific imaging data.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from typing import List, Optional, Tuple
import time

# Configure PIL for large scientific images
Image.MAX_IMAGE_PIXELS = None


def find_tiff_files(root_dir: str, pattern: Optional[str] = None) -> List[Path]:
    """
    Find all .tiff files in the root directory and its subdirectories.
    
    Args:
        root_dir: Root directory to search for .tiff files
        pattern: Optional pattern to filter filenames
        
    Returns:
        List of Path objects pointing to .tiff files
    """
    tiff_files = []
    root_path = Path(root_dir)
    
    for tiff_file in root_path.rglob("*.tiff"):
        if pattern is None or pattern.lower() in tiff_file.name.lower():
            tiff_files.append(tiff_file)
    
    return tiff_files


def get_output_filename(input_path: Path, suffix: str = "_nlm_filtered") -> Path:
    """
    Generate output filename for filtered image.
    
    Args:
        input_path: Path to input .tiff file
        suffix: Suffix to add to the filename
        
    Returns:
        Path object for output file
    """
    stem = input_path.stem
    return input_path.parent / f"{stem}{suffix}.tiff"


def apply_nonlocal_means(image: np.ndarray, h: float = 6, template_window_size: int = 3, 
                        search_window_size: int = 10) -> np.ndarray:
    """
    Apply non-local means filtering to an image.
    
    Default parameters match Avizo settings:
    - h=6 corresponds to Avizo similarity value 0.8
    - template_window_size=3 matches Avizo local neighborhood 3 pixels
    - search_window_size=10 matches Avizo search window 10 px
    
    Args:
        image: Input image as numpy array
        h: Filter strength. Higher h value removes more noise but removes image details too
        template_window_size: Size of template patch (should be odd)
        search_window_size: Size of search window (should be even for OpenCV)
        
    Returns:
        Filtered image as numpy array
    """
    # Convert to uint8 if needed (OpenCV requirement)
    if image.dtype != np.uint8:
        # Normalize to 0-255 range
        img_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    else:
        img_normalized = image
    
    # Apply non-local means filtering
    if len(img_normalized.shape) == 2:
        # Grayscale image
        filtered = cv2.fastNlMeansDenoising(img_normalized, None, h, template_window_size, search_window_size)
    else:
        # Color image
        filtered = cv2.fastNlMeansDenoisingColored(img_normalized, None, h, h, template_window_size, search_window_size)
    
    # Convert back to original data type if needed
    if image.dtype != np.uint8:
        # Scale back to original range
        filtered = filtered.astype(np.float32) / 255.0
        filtered = filtered * (image.max() - image.min()) + image.min()
        filtered = filtered.astype(image.dtype)
    
    return filtered


def process_large_image_in_tiles(image: np.ndarray, tile_size: int = 2048, overlap: int = 128,
                                h: float = 6, template_window_size: int = 3, 
                                search_window_size: int = 10) -> np.ndarray:
    """
    Process large images in tiles to manage memory usage.
    
    Args:
        image: Input image as numpy array
        tile_size: Size of each tile
        overlap: Overlap between tiles to avoid edge artifacts
        h: Filter strength parameter
        template_window_size: Template patch size
        search_window_size: Search window size
        
    Returns:
        Filtered image as numpy array
    """
    height, width = image.shape[:2]
    
    # If image is small enough, process directly
    if height <= tile_size and width <= tile_size:
        return apply_nonlocal_means(image, h, template_window_size, search_window_size)
    
    print(f"  Processing large image in tiles ({tile_size}x{tile_size} with {overlap}px overlap)")
    
    # Initialize output image
    filtered_image = np.zeros_like(image)
    
    # Process tiles
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            # Calculate tile boundaries
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            
            # Extract tile
            tile = image[y:y_end, x:x_end]
            
            # Apply filtering to tile
            filtered_tile = apply_nonlocal_means(tile, h, template_window_size, search_window_size)
            
            # Calculate region to copy (excluding overlap for interior tiles)
            copy_y_start = overlap // 2 if y > 0 else 0
            copy_x_start = overlap // 2 if x > 0 else 0
            copy_y_end = filtered_tile.shape[0] - overlap // 2 if y_end < height else filtered_tile.shape[0]
            copy_x_end = filtered_tile.shape[1] - overlap // 2 if x_end < width else filtered_tile.shape[1]
            
            # Copy filtered tile to output
            filtered_image[y + copy_y_start:y + copy_y_end, 
                          x + copy_x_start:x + copy_x_end] = filtered_tile[copy_y_start:copy_y_end, 
                                                                          copy_x_start:copy_x_end]
    
    return filtered_image


def apply_nonlocal_means_filter_to_image(image: Image.Image, h: float = 6, 
                                        template_size: int = 3, search_size: int = 10,
                                        tile_size: int = 2048) -> Optional[Image.Image]:
    """
    Apply non-local means filtering to a PIL Image object and return the filtered image.
    
    Args:
        image: PIL Image object to filter
        h: Filter strength parameter (higher = more smoothing)
        template_size: Template window size in pixels
        search_size: Search window size in pixels
        tile_size: Tile size for large image processing
        
    Returns:
        PIL Image object with filtering applied, or None if failed
    """
    try:
        print(f"    Applying non-local means filter (h={h}, template={template_size}, search={search_size})")
        print(f"    Image mode: {image.mode}, size: {image.size}")
        
        # Convert PIL Image to numpy array
        if image.mode == 'RGBA':
            # Handle alpha channel separately for RGBA images
            img_array = np.array(image)
            rgb_array = img_array[:, :, :3]
            alpha_array = img_array[:, :, 3]
            
            # Apply filtering to RGB channels only
            if rgb_array.shape[0] * rgb_array.shape[1] > tile_size * tile_size:
                print(f"    Using tile-based processing for large image...")
                filtered_rgb = process_large_image_in_tiles(
                    rgb_array, tile_size, overlap=128, h=h, 
                    template_window_size=template_size, search_window_size=search_size
                )
            else:
                filtered_rgb = apply_nonlocal_means(
                    rgb_array, h=h, template_window_size=template_size, 
                    search_window_size=search_size
                )
            
            # Combine filtered RGB with original alpha
            filtered_array = np.dstack([filtered_rgb, alpha_array])
            filtered_image = Image.fromarray(filtered_array, mode='RGBA')
            
        else:
            # Handle other image modes
            img_array = np.array(image)
            
            if img_array.shape[0] * img_array.shape[1] > tile_size * tile_size:
                print(f"    Using tile-based processing for large image...")
                filtered_array = process_large_image_in_tiles(
                    img_array, tile_size, overlap=128, h=h, 
                    template_window_size=template_size, search_window_size=search_size
                )
            else:
                filtered_array = apply_nonlocal_means(
                    img_array, h=h, template_window_size=template_size, 
                    search_window_size=search_size
                )
            
            # Convert back to PIL Image with original mode
            filtered_image = Image.fromarray(filtered_array, mode=image.mode)
        
        print(f"    ✓ Filtering completed successfully")
        return filtered_image
        
    except Exception as e:
        print(f"    ✗ Error applying filter: {str(e)}")
        return None


def process_tiff_file(input_path: Path, output_path: Path, h: float = 6, 
                     template_window_size: int = 3, search_window_size: int = 10,
                     tile_size: int = 2048, use_tiling: bool = True) -> bool:
    """
    Process a single .tiff file with non-local means filtering.
    
    Args:
        input_path: Path to input .tiff file
        output_path: Path to output .tiff file
        h: Filter strength parameter
        template_window_size: Template patch size
        search_window_size: Search window size
        tile_size: Size of tiles for large image processing
        use_tiling: Whether to use tiling for large images
        
    Returns:
        True if processing successful, False otherwise
    """
    try:
        start_time = time.time()
        
        print(f"Processing: {input_path.name}")
        
        # Load image
        with Image.open(input_path) as img:
            image_array = np.array(img)
            print(f"  Image size: {image_array.shape}")
            print(f"  Data type: {image_array.dtype}")
        
        # Apply non-local means filtering
        print(f"  Applying non-local means filter (h={h}, template={template_window_size}, search={search_window_size})")
        
        if use_tiling and (image_array.shape[0] > tile_size or image_array.shape[1] > tile_size):
            filtered_image = process_large_image_in_tiles(
                image_array, tile_size, overlap=128, h=h, 
                template_window_size=template_window_size, 
                search_window_size=search_window_size
            )
        else:
            filtered_image = apply_nonlocal_means(
                image_array, h, template_window_size, search_window_size
            )
        
        # Save filtered image
        print(f"  Saving filtered image to: {output_path.name}")
        filtered_pil = Image.fromarray(filtered_image)
        filtered_pil.save(output_path, format='TIFF', compression='lzw')
        
        # Calculate file sizes
        input_size_mb = input_path.stat().st_size / (1024 * 1024)
        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        processing_time = time.time() - start_time
        print(f"✓ Successfully processed: {input_path.name}")
        print(f"  Input size: {input_size_mb:.1f} MB")
        print(f"  Output size: {output_size_mb:.1f} MB")
        print(f"  Processing time: {processing_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {str(e)}")
        return False


def main():
    """Main function to handle command-line arguments and coordinate the filtering process."""
    parser = argparse.ArgumentParser(
        description="Apply non-local means filtering to .tiff files for denoising",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nonlocal_means_filter.py                              # Process all .tiff files
  python nonlocal_means_filter.py --h 15                       # Stronger denoising
  python nonlocal_means_filter.py --pattern "6mM"              # Process only 6mM files
  python nonlocal_means_filter.py --template-size 5 --search-size 8   # Faster processing
  python nonlocal_means_filter.py --no-tiling                  # Process without tiling
        
Filter strength guide (Avizo similarity equivalent):
  h=3-5:   Very light denoising (similarity ~0.9)
  h=6-8:   Light denoising (similarity ~0.8, default)
  h=10-15: Moderate denoising (similarity ~0.6-0.7)
  h=20+:   Strong denoising (similarity ~0.5), may blur details
        """
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=str,
        default=".",
        help="Root directory to search for .tiff files (default: current directory)"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        help="Pattern to filter filenames (e.g., '6mM', 'AfterFPN')"
    )
    
    parser.add_argument(
        "--h",
        type=float,
        default=6.0,
        help="Filter strength (default: 6.0, equivalent to Avizo similarity 0.8). Higher values remove more noise but may blur details"
    )
    
    parser.add_argument(
        "--template-size", "-t",
        type=int,
        default=3,
        help="Local neighborhood size (default: 3, matches Avizo setting, must be odd)"
    )
    
    parser.add_argument(
        "--search-size", "-s",
        type=int,
        default=10,
        help="Search window size (default: 10, matches Avizo setting, must be even for OpenCV)"
    )
    
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Tile size for large image processing (default: 2048)"
    )
    
    parser.add_argument(
        "--no-tiling",
        action="store_true",
        help="Disable tiling for large images (may use more memory)"
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        default="_nlm_filtered",
        help="Suffix to add to output filenames (default: '_nlm_filtered')"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force processing even if output file exists"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.template_size % 2 == 0:
        print("Error: Template size (local neighborhood) must be odd")
        sys.exit(1)
    
    if args.search_size % 2 != 0:
        print("Error: Search window size must be even for OpenCV")
        sys.exit(1)
    
    if args.h <= 0:
        print("Error: Filter strength (h) must be positive")
        sys.exit(1)
    
    # Find .tiff files
    print(f"Searching for .tiff files in: {Path(args.directory).absolute()}")
    if args.pattern:
        print(f"Filtering by pattern: {args.pattern}")
    
    tiff_files = find_tiff_files(args.directory, args.pattern)
    
    if not tiff_files:
        print("No .tiff files found!")
        return
    
    print(f"Found {len(tiff_files)} .tiff file(s)")
    print("-" * 60)
    
    # Process files
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for tiff_file in tiff_files:
        output_file = get_output_filename(tiff_file, args.suffix)
        
        # Check if output already exists
        if output_file.exists() and not args.force:
            print(f"⏭ Skipping: {tiff_file.name} (output already exists)")
            skipped_count += 1
            continue
        
        if args.dry_run:
            print(f"Would process: {tiff_file.name} -> {output_file.name}")
            processed_count += 1
        else:
            success = process_tiff_file(
                tiff_file, output_file, args.h, args.template_size, args.search_size,
                args.tile_size, not args.no_tiling
            )
            
            if success:
                processed_count += 1
            else:
                error_count += 1
        
        print()
    
    # Print summary
    print("-" * 60)
    if args.dry_run:
        print(f"Dry run complete:")
        print(f"  Would process: {processed_count} files")
        print(f"  Would skip: {skipped_count} files")
    else:
        print(f"Processing complete:")
        print(f"  Processed: {processed_count} files")
        print(f"  Skipped: {skipped_count} files")
        if error_count > 0:
            print(f"  Errors: {error_count} files")


if __name__ == "__main__":
    main() 