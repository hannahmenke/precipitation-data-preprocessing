#!/usr/bin/env python3
"""
BMP to TIFF Converter for Precipitation Data

This script converts .bmp files to .tiff format while preserving the original filenames.
It scans all subdirectories and intelligently checks if files need reconversion based on
the requested format (color, grayscale, or specific channel extraction).
"""

import os
import sys
from pathlib import Path
from PIL import Image
import argparse
from typing import List, Tuple

# Configure PIL for large scientific images
# Increase the pixel limit for legitimate large scientific data
Image.MAX_IMAGE_PIXELS = None  # Remove the limit entirely for scientific data


def find_bmp_files(root_dir: str) -> List[Path]:
    """
    Find all .bmp files in the root directory and its subdirectories.
    
    Args:
        root_dir: Root directory to search for .bmp files
        
    Returns:
        List of Path objects pointing to .bmp files
    """
    bmp_files = []
    root_path = Path(root_dir)
    
    for bmp_file in root_path.rglob("*.bmp"):
        bmp_files.append(bmp_file)
    
    return bmp_files


def get_tiff_filename(bmp_path: Path) -> Path:
    """
    Generate the corresponding .tiff filename for a .bmp file.
    
    Args:
        bmp_path: Path to the .bmp file
        
    Returns:
        Path object with .tiff extension
    """
    return bmp_path.with_suffix('.tiff')


def convert_bmp_to_tiff(bmp_path: Path, tiff_path: Path = None, quality: int = 95, 
                       convert_to_grayscale: bool = False, channel_extract: int = None, 
                       return_image: bool = False) -> bool or Image.Image:
    """
    Convert a .bmp file to .tiff format.
    
    Args:
        bmp_path: Path to input .bmp file
        tiff_path: Path to output .tiff file (optional if return_image=True)
        quality: TIFF compression quality (1-100)
        convert_to_grayscale: Convert to grayscale using standard RGB weights
        channel_extract: Extract specific channel (0=R, 1=G, 2=B, 3=A, 4=custom)
        return_image: If True, return PIL Image object instead of saving to disk
        
    Returns:
        If return_image=True: PIL Image object if successful, None if failed
        If return_image=False: True if conversion successful, False otherwise
    """
    try:
        conversion_type = ""
        if channel_extract is not None:
            channel_names = {0: "red", 1: "green", 2: "blue", 3: "alpha", 4: "alpha/green"}
            conversion_type = f" (extract {channel_names.get(channel_extract, 'unknown')} channel)"
        elif convert_to_grayscale:
            conversion_type = " (convert to grayscale)"
            
        if return_image:
            print(f"Processing in memory: {bmp_path.name}{conversion_type}")
        else:
            print(f"Converting: {bmp_path.name} -> {tiff_path.name}{conversion_type}")
        
        # Get file size for progress indication
        file_size_mb = bmp_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.1f} MB")
        
        # Open the BMP image
        with Image.open(bmp_path) as img:
            print(f"  Image dimensions: {img.size[0]} x {img.size[1]} pixels")
            print(f"  Image mode: {img.mode}")
            print(f"  Processing large image... (this may take a moment)")
            
            # Handle grayscale conversion or channel extraction
            processed_img = img
            
            if channel_extract is not None:
                if img.mode in ('RGB', 'RGBA'):
                    # Convert to numpy array for channel extraction
                    import numpy as np
                    img_array = np.array(img)
                    
                    if channel_extract == 4:  # Custom channel 4 logic
                        if img.mode == 'RGBA' and img_array.shape[2] >= 4:
                            # Extract alpha channel
                            channel_data = img_array[:, :, 3]
                            print(f"  Extracting alpha channel (channel 4)")
                        else:
                            # If no alpha, use green channel as fallback
                            channel_data = img_array[:, :, 1]
                            print(f"  No alpha channel found, using green channel as fallback")
                    elif 0 <= channel_extract <= 2:
                        # Extract RGB channels
                        channel_names = ['red', 'green', 'blue']
                        channel_data = img_array[:, :, channel_extract]
                        print(f"  Extracting {channel_names[channel_extract]} channel")
                    elif channel_extract == 3 and img.mode == 'RGBA':
                        # Extract alpha channel
                        channel_data = img_array[:, :, 3]
                        print(f"  Extracting alpha channel")
                    else:
                        print(f"  Warning: Invalid channel {channel_extract}, using original image")
                        channel_data = None
                    
                    if channel_data is not None:
                        processed_img = Image.fromarray(channel_data, mode='L')
                else:
                    print(f"  Warning: Cannot extract channel from {img.mode} image, using original")
                    
            elif convert_to_grayscale:
                print(f"  Converting to grayscale...")
                if img.mode in ('RGB', 'RGBA'):
                    processed_img = img.convert('L')
                else:
                    print(f"  Image already in grayscale mode ({img.mode})")
            
            if return_image:
                # Return the processed image in memory
                # Make a copy to ensure the image data is not tied to the file handle
                image_copy = processed_img.copy()
                print(f"✓ Successfully processed in memory: {bmp_path.name}")
                return image_copy
            else:
                # Save as TIFF with LZW compression for better file size
                # Use tile-based writing for large images to reduce memory usage
                processed_img.save(tiff_path, format='TIFF', compression='lzw', 
                                  tiled=True, tile=(512, 512))
            
                # Check output file size
                output_size_mb = tiff_path.stat().st_size / (1024 * 1024)
                compression_ratio = (1 - output_size_mb / file_size_mb) * 100
                print(f"✓ Successfully converted: {bmp_path.name}")
                print(f"  Output size: {output_size_mb:.1f} MB (compressed {compression_ratio:.1f}%)")
                return True
        
    except MemoryError:
        print(f"✗ Memory error converting {bmp_path.name}: Image too large for available memory")
        return None if return_image else False
    except Exception as e:
        print(f"✗ Error converting {bmp_path.name}: {str(e)}")
        return None if return_image else False


def should_convert(bmp_path: Path, tiff_path: Path, force: bool = False, 
                  convert_to_grayscale: bool = False, channel_extract: int = None) -> bool:
    """
    Determine if a .bmp file should be converted to .tiff.
    
    Args:
        bmp_path: Path to .bmp file
        tiff_path: Path to potential .tiff file
        force: Force conversion even if .tiff exists
        convert_to_grayscale: Whether conversion should produce grayscale
        channel_extract: Whether conversion should extract specific channel
        
    Returns:
        True if file should be converted, False otherwise
    """
    if force:
        return True
    
    if not tiff_path.exists():
        return True
    
    # Check if .tiff file is newer than .bmp file
    bmp_mtime = bmp_path.stat().st_mtime
    tiff_mtime = tiff_path.stat().st_mtime
    
    if bmp_mtime > tiff_mtime:
        print(f"ℹ {tiff_path.name} exists but is older than source .bmp file")
        return True
    
    # Check if existing TIFF matches the requested conversion options
    try:
        # Temporarily increase PIL limit for large scientific images
        original_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        
        with Image.open(tiff_path) as existing_tiff:
            # Quick mode check without loading the full image
            existing_mode = existing_tiff.mode
            
            # Determine what the output should be based on flags
            expected_grayscale = convert_to_grayscale or (channel_extract is not None)
            
            if expected_grayscale:
                # We expect a grayscale output
                if existing_mode not in ('L', 'LA'):  # L = grayscale, LA = grayscale with alpha
                    print(f"ℹ {tiff_path.name} exists but is not grayscale (mode: {existing_mode}). Need to reconvert.")
                    return True
            else:
                # We expect a color output (same as original BMP)
                if existing_mode in ('L', 'LA'):  # If existing is grayscale but we want color
                    print(f"ℹ {tiff_path.name} exists but is grayscale (mode: {existing_mode}). Need color conversion.")
                    return True
                    
        # Restore original limit
        Image.MAX_IMAGE_PIXELS = original_limit
        
        # If we get here, the existing file matches the expected format
        print(f"ℹ {tiff_path.name} exists and matches requested format (mode: {existing_mode})")
        return False
        
    except Exception as e:
        # Restore original limit in case of exception
        Image.MAX_IMAGE_PIXELS = original_limit
        print(f"ℹ Cannot read existing {tiff_path.name}: {e}. Will reconvert.")
        return True


def process_bmp_in_memory(bmp_path: Path, convert_to_grayscale: bool = False, 
                         channel_extract: int = None) -> Image.Image:
    """
    Process a BMP file in memory and return the PIL Image object.
    This is a convenience function for in-memory workflows.
    
    Args:
        bmp_path: Path to input .bmp file
        convert_to_grayscale: Convert to grayscale using standard RGB weights
        channel_extract: Extract specific channel (0=R, 1=G, 2=B, 3=A, 4=custom)
        
    Returns:
        PIL Image object if successful, None if failed
    """
    return convert_bmp_to_tiff(
        bmp_path=bmp_path,
        tiff_path=None,
        convert_to_grayscale=convert_to_grayscale,
        channel_extract=channel_extract,
        return_image=True
    )


def main():
    """Main function to handle command-line arguments and coordinate the conversion process."""
    parser = argparse.ArgumentParser(
        description="Convert .bmp files to .tiff format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bmp_to_tiff_converter.py                    # Convert all .bmp files in current directory
  python bmp_to_tiff_converter.py --force            # Force conversion even if .tiff exists
  python bmp_to_tiff_converter.py --dry-run          # Show what would be converted without doing it
  python bmp_to_tiff_converter.py --grayscale        # Convert to grayscale TIFF files
  python bmp_to_tiff_converter.py --channel 4        # Extract channel 4 (alpha or green fallback)
  python bmp_to_tiff_converter.py -c 1 -g            # Extract green channel and convert to grayscale
        """
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=str,
        default=".",
        help="Root directory to search for .bmp files (default: current directory)"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force conversion even if .tiff file already exists"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting"
    )
    
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=95,
        help="TIFF compression quality (1-100, default: 95)"
    )
    
    parser.add_argument(
        "--grayscale", "-g",
        action="store_true",
        help="Convert images to grayscale using standard RGB weights"
    )
    
    parser.add_argument(
        "--channel", "-c",
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="Extract specific channel (0=Red, 1=Green, 2=Blue, 3=Alpha, 4=Alpha or Green fallback)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Process images in memory only (for pipeline use - requires external filtering step)"
    )
    
    args = parser.parse_args()
    
    # Validate parameters
    if not 1 <= args.quality <= 100:
        print("Error: Quality must be between 1 and 100")
        sys.exit(1)
    
    if args.channel is not None and args.grayscale:
        print("Warning: Both --channel and --grayscale specified. Channel extraction takes precedence.")
        args.grayscale = False
    
    if args.no_save:
        print("Warning: --no-save specified. This option is for pipeline use and won't create output files.")
        print("Use this option only when calling from other scripts that will handle the in-memory images.")
    
    # Find all .bmp files
    print(f"Searching for .bmp files in: {Path(args.directory).absolute()}")
    bmp_files = find_bmp_files(args.directory)
    
    if not bmp_files:
        print("No .bmp files found!")
        return
    
    print(f"Found {len(bmp_files)} .bmp file(s)")
    print("-" * 50)
    
    # Process each .bmp file
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    for bmp_path in bmp_files:
        tiff_path = get_tiff_filename(bmp_path)
        
        if should_convert(bmp_path, tiff_path, args.force, args.grayscale, args.channel):
            if args.dry_run:
                conversion_type = ""
                if args.channel is not None:
                    channel_names = {0: "red", 1: "green", 2: "blue", 3: "alpha", 4: "alpha/green"}
                    conversion_type = f" (extract {channel_names.get(args.channel, 'unknown')} channel)"
                elif args.grayscale:
                    conversion_type = " (convert to grayscale)"
                    
                print(f"Would convert: {bmp_path.name} -> {tiff_path.name}{conversion_type}")
                converted_count += 1
            else:
                if args.no_save:
                    # Process in memory only - mainly for demonstration/testing
                    result = convert_bmp_to_tiff(bmp_path, None, args.quality, 
                                               args.grayscale, args.channel, return_image=True)
                    if result is not None:
                        print(f"✓ Processed in memory: {bmp_path.name} ({result.mode}, {result.size})")
                        converted_count += 1
                    else:
                        error_count += 1
                else:
                    if convert_bmp_to_tiff(bmp_path, tiff_path, args.quality, 
                                         args.grayscale, args.channel):
                        converted_count += 1
                    else:
                        error_count += 1
        else:
            print(f"⏭ Skipping: {bmp_path.name} (already converted)")
            skipped_count += 1
    
    # Print summary
    print("-" * 50)
    if args.dry_run:
        print(f"Dry run complete:")
        print(f"  Would convert: {converted_count} files")
        print(f"  Would skip: {skipped_count} files")
    else:
        print(f"Conversion complete:")
        print(f"  Converted: {converted_count} files")
        print(f"  Skipped: {skipped_count} files")
        if error_count > 0:
            print(f"  Errors: {error_count} files")


if __name__ == "__main__":
    main() 