#!/usr/bin/env python3
"""
Image Quality Inspector for BMP vs TIFF Comparison

This script displays side-by-side comparisons of .bmp and .tiff files
to inspect conversion quality. It extracts 1000x1000 pixel subsets
from corresponding files for detailed comparison.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Tuple, Optional, List

# Configure PIL for large scientific images
Image.MAX_IMAGE_PIXELS = None


def find_image_pairs(root_dir: str) -> List[Tuple[Path, Path]]:
    """
    Find pairs of .bmp and .tiff files with matching base names.
    
    Args:
        root_dir: Root directory to search for image pairs
        
    Returns:
        List of tuples containing (bmp_path, tiff_path) pairs
    """
    pairs = []
    root_path = Path(root_dir)
    
    # Find all .bmp files
    for bmp_file in root_path.rglob("*.bmp"):
        # Look for corresponding .tiff file
        tiff_file = bmp_file.with_suffix('.tiff')
        if tiff_file.exists():
            pairs.append((bmp_file, tiff_file))
    
    return pairs


def get_image_info(image_path: Path) -> dict:
    """
    Get basic information about an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(image_path) as img:
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            return {
                'width': img.size[0],
                'height': img.size[1],
                'mode': img.mode,
                'file_size_mb': file_size_mb,
                'format': img.format
            }
    except Exception as e:
        return {'error': str(e)}


def extract_subset(image_path: Path, subset_size: int = 1000, 
                   x_offset: Optional[int] = None, y_offset: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Extract a subset from an image for comparison.
    
    Args:
        image_path: Path to the image file
        subset_size: Size of the square subset to extract
        x_offset: X coordinate for subset extraction (None for center)
        y_offset: Y coordinate for subset extraction (None for center)
        
    Returns:
        Tuple of (image_array, extraction_info)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Calculate offsets (center by default)
            if x_offset is None:
                x_offset = max(0, (width - subset_size) // 2)
            if y_offset is None:
                y_offset = max(0, (height - subset_size) // 2)
            
            # Ensure we don't go out of bounds
            actual_width = min(subset_size, width - x_offset)
            actual_height = min(subset_size, height - y_offset)
            
            # Extract the subset
            bbox = (x_offset, y_offset, x_offset + actual_width, y_offset + actual_height)
            subset = img.crop(bbox)
            
            # Convert to array for display
            subset_array = np.array(subset)
            
            extraction_info = {
                'original_size': (width, height),
                'subset_size': (actual_width, actual_height),
                'bbox': bbox,
                'x_offset': x_offset,
                'y_offset': y_offset
            }
            
            return subset_array, extraction_info
            
    except Exception as e:
        print(f"Error extracting subset from {image_path.name}: {e}")
        return None, {'error': str(e)}


def display_comparison(bmp_path: Path, tiff_path: Path, subset_size: int = 1000,
                      x_offset: Optional[int] = None, y_offset: Optional[int] = None,
                      save_output: bool = False):
    """
    Display side-by-side comparison of BMP and TIFF subsets.
    
    Args:
        bmp_path: Path to the .bmp file
        tiff_path: Path to the .tiff file
        subset_size: Size of the square subset to extract
        x_offset: X coordinate for subset extraction
        y_offset: Y coordinate for subset extraction
        save_output: Whether to save the comparison image
    """
    print(f"\nComparing: {bmp_path.name} vs {tiff_path.name}")
    print("-" * 60)
    
    # Get image information
    bmp_info = get_image_info(bmp_path)
    tiff_info = get_image_info(tiff_path)
    
    print(f"BMP:  {bmp_info.get('width', 'N/A')}x{bmp_info.get('height', 'N/A')} pixels, "
          f"{bmp_info.get('file_size_mb', 0):.1f} MB")
    print(f"TIFF: {tiff_info.get('width', 'N/A')}x{tiff_info.get('height', 'N/A')} pixels, "
          f"{tiff_info.get('file_size_mb', 0):.1f} MB")
    
    # Extract subsets
    print(f"Extracting {subset_size}x{subset_size} pixel subsets...")
    
    bmp_subset, bmp_extract_info = extract_subset(bmp_path, subset_size, x_offset, y_offset)
    tiff_subset, tiff_extract_info = extract_subset(tiff_path, subset_size, x_offset, y_offset)
    
    if bmp_subset is None or tiff_subset is None:
        print("Failed to extract subsets!")
        return
    
    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display BMP subset
    ax1.imshow(bmp_subset, cmap='gray' if len(bmp_subset.shape) == 2 else None)
    ax1.set_title(f'BMP Original\n{bmp_extract_info["subset_size"][0]}x{bmp_extract_info["subset_size"][1]} pixels', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Display TIFF subset
    ax2.imshow(tiff_subset, cmap='gray' if len(tiff_subset.shape) == 2 else None)
    ax2.set_title(f'TIFF Converted\n{tiff_extract_info["subset_size"][0]}x{tiff_extract_info["subset_size"][1]} pixels', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Display difference (if same size)
    if bmp_subset.shape == tiff_subset.shape:
        # Convert to same data type for comparison
        if bmp_subset.dtype != tiff_subset.dtype:
            bmp_compare = bmp_subset.astype(np.float32)
            tiff_compare = tiff_subset.astype(np.float32)
        else:
            bmp_compare = bmp_subset
            tiff_compare = tiff_subset
        
        diff = np.abs(bmp_compare - tiff_compare)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        ax3.imshow(diff, cmap='hot')
        ax3.set_title(f'Absolute Difference\nMax: {max_diff:.1f}, Mean: {mean_diff:.3f}', 
                      fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Add colorbar for difference
        cbar = plt.colorbar(ax3.images[0], ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Pixel Difference', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'Cannot compute\ndifference:\nDifferent dimensions', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.axis('off')
    
    # Add extraction location info
    extraction_text = f"Extraction region: ({bmp_extract_info['x_offset']}, {bmp_extract_info['y_offset']}) to " \
                     f"({bmp_extract_info['x_offset'] + bmp_extract_info['subset_size'][0]}, " \
                     f"{bmp_extract_info['y_offset'] + bmp_extract_info['subset_size'][1]})"
    
    plt.suptitle(f"Quality Comparison: {bmp_path.stem}\n{extraction_text}", 
                 fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save if requested
    if save_output:
        output_name = f"comparison_{bmp_path.stem}_{subset_size}x{subset_size}.png"
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {output_name}")
    
    plt.show()


def main():
    """Main function to handle command-line arguments and coordinate the inspection process."""
    parser = argparse.ArgumentParser(
        description="Compare BMP and TIFF image quality side by side",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_quality_inspector.py                           # Interactive mode to select files
  python image_quality_inspector.py --auto                    # Compare all available pairs
  python image_quality_inspector.py --file "6mM-0708-9-mix"  # Compare specific file pattern
  python image_quality_inspector.py --size 500 --x 1000 --y 1000  # Custom subset and position
        """
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=str,
        default=".",
        help="Root directory to search for image pairs (default: current directory)"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Specific file pattern to compare (partial filename match)"
    )
    
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=1000,
        help="Size of square subset to extract (default: 1000)"
    )
    
    parser.add_argument(
        "--x",
        type=int,
        help="X coordinate for subset extraction (default: center)"
    )
    
    parser.add_argument(
        "--y",
        type=int,
        help="Y coordinate for subset extraction (default: center)"
    )
    
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically compare all available pairs"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save comparison images to PNG files"
    )
    
    args = parser.parse_args()
    
    # Find image pairs
    print(f"Searching for BMP/TIFF pairs in: {Path(args.directory).absolute()}")
    pairs = find_image_pairs(args.directory)
    
    if not pairs:
        print("No BMP/TIFF pairs found!")
        return
    
    print(f"Found {len(pairs)} image pair(s)")
    
    # Filter by file pattern if specified
    if args.file:
        filtered_pairs = [(bmp, tiff) for bmp, tiff in pairs if args.file.lower() in bmp.stem.lower()]
        if not filtered_pairs:
            print(f"No pairs match pattern: {args.file}")
            return
        pairs = filtered_pairs
        print(f"Filtered to {len(pairs)} pair(s) matching '{args.file}'")
    
    # Process pairs
    if args.auto:
        # Compare all pairs automatically
        for i, (bmp_path, tiff_path) in enumerate(pairs, 1):
            print(f"\n{'='*60}")
            print(f"Comparison {i}/{len(pairs)}")
            display_comparison(bmp_path, tiff_path, args.size, args.x, args.y, args.save)
    else:
        # Interactive mode
        if len(pairs) == 1:
            bmp_path, tiff_path = pairs[0]
            display_comparison(bmp_path, tiff_path, args.size, args.x, args.y, args.save)
        else:
            print("\nAvailable pairs:")
            for i, (bmp_path, tiff_path) in enumerate(pairs, 1):
                print(f"  {i}. {bmp_path.stem}")
            
            try:
                choice = int(input(f"\nSelect pair to compare (1-{len(pairs)}): ")) - 1
                if 0 <= choice < len(pairs):
                    bmp_path, tiff_path = pairs[choice]
                    display_comparison(bmp_path, tiff_path, args.size, args.x, args.y, args.save)
                else:
                    print("Invalid selection!")
            except (ValueError, KeyboardInterrupt):
                print("Cancelled by user.")


if __name__ == "__main__":
    main() 