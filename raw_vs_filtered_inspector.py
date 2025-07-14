#!/usr/bin/env python3
"""
Raw vs Filtered Image Inspector

This script displays side-by-side comparisons of original .tiff files
and their non-local means filtered versions to inspect filtering quality.
It extracts 1000x1000 pixel subsets from corresponding files for detailed comparison.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, List

# Configure PIL for large scientific images
Image.MAX_IMAGE_PIXELS = None


def find_raw_filtered_pairs(root_dir: str, filter_suffix: str = "_nlm_filtered") -> List[Tuple[Path, Path]]:
    """
    Find pairs of raw .tiff files and their filtered versions.
    
    Args:
        root_dir: Root directory to search for image pairs
        filter_suffix: Suffix used for filtered files
        
    Returns:
        List of tuples containing (raw_path, filtered_path) pairs
    """
    pairs = []
    root_path = Path(root_dir)
    
    # Find all filtered .tiff files
    for filtered_file in root_path.rglob(f"*{filter_suffix}.tiff"):
        # Look for corresponding raw file
        raw_name = filtered_file.name.replace(filter_suffix, "")
        raw_file = filtered_file.parent / raw_name
        if raw_file.exists():
            pairs.append((raw_file, filtered_file))
    
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


def calculate_noise_metrics(raw_subset: np.ndarray, filtered_subset: np.ndarray) -> dict:
    """
    Calculate noise reduction metrics between raw and filtered images.
    
    Args:
        raw_subset: Raw image subset
        filtered_subset: Filtered image subset
        
    Returns:
        Dictionary with noise metrics
    """
    if raw_subset.shape != filtered_subset.shape:
        return {'error': 'Image dimensions do not match'}
    
    # Convert to float for calculations
    raw_float = raw_subset.astype(np.float32)
    filtered_float = filtered_subset.astype(np.float32)
    
    # Calculate metrics
    noise_removed = raw_float - filtered_float
    mean_noise_removed = np.mean(np.abs(noise_removed))
    std_noise_removed = np.std(noise_removed)
    
    # Signal-to-noise ratio approximation
    signal_strength = np.mean(filtered_float)
    noise_strength = np.std(noise_removed)
    snr_improvement = np.std(raw_float) / max(noise_strength, 1e-10)
    
    return {
        'mean_noise_removed': mean_noise_removed,
        'std_noise_removed': std_noise_removed,
        'snr_improvement': snr_improvement,
        'signal_preservation': np.corrcoef(raw_float.flatten(), filtered_float.flatten())[0,1]
    }


def display_comparison(raw_path: Path, filtered_path: Path, subset_size: int = 1000,
                      x_offset: Optional[int] = None, y_offset: Optional[int] = None,
                      save_output: bool = False):
    """
    Display side-by-side comparison of raw and filtered image subsets.
    
    Args:
        raw_path: Path to the raw .tiff file
        filtered_path: Path to the filtered .tiff file
        subset_size: Size of the square subset to extract
        x_offset: X coordinate for subset extraction
        y_offset: Y coordinate for subset extraction
        save_output: Whether to save the comparison image
    """
    print(f"\nComparing: {raw_path.name} vs {filtered_path.name}")
    print("-" * 80)
    
    # Get image information
    raw_info = get_image_info(raw_path)
    filtered_info = get_image_info(filtered_path)
    
    print(f"Raw:      {raw_info.get('width', 'N/A')}x{raw_info.get('height', 'N/A')} pixels, "
          f"{raw_info.get('file_size_mb', 0):.1f} MB")
    print(f"Filtered: {filtered_info.get('width', 'N/A')}x{filtered_info.get('height', 'N/A')} pixels, "
          f"{filtered_info.get('file_size_mb', 0):.1f} MB")
    
    # Extract subsets
    print(f"Extracting {subset_size}x{subset_size} pixel subsets...")
    
    raw_subset, raw_extract_info = extract_subset(raw_path, subset_size, x_offset, y_offset)
    filtered_subset, filtered_extract_info = extract_subset(filtered_path, subset_size, x_offset, y_offset)
    
    if raw_subset is None or filtered_subset is None:
        print("Failed to extract subsets!")
        return
    
    # Calculate noise metrics
    metrics = calculate_noise_metrics(raw_subset, filtered_subset)
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Display raw subset
    im1 = ax1.imshow(raw_subset, cmap='gray' if len(raw_subset.shape) == 2 else None)
    ax1.set_title(f'Raw Original\n{raw_extract_info["subset_size"][0]}x{raw_extract_info["subset_size"][1]} pixels', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Display filtered subset
    im2 = ax2.imshow(filtered_subset, cmap='gray' if len(filtered_subset.shape) == 2 else None)
    ax2.set_title(f'Non-Local Means Filtered\n{filtered_extract_info["subset_size"][0]}x{filtered_extract_info["subset_size"][1]} pixels', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Display difference (noise removed)
    if raw_subset.shape == filtered_subset.shape:
        # Convert to same data type for comparison
        if raw_subset.dtype != filtered_subset.dtype:
            raw_compare = raw_subset.astype(np.float32)
            filtered_compare = filtered_subset.astype(np.float32)
        else:
            raw_compare = raw_subset
            filtered_compare = filtered_subset
        
        noise_removed = raw_compare - filtered_compare
        
        im3 = ax3.imshow(noise_removed, cmap='RdBu_r', vmin=-np.max(np.abs(noise_removed)), 
                        vmax=np.max(np.abs(noise_removed)))
        ax3.set_title(f'Noise Removed (Raw - Filtered)\nMean: {metrics.get("mean_noise_removed", 0):.2f}', 
                      fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Display absolute difference
        abs_diff = np.abs(noise_removed)
        im4 = ax4.imshow(abs_diff, cmap='hot')
        ax4.set_title(f'Absolute Noise Removed\nStd: {metrics.get("std_noise_removed", 0):.2f}', 
                      fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
    else:
        ax3.text(0.5, 0.5, 'Cannot compute\ndifference:\nDifferent dimensions', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.axis('off')
        ax4.axis('off')
    
    # Add extraction location and metrics info
    extraction_text = f"Extraction region: ({raw_extract_info['x_offset']}, {raw_extract_info['y_offset']}) to " \
                     f"({raw_extract_info['x_offset'] + raw_extract_info['subset_size'][0]}, " \
                     f"{raw_extract_info['y_offset'] + raw_extract_info['subset_size'][1]})"
    
    if 'error' not in metrics:
        metrics_text = f"SNR Improvement: {metrics['snr_improvement']:.2f}x | " \
                      f"Signal Preservation: {metrics['signal_preservation']:.3f}"
        title_text = f"Filtering Quality Assessment: {raw_path.stem}\n{extraction_text}\n{metrics_text}"
    else:
        title_text = f"Filtering Quality Assessment: {raw_path.stem}\n{extraction_text}"
    
    plt.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save if requested
    if save_output:
        output_name = f"filtering_comparison_{raw_path.stem}_{subset_size}x{subset_size}.png"
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {output_name}")
    
    plt.show()
    
    # Print detailed metrics
    if 'error' not in metrics:
        print(f"Filtering Quality Metrics:")
        print(f"  Mean noise removed: {metrics['mean_noise_removed']:.2f}")
        print(f"  Noise variability: {metrics['std_noise_removed']:.2f}")
        print(f"  SNR improvement: {metrics['snr_improvement']:.2f}x")
        print(f"  Signal preservation: {metrics['signal_preservation']:.3f} (1.0 = perfect)")


def main():
    """Main function to handle command-line arguments and coordinate the inspection process."""
    parser = argparse.ArgumentParser(
        description="Compare raw and non-local means filtered image quality side by side",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python raw_vs_filtered_inspector.py                           # Interactive mode to select files
  python raw_vs_filtered_inspector.py --auto                    # Compare all available pairs
  python raw_vs_filtered_inspector.py --file "6mM-0708"        # Compare specific file pattern
  python raw_vs_filtered_inspector.py --size 500 --x 1000 --y 1000  # Custom subset and position
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
    
    parser.add_argument(
        "--filter-suffix",
        type=str,
        default="_nlm_filtered",
        help="Suffix used for filtered files (default: '_nlm_filtered')"
    )
    
    args = parser.parse_args()
    
    # Find image pairs
    print(f"Searching for raw/filtered pairs in: {Path(args.directory).absolute()}")
    pairs = find_raw_filtered_pairs(args.directory, args.filter_suffix)
    
    if not pairs:
        print(f"No raw/filtered pairs found with suffix '{args.filter_suffix}'!")
        print("Make sure you've run the non-local means filter first.")
        return
    
    print(f"Found {len(pairs)} raw/filtered pair(s)")
    
    # Filter by file pattern if specified
    if args.file:
        filtered_pairs = [(raw, filt) for raw, filt in pairs if args.file.lower() in raw.stem.lower()]
        if not filtered_pairs:
            print(f"No pairs match pattern: {args.file}")
            return
        pairs = filtered_pairs
        print(f"Filtered to {len(pairs)} pair(s) matching '{args.file}'")
    
    # Process pairs
    if args.auto:
        # Compare all pairs automatically
        for i, (raw_path, filtered_path) in enumerate(pairs, 1):
            print(f"\n{'='*80}")
            print(f"Comparison {i}/{len(pairs)}")
            display_comparison(raw_path, filtered_path, args.size, args.x, args.y, args.save)
    else:
        # Interactive mode
        if len(pairs) == 1:
            raw_path, filtered_path = pairs[0]
            display_comparison(raw_path, filtered_path, args.size, args.x, args.y, args.save)
        else:
            print("\nAvailable pairs:")
            for i, (raw_path, filtered_path) in enumerate(pairs, 1):
                print(f"  {i}. {raw_path.stem}")
            
            try:
                choice = int(input(f"\nSelect pair to compare (1-{len(pairs)}): ")) - 1
                if 0 <= choice < len(pairs):
                    raw_path, filtered_path = pairs[choice]
                    display_comparison(raw_path, filtered_path, args.size, args.x, args.y, args.save)
                else:
                    print("Invalid selection!")
            except (ValueError, KeyboardInterrupt):
                print("Cancelled by user.")


if __name__ == "__main__":
    main() 