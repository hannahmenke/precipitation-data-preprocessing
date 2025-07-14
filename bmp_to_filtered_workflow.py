#!/usr/bin/env python3
"""
BMP to Filtered TIFF Workflow

This script combines BMP processing and non-local means filtering in a single workflow,
keeping intermediate images in memory to avoid unnecessary disk I/O and storage.
Perfect for automated pipelines where intermediate TIFF files are not needed.
"""

import os
import sys
from pathlib import Path
import argparse
import time
from typing import List, Optional

# Import our modules
from bmp_to_tiff_converter import process_bmp_in_memory, find_bmp_files
from nonlocal_means_filter import apply_nonlocal_means_filter_to_image

def process_bmp_to_filtered_tiff(bmp_path: Path, output_path: Path = None,
                                convert_to_grayscale: bool = False, 
                                channel_extract: int = None,
                                h: float = 6.0, template_size: int = 3, 
                                search_size: int = 10) -> bool:
    """
    Process a BMP file directly to filtered TIFF without saving intermediate files.
    
    Args:
        bmp_path: Path to input .bmp file
        output_path: Path to output filtered .tiff file (auto-generated if None)
        convert_to_grayscale: Convert to grayscale using standard RGB weights
        channel_extract: Extract specific channel (0=R, 1=G, 2=B, 3=A, 4=custom)
        h: Filter strength parameter
        template_size: Template window size
        search_size: Search window size
        
    Returns:
        True if successful, False otherwise
    """
    try:
        start_time = time.time()
        
        # Generate output path if not provided
        if output_path is None:
            # Create output filename: input_name_nlm_filtered.tiff
            stem = bmp_path.stem  # filename without extension
            output_path = bmp_path.parent / f"{stem}_nlm_filtered.tiff"
        
        conversion_type = ""
        if channel_extract is not None:
            channel_names = {0: "red", 1: "green", 2: "blue", 3: "alpha", 4: "alpha/green"}
            conversion_type = f" (extract {channel_names.get(channel_extract, 'unknown')} channel)"
        elif convert_to_grayscale:
            conversion_type = " (convert to grayscale)"
        
        print(f"Processing: {bmp_path.name} -> {output_path.name}{conversion_type}")
        
        # Step 1: Process BMP in memory
        print(f"  Step 1/2: Loading and processing BMP...")
        processed_image = process_bmp_in_memory(
            bmp_path=bmp_path,
            convert_to_grayscale=convert_to_grayscale,
            channel_extract=channel_extract
        )
        
        if processed_image is None:
            print(f"✗ Failed to process BMP: {bmp_path.name}")
            return False
        
        # Step 2: Apply non-local means filtering to in-memory image
        print(f"  Step 2/2: Applying non-local means filtering...")
        filtered_image = apply_nonlocal_means_filter_to_image(
            image=processed_image,
            h=h,
            template_size=template_size,
            search_size=search_size
        )
        
        if filtered_image is None:
            print(f"✗ Failed to apply filtering: {bmp_path.name}")
            return False
        
        # Step 3: Save final result
        print(f"  Saving filtered result...")
        filtered_image.save(output_path, format='TIFF', compression='lzw', 
                           tiled=True, tile=(512, 512))
        
        # Calculate processing time and file sizes
        processing_time = time.time() - start_time
        input_size_mb = bmp_path.stat().st_size / (1024 * 1024)
        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        compression_ratio = (1 - output_size_mb / input_size_mb) * 100
        
        print(f"✓ Successfully processed: {bmp_path.name}")
        print(f"  Processing time: {processing_time:.1f} seconds")
        print(f"  Input size: {input_size_mb:.1f} MB")
        print(f"  Output size: {output_size_mb:.1f} MB (compressed {compression_ratio:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {bmp_path.name}: {str(e)}")
        return False


def main():
    """Main function to handle command-line arguments and coordinate the workflow."""
    parser = argparse.ArgumentParser(
        description="Process BMP files to filtered TIFF in memory (no intermediate files)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bmp_to_filtered_workflow.py 3mM 6mM           # Process all BMPs in specified directories
  python bmp_to_filtered_workflow.py --grayscale       # Convert to grayscale before filtering
  python bmp_to_filtered_workflow.py --channel 4       # Extract channel 4 then filter
  python bmp_to_filtered_workflow.py --h 8.0           # Use stronger filtering
  python bmp_to_filtered_workflow.py --pattern "6mM"   # Process only files matching pattern
  python bmp_to_filtered_workflow.py --force           # Force reprocessing of existing files

This workflow avoids creating intermediate TIFF files, saving disk space and I/O time.
        """
    )
    
    parser.add_argument(
        "directories",
        nargs="*",
        default=["."],
        help="Directories to search for .bmp files (default: current directory)"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force processing even if filtered .tiff file already exists"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        help="Process only files matching this pattern"
    )
    
    parser.add_argument(
        "--grayscale", "-g",
        action="store_true",
        help="Convert images to grayscale before filtering"
    )
    
    parser.add_argument(
        "--channel", "-c",
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="Extract specific channel before filtering (0=Red, 1=Green, 2=Blue, 3=Alpha, 4=Alpha or Green fallback)"
    )
    
    # Filtering parameters
    parser.add_argument(
        "--h",
        type=float,
        default=6.0,
        help="Filter strength (default: 6.0, higher = more smoothing)"
    )
    
    parser.add_argument(
        "--template-size",
        type=int,
        default=3,
        help="Template window size in pixels (default: 3)"
    )
    
    parser.add_argument(
        "--search-size",
        type=int,
        default=10,
        help="Search window size in pixels (default: 10)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.h <= 0:
        print("Error: Filter strength (h) must be positive")
        sys.exit(1)
    
    if args.template_size <= 0 or args.search_size <= 0:
        print("Error: Window sizes must be positive")
        sys.exit(1)
    
    if args.channel is not None and args.grayscale:
        print("Warning: Both --channel and --grayscale specified. Channel extraction takes precedence.")
        args.grayscale = False
    
    # Find all .bmp files in specified directories
    all_bmp_files = []
    for directory in args.directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory {directory} does not exist, skipping")
            continue
        
        print(f"Searching for .bmp files in: {dir_path.absolute()}")
        bmp_files = find_bmp_files(directory)
        
        # Apply pattern filtering if specified
        if args.pattern:
            bmp_files = [f for f in bmp_files if args.pattern in f.name]
        
        all_bmp_files.extend(bmp_files)
    
    if not all_bmp_files:
        print("No .bmp files found!")
        return
    
    print(f"Found {len(all_bmp_files)} .bmp file(s)")
    print("-" * 60)
    
    # Process each .bmp file
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for bmp_path in all_bmp_files:
        # Generate output filename
        stem = bmp_path.stem  # filename without extension
        output_path = bmp_path.parent / f"{stem}_nlm_filtered.tiff"
        
        # Check if we should process this file (check for final filtered output, not intermediate)
        should_process = args.force or not output_path.exists()
        
        if not should_process:
            # Check if output is newer than input
            if output_path.exists() and bmp_path.stat().st_mtime > output_path.stat().st_mtime:
                should_process = True
                print(f"ℹ {output_path.name} exists but is older than source .bmp file")
        
        if not should_process:
            print(f"⏭ Skipping: {bmp_path.name} (filtered file exists)")
            skipped_count += 1
            continue
        
        if args.dry_run:
            conversion_type = ""
            if args.channel is not None:
                channel_names = {0: "red", 1: "green", 2: "blue", 3: "alpha", 4: "alpha/green"}
                conversion_type = f" (extract {channel_names.get(args.channel, 'unknown')} channel)"
            elif args.grayscale:
                conversion_type = " (convert to grayscale)"
            
            print(f"Would process: {bmp_path.name} -> {output_path.name}{conversion_type}")
            processed_count += 1
        else:
            if process_bmp_to_filtered_tiff(
                bmp_path=bmp_path,
                output_path=output_path,
                convert_to_grayscale=args.grayscale,
                channel_extract=args.channel,
                h=args.h,
                template_size=args.template_size,
                search_size=args.search_size
            ):
                processed_count += 1
            else:
                error_count += 1
    
    # Print summary
    print("-" * 60)
    if args.dry_run:
        print(f"Dry run complete:")
        print(f"  Would process: {processed_count} files")
        print(f"  Would skip: {skipped_count} files")
    else:
        print(f"In-memory workflow complete:")
        print(f"  Processed: {processed_count} files")
        print(f"  Skipped: {skipped_count} files")
        if error_count > 0:
            print(f"  Errors: {error_count} files")
        
        print(f"\nBenefits of in-memory workflow:")
        print(f"  - No intermediate TIFF files created")
        print(f"  - Reduced disk I/O and storage requirements")
        print(f"  - Faster processing pipeline")


if __name__ == "__main__":
    main() 