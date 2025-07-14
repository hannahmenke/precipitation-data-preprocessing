#!/usr/bin/env python3
"""
HDF5 Inspector for Precipitation Data

This script inspects the structure and contents of HDF5 files created by images_to_hdf5.py
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
import sys


def inspect_hdf5_file(filepath: Path):
    """
    Inspect and display information about an HDF5 file.
    
    Args:
        filepath: Path to the HDF5 file
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*80}")
    
    try:
        with h5py.File(filepath, 'r') as h5f:
            # Display file-level attributes
            print("\n📋 File Attributes:")
            for key, value in h5f.attrs.items():
                print(f"  {key}: {value}")
            
            # Display datasets
            print(f"\n📊 Datasets:")
            for key in h5f.keys():
                dataset = h5f[key]
                print(f"  /{key}:")
                print(f"    Shape: {dataset.shape}")
                print(f"    Data type: {dataset.dtype}")
                print(f"    Size: {dataset.size:,} elements")
                
                if key == 'images':
                    print(f"    Memory size: {dataset.nbytes / (1024**2):.1f} MB")
                    print(f"    Pixel range: {dataset[:].min()} - {dataset[:].max()}")
                elif key == 'times':
                    times = dataset[:]
                    print(f"    Time range: {times[0]:.3f} - {times[-1]:.3f} seconds")
                    print(f"    Duration: {times[-1]:.1f} seconds ({times[-1]/3600:.2f} hours)")
                    if len(times) > 1:
                        intervals = np.diff(times)
                        print(f"    Time intervals: {intervals}")
                elif key == 'timestamps':
                    timestamps = [ts.decode() if isinstance(ts, bytes) else ts for ts in dataset[:]]
                    print(f"    First timestamp: {timestamps[0]}")
                    print(f"    Last timestamp: {timestamps[-1]}")
                elif key == 'filenames':
                    filenames = [fn.decode() if isinstance(fn, bytes) else fn for fn in dataset[:]]
                    print(f"    Files included:")
                    for i, filename in enumerate(filenames):
                        print(f"      {i}: {filename}")
            
            # Calculate compression ratio if original size info is available
            print(f"\n💾 File Information:")
            file_size = filepath.stat().st_size
            print(f"  HDF5 file size: {file_size / (1024**2):.1f} MB")
            
            if 'images' in h5f:
                uncompressed_size = h5f['images'].nbytes
                compression_ratio = uncompressed_size / file_size
                print(f"  Uncompressed size: {uncompressed_size / (1024**2):.1f} MB")
                print(f"  Compression ratio: {compression_ratio:.1f}x")
            
    except Exception as e:
        print(f"❌ Error inspecting {filepath}: {e}")


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 time series files created by images_to_hdf5.py"
    )
    
    parser.add_argument(
        "files",
        nargs="*",
        help="HDF5 files to inspect (default: find all *_filtered_timeseries.h5 files)"
    )
    
    args = parser.parse_args()
    
    # Determine files to inspect
    if args.files:
        file_paths = [Path(f) for f in args.files]
    else:
        # Find all HDF5 time series files
        file_paths = list(Path(".").rglob("*_filtered_timeseries.h5"))
    
    if not file_paths:
        print("No HDF5 files found to inspect!")
        return
    
    print(f"Found {len(file_paths)} HDF5 file(s) to inspect")
    
    for filepath in file_paths:
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            continue
        inspect_hdf5_file(filepath)
    
    print(f"\n{'='*80}")
    print("Inspection complete!")


if __name__ == "__main__":
    main() 