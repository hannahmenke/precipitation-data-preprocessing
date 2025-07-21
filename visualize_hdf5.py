#!/usr/bin/env python3
"""
HDF5 Visualizer for Precipitation Data

This script helps you inspect and visualize HDF5 files produced by the pipeline.
- Lists all datasets and metadata
- Shows summary statistics for the image stack
- Displays a matplotlib preview of a selected image
- Optionally plots the time series

Usage:
    python visualize_hdf5.py --file path/to/file.h5 [--index 0] [--show-all-meta] [--plot-timeseries]
"""
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


def print_hdf5_structure(h5f):
    print("\nDatasets and groups:")
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  [DATASET] {name} shape={obj.shape} dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  [GROUP]   {name}")
    h5f.visititems(visitor)
    print("\nAttributes:")
    for k, v in h5f.attrs.items():
        print(f"  {k}: {v}")


def show_image_stack_info(h5f):
    if 'images' not in h5f:
        print("No 'images' dataset found!")
        return
    images = h5f['images']
    print(f"\nImage stack shape: {images.shape}")
    print(f"Data type: {images.dtype}")
    print(f"Min: {images[:].min()}  Max: {images[:].max()}")
    print(f"Num images: {images.shape[0]}")


def plot_image(h5f, index=0):
    if 'images' not in h5f:
        print("No 'images' dataset found!")
        return
    images = h5f['images']
    if index < 0 or index >= images.shape[0]:
        print(f"Index {index} out of range (0-{images.shape[0]-1})")
        return
    img = images[index]
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"Image index {index}")
    plt.axis('off')
    plt.show()


def plot_timeseries(h5f):
    if 'times' not in h5f:
        print("No 'times' dataset found!")
        return
    times = h5f['times'][:]
    plt.figure(figsize=(8, 3))
    plt.plot(times, marker='o')
    plt.title("Relative Times (seconds)")
    plt.xlabel("Frame index")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize HDF5 file contents and images")
    parser.add_argument('--file', required=True, help='Path to HDF5 file')
    parser.add_argument('--index', type=int, default=0, help='Image index to display (default: 0)')
    parser.add_argument('--show-all-meta', action='store_true', help='Print all datasets and metadata')
    parser.add_argument('--plot-timeseries', action='store_true', help='Plot the relative timeseries if available')
    args = parser.parse_args()

    with h5py.File(args.file, 'r') as h5f:
        if args.show_all_meta:
            print_hdf5_structure(h5f)
        show_image_stack_info(h5f)
        plot_image(h5f, args.index)
        if args.plot_timeseries:
            plot_timeseries(h5f)

if __name__ == "__main__":
    main() 