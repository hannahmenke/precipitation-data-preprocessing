#!/usr/bin/env python3
"""
check_and_clean_h5.py

Recursively checks all .h5 files in the image folder for corruption (cannot be opened or 'images' dataset missing/unreadable).
Deletes corrupted files and prints a summary.

Usage:
    python check_and_clean_h5.py --root image
"""
import h5py
from pathlib import Path
import argparse


def is_h5_corrupted(h5_path):
    try:
        with h5py.File(h5_path, 'r') as h5f:
            # Try to access the 'images' dataset
            _ = h5f['images']
        return False
    except Exception as e:
        print(f"[CORRUPT] {h5_path}: {e}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Check and clean corrupted HDF5 files recursively.")
    parser.add_argument('--root', default='image', help='Root folder to search for .h5 files (default: image)')
    args = parser.parse_args()

    root = Path(args.root)
    h5_files = list(root.rglob('*.h5'))
    print(f"Found {len(h5_files)} .h5 files under {root.resolve()}")

    deleted = 0
    good = 0
    for h5_path in h5_files:
        if is_h5_corrupted(h5_path):
            try:
                h5_path.unlink()
                print(f"[DELETED] {h5_path}")
                deleted += 1
            except Exception as e:
                print(f"[ERROR] Could not delete {h5_path}: {e}")
        else:
            print(f"[OK] {h5_path}")
            good += 1
    print(f"\nSummary: Checked {len(h5_files)} files, deleted {deleted} corrupted, {good} good.")

if __name__ == "__main__":
    main() 