#!/usr/bin/env python3
"""
check_and_clean_h5.py

Recursively checks all .h5 files in the image folder for corruption (cannot be opened or 'images' dataset missing/unreadable).
Deletes any ._*.h5 files found (macOS metadata). Prints a summary and a list of problematic files. Optionally deletes problematic files if --delete is specified.

Usage:
    python check_and_clean_h5.py --root image [--delete]
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
    parser = argparse.ArgumentParser(description="Check for corrupted or incomplete HDF5 files recursively. Optionally delete them. Always deletes ._*.h5 files (macOS metadata).")
    parser.add_argument('--root', default='image', help='Root folder to search for .h5 files (default: image)')
    parser.add_argument('--delete', action='store_true', help='Delete corrupted/incomplete files')
    args = parser.parse_args()

    root = Path(args.root)
    h5_files = list(root.rglob('*.h5'))
    print(f"Found {len(h5_files)} .h5 files under {root.resolve()}")

    problematic = []
    deleted = 0
    deleted_dot_underscore = 0
    good = 0
    for h5_path in h5_files:
        if h5_path.name.startswith('._'):
            try:
                h5_path.unlink()
                print(f"[DELETED ._FILE] {h5_path}")
                deleted_dot_underscore += 1
            except Exception as e:
                print(f"[ERROR] Could not delete ._ file {h5_path}: {e}")
            continue
        if is_h5_corrupted(h5_path):
            problematic.append(str(h5_path))
            if args.delete:
                try:
                    Path(h5_path).unlink()
                    print(f"[DELETED] {h5_path}")
                    deleted += 1
                except Exception as e:
                    print(f"[ERROR] Could not delete {h5_path}: {e}")
        else:
            print(f"[OK] {h5_path}")
            good += 1
    print(f"\nSummary: Checked {len(h5_files)} files, {len(problematic)} problematic, {good} good.")
    print(f"Deleted {deleted_dot_underscore} ._*.h5 files.")
    if args.delete:
        print(f"Deleted {deleted} problematic files.")
    if problematic:
        print("\nList of problematic files:")
        for p in problematic:
            print(p)
    else:
        print("\nNo problematic files found!")

if __name__ == "__main__":
    main() 