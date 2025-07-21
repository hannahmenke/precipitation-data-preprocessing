#!/usr/bin/env python3
"""
Add chemistry and replicate attributes to existing HDF5 files based on folder names.

- If the folder name contains a number (possibly with ` for decimal) followed by mM, set both Na2CO3 and CaCl to that value (as float, e.g., 2`5mM -> 2.5).
- If the folder name contains two such numbers separated by + (e.g., 2`5+100mM), set Na2CO3 to the first and CaCl to the second (as float).
- If the folder name contains a number followed by -mix, set replicate attribute to that number (as int).

Usage:
    python add_chemistry_attributes_to_hdf5.py [--root path/to/search] [file1.h5 file2.h5 ...]
    # If no files are given, updates all *_filtered_normalized_timeseries.h5 files recursively from the root directory.
"""
import h5py
from pathlib import Path
import argparse
import re

def parse_chemistry_and_replicate(folder_name):
    na2co3 = None
    cacl = None
    replicate = None
    # Find replicate: number before -mix
    rep_match = re.search(r'(\d+)-mix', folder_name)
    if rep_match:
        replicate = int(rep_match.group(1))
    # Find chemistry: look for e.g. 2`5+100mM or 6mM
    chem_match = re.search(r'([\d`\+]+)mM', folder_name)
    if chem_match:
        chem_str = chem_match.group(1)
        if '+' in chem_str:
            parts = chem_str.split('+')
            if len(parts) == 2:
                na_str, ca_str = parts
                na2co3 = float(na_str.replace('`', '.'))
                cacl = float(ca_str.replace('`', '.'))
        else:
            val = float(chem_str.replace('`', '.'))
            na2co3 = val
            cacl = val
    return na2co3, cacl, replicate

def update_hdf5_attributes(h5_path):
    folder_name = h5_path.parent.name
    na2co3, cacl, replicate = parse_chemistry_and_replicate(folder_name)
    updated = False
    with h5py.File(h5_path, 'r+') as h5f:
        if na2co3 is not None:
            h5f.attrs['Na2CO3_mM'] = na2co3
            updated = True
        if cacl is not None:
            h5f.attrs['CaCl_mM'] = cacl
            updated = True
        if replicate is not None:
            h5f.attrs['replicate'] = replicate
            updated = True
    return updated, na2co3, cacl, replicate

def main():
    parser = argparse.ArgumentParser(description="Add Na2CO3, CaCl, and replicate attributes to HDF5 files based on folder names.")
    parser.add_argument('--root', type=str, default=None, help='Root directory to search for HDF5 files (default: image/ if exists, else .)')
    parser.add_argument('files', nargs='*', help='HDF5 files to update (if not given, search recursively under --root)')
    args = parser.parse_args()

    if args.files:
        h5_files = [Path(f) for f in args.files]
    else:
        # Use --root if provided, else image/ if exists, else .
        if args.root:
            search_root = Path(args.root)
        else:
            image_dir = Path('image')
            if image_dir.exists() and image_dir.is_dir():
                search_root = image_dir
            else:
                search_root = Path('.')
        print(f"Searching for HDF5 files under: {search_root.resolve()}")
        h5_files = list(search_root.rglob('*_filtered_normalized_timeseries.h5'))

    if not h5_files:
        print("No HDF5 files found to update.")
        return

    print(f"Found {len(h5_files)} HDF5 file(s) to update.")
    for h5_path in h5_files:
        if not h5_path.exists():
            print(f"[SKIP] File not found: {h5_path}")
            continue
        try:
            updated, na2co3, cacl, replicate = update_hdf5_attributes(h5_path)
            if updated:
                print(f"[UPDATED] {h5_path}")
                print(f"  Na2CO3_mM: {na2co3}")
                print(f"  CaCl_mM: {cacl}")
                print(f"  replicate: {replicate}")
            else:
                print(f"[NO CHANGE] {h5_path} (no matching pattern in folder name)")
        except Exception as e:
            print(f"[ERROR] {h5_path}: {e}")

if __name__ == "__main__":
    main() 