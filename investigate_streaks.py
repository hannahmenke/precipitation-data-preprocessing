#!/usr/bin/env python3
"""
Streak Investigation Tool

This script investigates the cause of streak artifacts appearing in normalized precipitation data.
It examines:
1. Raw BMP images for pre-existing streaks
2. Reference image quality and characteristics
3. Non-local means filtering effects
4. Peak alignment normalization effects
5. Step-by-step processing pipeline

Usage:
    python investigate_streaks.py --h5_file path/to/problematic.h5
    python investigate_streaks.py --bmp_folder path/to/folder --reference_image path/to/ref.bmp
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
from datetime import datetime
from scipy import ndimage

# Import our pipeline components
from image_normalization import ImageNormalizer
from nonlocal_means_filter import apply_nonlocal_means


def detect_regional_intensity_changes(image, region_size=1000, sensitivity='medium'):
    """
    Detect regional intensity changes that appear as vertical or horizontal streaks.
    
    Args:
        image: Input image as numpy array
        region_size: Size of regions to analyze for intensity changes
        sensitivity: 'low', 'medium', or 'high' detection sensitivity
        
    Returns:
        Dictionary with streak detection results
    """
    # Convert to float for processing
    img_float = image.astype(np.float32)
    h, w = img_float.shape
    
    # Divide image into regions and analyze intensity differences
    regions_h = h // region_size + (1 if h % region_size > 0 else 0)
    regions_w = w // region_size + (1 if w % region_size > 0 else 0)
    
    region_means = np.zeros((regions_h, regions_w))
    region_stds = np.zeros((regions_h, regions_w))
    
    # Calculate mean and std for each region
    for i in range(regions_h):
        for j in range(regions_w):
            start_h = i * region_size
            end_h = min((i + 1) * region_size, h)
            start_w = j * region_size
            end_w = min((j + 1) * region_size, w)
            
            region = img_float[start_h:end_h, start_w:end_w]
            region_means[i, j] = np.mean(region)
            region_stds[i, j] = np.std(region)
    
    # Set thresholds based on sensitivity
    if sensitivity == 'low':
        mean_threshold = 3.0
        std_threshold = 3.0
    elif sensitivity == 'medium':
        mean_threshold = 2.0
        std_threshold = 2.0
    else:  # high
        mean_threshold = 1.5
        std_threshold = 1.5
    
    # Detect regions with significantly different intensity
    global_mean = np.mean(region_means)
    global_std = np.std(region_means)
    
    # Find anomalous regions
    mean_outliers = np.abs(region_means - global_mean) > mean_threshold * global_std
    std_outliers = region_stds > np.mean(region_stds) + std_threshold * np.std(region_stds)
    
    anomalous_regions = mean_outliers | std_outliers
    
    # Check for vertical streaks (columns of anomalous regions)
    vertical_streak_cols = []
    for j in range(regions_w):
        if np.sum(anomalous_regions[:, j]) >= regions_h * 0.3:  # At least 30% of column is anomalous
            vertical_streak_cols.append(j)
    
    # Check for horizontal streaks (rows of anomalous regions)
    horizontal_streak_rows = []
    for i in range(regions_h):
        if np.sum(anomalous_regions[i, :]) >= regions_w * 0.3:  # At least 30% of row is anomalous
            horizontal_streak_rows.append(i)
    
    # Convert back to pixel coordinates
    vertical_streak_pixels = []
    for col in vertical_streak_cols:
        start_pixel = col * region_size
        end_pixel = min((col + 1) * region_size, w)
        vertical_streak_pixels.extend(range(start_pixel, end_pixel))
    
    horizontal_streak_pixels = []
    for row in horizontal_streak_rows:
        start_pixel = row * region_size
        end_pixel = min((row + 1) * region_size, h)
        horizontal_streak_pixels.extend(range(start_pixel, end_pixel))
    
    # Also analyze overall column and row statistics for comparison
    col_means = np.mean(img_float, axis=0)
    row_means = np.mean(img_float, axis=1)
    
    # Detect major intensity shifts (like left 1/4 being different)
    # Divide image into quarters and compare
    quarter_w = w // 4
    quarter_h = h // 4
    
    quarters = {
        'top_left': img_float[:quarter_h, :quarter_w],
        'top_right': img_float[:quarter_h, 3*quarter_w:],
        'bottom_left': img_float[3*quarter_h:, :quarter_w],
        'bottom_right': img_float[3*quarter_h:, 3*quarter_w:],
        'left_half': img_float[:, :w//2],
        'right_half': img_float[:, w//2:],
        'top_half': img_float[:h//2, :],
        'bottom_half': img_float[h//2:, :]
    }
    
    quarter_stats = {}
    for name, quarter in quarters.items():
        quarter_stats[name] = {
            'mean': np.mean(quarter),
            'std': np.std(quarter),
            'median': np.median(quarter)
        }
    
    # Check for significant differences between regions
    region_differences = {
        'left_vs_right': abs(quarter_stats['left_half']['mean'] - quarter_stats['right_half']['mean']),
        'top_vs_bottom': abs(quarter_stats['top_half']['mean'] - quarter_stats['bottom_half']['mean']),
        'corners_diff': np.std([quarter_stats['top_left']['mean'], quarter_stats['top_right']['mean'], 
                               quarter_stats['bottom_left']['mean'], quarter_stats['bottom_right']['mean']])
    }
    
    return {
        'vertical_streaks': np.array(vertical_streak_pixels),
        'horizontal_streaks': np.array(horizontal_streak_pixels),
        'region_means': region_means,
        'region_stds': region_stds,
        'anomalous_regions': anomalous_regions,
        'col_means': col_means,
        'row_means': row_means,
        'quarter_stats': quarter_stats,
        'region_differences': region_differences,
        'region_size': region_size,
        'regions_shape': (regions_h, regions_w),
        'method': f'regional_{sensitivity}'
    }


def analyze_reference_image(reference_path: Path, save_dir: Path):
    """Analyze the reference image for pre-existing artifacts."""
    print(f"\n{'='*60}")
    print("REFERENCE IMAGE ANALYSIS")
    print(f"{'='*60}")
    
    with Image.open(reference_path) as img:
        img = img.convert('L')
        ref_array = np.array(img)
    
    print(f"Reference image: {reference_path.name}")
    print(f"Shape: {ref_array.shape}")
    print(f"Data type: {ref_array.dtype}")
    print(f"Value range: [{ref_array.min()}, {ref_array.max()}]")
    print(f"Mean: {ref_array.mean():.2f}, Std: {ref_array.std():.2f}")
    
    # Use improved regional streak detection
    streak_analysis = detect_regional_intensity_changes(ref_array, region_size=1000, sensitivity='medium')
    
    print(f"\nRegional intensity analysis:")
    print(f"Vertical streak pixels: {len(streak_analysis['vertical_streaks'])} ({len(streak_analysis['vertical_streaks'])/ref_array.shape[1]*100:.2f}%)")
    print(f"Horizontal streak pixels: {len(streak_analysis['horizontal_streaks'])} ({len(streak_analysis['horizontal_streaks'])/ref_array.shape[0]*100:.2f}%)")
    print(f"Anomalous regions: {np.sum(streak_analysis['anomalous_regions'])} / {np.prod(streak_analysis['anomalous_regions'].shape)}")
    
    print(f"\nQuarter comparisons:")
    for name, value in streak_analysis['region_differences'].items():
        print(f"  {name}: {value:.2f}")
    
    print(f"\nQuarter statistics:")
    for name, stats in streak_analysis['quarter_stats'].items():
        print(f"  {name}: mean={stats['mean']:.1f}, std={stats['std']:.1f}")
    
    # Create visualization (save instead of show)
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Reference image
    axes[0, 0].imshow(ref_array, cmap='gray')
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].axis('off')
    
    # Regional analysis heatmap
    im1 = axes[0, 1].imshow(streak_analysis['region_means'], cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f'Regional Means\n({streak_analysis["regions_shape"][0]}x{streak_analysis["regions_shape"][1]} regions)')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.6)
    
    # Anomalous regions
    axes[0, 2].imshow(streak_analysis['anomalous_regions'], cmap='Reds', aspect='auto')
    axes[0, 2].set_title(f'Anomalous Regions\n{np.sum(streak_analysis["anomalous_regions"])} detected')
    
    # Column means
    axes[1, 0].plot(streak_analysis['col_means'], alpha=0.7)
    axes[1, 0].set_title('Column Means')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Mean Intensity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Row means
    axes[1, 1].plot(streak_analysis['row_means'], alpha=0.7)
    axes[1, 1].set_title('Row Means')
    axes[1, 1].set_xlabel('Row')
    axes[1, 1].set_ylabel('Mean Intensity')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Histogram
    axes[1, 2].hist(ref_array.ravel(), bins=50, alpha=0.7, density=True)
    peak_idx = np.argmax(np.histogram(ref_array.ravel(), bins=256)[0])
    axes[1, 2].set_title(f'Reference Histogram\nPeak: {peak_idx}')
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Quarter means comparison
    quarter_names = list(streak_analysis['quarter_stats'].keys())
    quarter_means = [streak_analysis['quarter_stats'][name]['mean'] for name in quarter_names]
    axes[2, 0].bar(range(len(quarter_names)), quarter_means)
    axes[2, 0].set_title('Quarter Mean Intensities')
    axes[2, 0].set_xticks(range(len(quarter_names)))
    axes[2, 0].set_xticklabels(quarter_names, rotation=45, ha='right')
    axes[2, 0].set_ylabel('Mean Intensity')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Region differences
    diff_names = list(streak_analysis['region_differences'].keys())
    diff_values = list(streak_analysis['region_differences'].values())
    axes[2, 1].bar(diff_names, diff_values)
    axes[2, 1].set_title('Region Intensity Differences')
    axes[2, 1].set_ylabel('Intensity Difference')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Image with quarter boundaries
    quarter_overlay = ref_array.copy()
    h, w = ref_array.shape
    # Draw quarter boundaries
    quarter_overlay[h//4, :] = ref_array.max()  # Horizontal line
    quarter_overlay[3*h//4, :] = ref_array.max()  # Horizontal line
    quarter_overlay[:, w//4] = ref_array.max()  # Vertical line
    quarter_overlay[:, 3*w//4] = ref_array.max()  # Vertical line
    quarter_overlay[:, w//2] = ref_array.max()  # Center vertical line
    quarter_overlay[h//2, :] = ref_array.max()  # Center horizontal line
    axes[2, 2].imshow(quarter_overlay, cmap='gray')
    axes[2, 2].set_title('Image with Region Boundaries')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / f'reference_analysis_regional_{reference_path.stem}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reference analysis saved to: {save_path}")
    
    return ref_array, streak_analysis


def analyze_processing_pipeline(bmp_path: Path, reference_path: Path, save_dir: Path):
    """Analyze each step of the processing pipeline to identify where streaks are introduced."""
    print(f"\n{'='*60}")
    print("PROCESSING PIPELINE ANALYSIS")
    print(f"{'='*60}")
    
    # Step 1: Load raw BMP
    print(f"Step 1: Loading raw BMP: {bmp_path.name}")
    with Image.open(bmp_path) as img:
        img = img.convert('L')
        raw_array = np.array(img)
    
    print(f"Raw image shape: {raw_array.shape}")
    print(f"Raw image range: [{raw_array.min()}, {raw_array.max()}]")
    print(f"Raw image mean: {raw_array.mean():.2f}, std: {raw_array.std():.2f}")
    
    # Step 2: Apply non-local means filtering
    print(f"Step 2: Applying non-local means filtering...")
    filtered_array = apply_nonlocal_means(raw_array, h=6.0, template_window_size=3, search_window_size=10)
    
    print(f"Filtered image range: [{filtered_array.min()}, {filtered_array.max()}]")
    print(f"Filtered image mean: {filtered_array.mean():.2f}, std: {filtered_array.std():.2f}")
    
    # Step 3: Apply peak alignment normalization
    print(f"Step 3: Applying peak alignment normalization...")
    normalizer = ImageNormalizer(reference_path)
    normalized_array = normalizer._peak_alignment(filtered_array)
    
    print(f"Normalized image range: [{normalized_array.min()}, {normalized_array.max()}]")
    print(f"Normalized image mean: {normalized_array.mean():.2f}, std: {normalized_array.std():.2f}")
    
    # Analyze each step for regional intensity changes
    def analyze_step(image, name):
        streaks = detect_regional_intensity_changes(image, region_size=1000, sensitivity='medium')
        print(f"{name} - Vertical streak pixels: {len(streaks['vertical_streaks'])} ({len(streaks['vertical_streaks'])/image.shape[1]*100:.3f}%)")
        print(f"{name} - Horizontal streak pixels: {len(streaks['horizontal_streaks'])} ({len(streaks['horizontal_streaks'])/image.shape[0]*100:.3f}%)")
        print(f"{name} - Left vs Right difference: {streaks['region_differences']['left_vs_right']:.2f}")
        return streaks
    
    raw_streaks = analyze_step(raw_array, "Raw")
    filtered_streaks = analyze_step(filtered_array, "Filtered")
    normalized_streaks = analyze_step(normalized_array, "Normalized")
    
    # Create comprehensive visualization (save instead of show)
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    
    # Row 1: Raw image analysis
    axes[0, 0].imshow(raw_array, cmap='gray')
    axes[0, 0].set_title(f'Raw Image\n{bmp_path.name}')
    axes[0, 0].axis('off')
    
    # Raw regional means
    im1 = axes[0, 1].imshow(raw_streaks['region_means'], cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Raw Regional Means')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.6)
    
    axes[0, 2].plot(raw_streaks['col_means'])
    axes[0, 2].set_title('Raw Column Means')
    axes[0, 2].set_ylabel('Mean Intensity')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[0, 3].hist(raw_array.ravel(), bins=50, alpha=0.7, density=True, color='blue')
    axes[0, 3].set_title(f'Raw Histogram\nMean: {raw_array.mean():.1f}')
    axes[0, 3].set_ylabel('Density')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: Filtered image analysis
    axes[1, 0].imshow(filtered_array, cmap='gray')
    axes[1, 0].set_title('Filtered Image')
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(filtered_streaks['region_means'], cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Filtered Regional Means')
    plt.colorbar(im2, ax=axes[1, 1], shrink=0.6)
    
    axes[1, 2].plot(filtered_streaks['col_means'])
    axes[1, 2].set_title('Filtered Column Means')
    axes[1, 2].set_ylabel('Mean Intensity')
    axes[1, 2].grid(True, alpha=0.3)
    
    axes[1, 3].hist(filtered_array.ravel(), bins=50, alpha=0.7, density=True, color='green')
    axes[1, 3].set_title(f'Filtered Histogram\nMean: {filtered_array.mean():.1f}')
    axes[1, 3].set_ylabel('Density')
    axes[1, 3].grid(True, alpha=0.3)
    
    # Row 3: Normalized image analysis
    axes[2, 0].imshow(normalized_array, cmap='gray')
    axes[2, 0].set_title('Normalized Image')
    axes[2, 0].axis('off')
    
    im3 = axes[2, 1].imshow(normalized_streaks['region_means'], cmap='viridis', aspect='auto')
    axes[2, 1].set_title('Normalized Regional Means')
    plt.colorbar(im3, ax=axes[2, 1], shrink=0.6)
    
    axes[2, 2].plot(normalized_streaks['col_means'])
    axes[2, 2].set_title('Normalized Column Means')
    axes[2, 2].set_xlabel('Column')
    axes[2, 2].set_ylabel('Mean Intensity')
    axes[2, 2].grid(True, alpha=0.3)
    
    axes[2, 3].hist(normalized_array.ravel(), bins=50, alpha=0.7, density=True, color='red')
    axes[2, 3].set_title(f'Normalized Histogram\nMean: {normalized_array.mean():.1f}')
    axes[2, 3].set_xlabel('Pixel Value')
    axes[2, 3].set_ylabel('Density')
    axes[2, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'pipeline_analysis_regional_{bmp_path.stem}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pipeline analysis saved to: {save_path}")
    
    return {
        'raw': raw_array,
        'filtered': filtered_array,
        'normalized': normalized_array,
        'raw_streaks': raw_streaks,
        'filtered_streaks': filtered_streaks,
        'normalized_streaks': normalized_streaks
    }


def analyze_h5_file(h5_path: Path, save_dir: Path):
    """Analyze an existing H5 file for streak patterns."""
    print(f"\n{'='*60}")
    print("HDF5 FILE ANALYSIS")
    print(f"{'='*60}")
    
    with h5py.File(h5_path, 'r') as h5f:
        images = h5f['images']
        times = h5f['times'][:]
        
        print(f"HDF5 file: {h5_path.name}")
        print(f"Number of images: {images.shape[0]}")
        print(f"Image shape: {images.shape[1:]} (height x width)")
        print(f"Duration: {times[-1]:.2f} seconds")
        
        # Analyze first, middle, and last images for streaks
        indices = [0, len(images)//2, len(images)-1]
        names = ['First', 'Middle', 'Last']
        
        streak_data = []
        
        for idx, name in zip(indices, names):
            img = images[idx]
            
            # Use regional intensity change detection
            streaks = detect_regional_intensity_changes(img, region_size=1000, sensitivity='medium')
            
            print(f"{name} image (t={times[idx]:.1f}s):")
            print(f"  Vertical streak pixels: {len(streaks['vertical_streaks'])} ({len(streaks['vertical_streaks'])/img.shape[1]*100:.3f}%)")
            print(f"  Horizontal streak pixels: {len(streaks['horizontal_streaks'])} ({len(streaks['horizontal_streaks'])/img.shape[0]*100:.3f}%)")
            print(f"  Left vs Right difference: {streaks['region_differences']['left_vs_right']:.2f}")
            print(f"  Mean: {img.mean():.2f}, Std: {img.std():.2f}")
            
            streak_data.append({
                'name': name,
                'index': idx,
                'time': times[idx],
                'image': img,
                'streaks': streaks
            })
        
        # Create visualization (save instead of show)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        for i, data in enumerate(streak_data):
            img = data['image']
            streaks = data['streaks']
            
            # Image
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'{data["name"]} Image\nt={data["time"]:.1f}s')
            axes[i, 0].axis('off')
            
            # Regional means
            im = axes[i, 1].imshow(streaks['region_means'], cmap='viridis', aspect='auto')
            axes[i, 1].set_title('Regional Means')
            plt.colorbar(im, ax=axes[i, 1], shrink=0.6)
            
            # Column means
            axes[i, 2].plot(streaks['col_means'])
            axes[i, 2].set_title('Column Means')
            axes[i, 2].set_ylabel('Mean Intensity')
            axes[i, 2].grid(True, alpha=0.3)
            
            # Histogram
            axes[i, 3].hist(img.ravel(), bins=50, alpha=0.7, density=True)
            axes[i, 3].set_title(f'Histogram\nMean: {img.mean():.1f}')
            axes[i, 3].set_ylabel('Density')
            axes[i, 3].grid(True, alpha=0.3)
        
        axes[2, 2].set_xlabel('Column')
        axes[2, 3].set_xlabel('Pixel Value')
        
        plt.tight_layout()
        save_path = save_dir / f'h5_analysis_regional_{h5_path.stem}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"H5 analysis saved to: {save_path}")
        
        return streak_data


def main():
    parser = argparse.ArgumentParser(description="Investigate regional intensity changes in normalized precipitation data")
    parser.add_argument('--h5_file', help='Path to HDF5 file to analyze')
    parser.add_argument('--bmp_folder', help='Path to folder containing BMP files to analyze')
    parser.add_argument('--reference_image', default='/Users/hm114/Desktop/Precipitation_Data_test/image/2025_0628/3mM-0628-4-mix/3mM-0628-4-mix_20250628213544658_17_AfterFPN.bmp', help='Path to reference image')
    parser.add_argument('--bmp_file', help='Specific BMP file to analyze in pipeline')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium', help='Detection sensitivity')
    parser.add_argument('--region_size', type=int, default=1000, help='Size of regions for analysis (default: 1000)')
    parser.add_argument('--output_dir', default='streak_analysis_output', help='Directory to save analysis plots')
    
    args = parser.parse_args()
    
    print("🔍 REGIONAL INTENSITY INVESTIGATION TOOL")
    print("Analyzing regional intensity changes that appear as streaks...")
    print(f"Using {args.sensitivity} sensitivity detection")
    print(f"Region size: {args.region_size} pixels")
    
    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True)
    print(f"Saving analysis plots to: {save_dir.absolute()}")
    
    # 1. Always analyze the reference image first
    if Path(args.reference_image).exists():
        ref_array, ref_analysis = analyze_reference_image(Path(args.reference_image), save_dir)
        
        if len(ref_analysis['vertical_streaks']) > 0 or len(ref_analysis['horizontal_streaks']) > 0:
            print("\n⚠️  WARNING: Reference image contains regional intensity variations!")
            print("   This could be the source of streaks in normalized data.")
        else:
            print("\n✅ Reference image appears uniform (no obvious regional variations)")
    else:
        print(f"❌ Reference image not found: {args.reference_image}")
        return
    
    # 2. Analyze HDF5 file if provided
    if args.h5_file:
        h5_path = Path(args.h5_file)
        if h5_path.exists():
            h5_analysis = analyze_h5_file(h5_path, save_dir)
        else:
            print(f"❌ HDF5 file not found: {args.h5_file}")
    
    # 3. Analyze processing pipeline if BMP file/folder provided
    if args.bmp_file:
        bmp_path = Path(args.bmp_file)
        if bmp_path.exists():
            pipeline_analysis = analyze_processing_pipeline(bmp_path, Path(args.reference_image), save_dir)
        else:
            print(f"❌ BMP file not found: {args.bmp_file}")
    
    elif args.bmp_folder:
        bmp_folder = Path(args.bmp_folder)
        if bmp_folder.exists():
            # Find a sample BMP file
            bmp_files = list(bmp_folder.glob("*.bmp"))
            if bmp_files:
                sample_bmp = bmp_files[0]
                print(f"\nAnalyzing sample BMP: {sample_bmp.name}")
                pipeline_analysis = analyze_processing_pipeline(sample_bmp, Path(args.reference_image), save_dir)
            else:
                print(f"❌ No BMP files found in: {args.bmp_folder}")
        else:
            print(f"❌ BMP folder not found: {args.bmp_folder}")
    
    print(f"\n✅ Investigation complete! Analysis plots saved to: {save_dir.absolute()}")
    print("\nImproved Recommendations:")
    print("1. If reference image shows regional variations: Use a different reference image")
    print("2. If streaks appear only after filtering: Adjust non-local means parameters or disable filtering")
    print("3. If streaks appear only after normalization: Try histogram_matching or min_max methods instead")
    print("4. If streaks are in raw BMPs: Check imaging system calibration and sensor uniformity")
    print("5. Check 'Left vs Right difference' values - large differences indicate left/right intensity variations")


if __name__ == "__main__":
    main() 