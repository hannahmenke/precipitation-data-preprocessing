#!/usr/bin/env python3
"""
Timeseries Analysis Script for HDF5 Files

This script analyzes HDF5 timeseries files from the precipitation data pipeline.
It displays:
- First, last, and middle images
- Difference image between first and last
- Histograms for first, last, and middle images

Usage:
    python analyze_timeseries.py --file path/to/timeseries.h5 [--save output.png]
"""
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_timeseries(h5_file_path, save_path=None):
    """
    Analyze a timeseries HDF5 file and create comprehensive visualization.
    
    Args:
        h5_file_path: Path to HDF5 file
        save_path: Optional path to save the plot
    """
    print(f"Analyzing timeseries: {h5_file_path}")
    
    with h5py.File(h5_file_path, 'r') as h5f:
        # Load data
        images = h5f['images']
        times = h5f['times'][:]
        num_images = images.shape[0]
        
        print(f"Number of images: {num_images}")
        print(f"Image shape: {images.shape[1:]} (height x width)")
        print(f"Duration: {times[-1]:.2f} seconds")
        print(f"Frame rate: {num_images / times[-1]:.2f} fps")
        
        # Get indices for first, middle, last
        first_idx = 0
        last_idx = num_images - 1
        middle_idx = num_images // 2
        
        # Load specific images
        first_img = images[first_idx]
        middle_img = images[middle_idx] 
        last_img = images[last_idx]
        
        # Calculate difference image
        diff_img = last_img.astype(np.float32) - first_img.astype(np.float32)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Image display parameters
        vmin = min(first_img.min(), middle_img.min(), last_img.min())
        vmax = max(first_img.max(), middle_img.max(), last_img.max())
        
        # Row 1: Images
        # First image
        ax1 = plt.subplot(3, 4, 1)
        plt.imshow(first_img, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f'First Image (t={times[first_idx]:.2f}s)\nIndex: {first_idx}')
        plt.axis('off')
        
        # Middle image
        ax2 = plt.subplot(3, 4, 2)
        plt.imshow(middle_img, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f'Middle Image (t={times[middle_idx]:.2f}s)\nIndex: {middle_idx}')
        plt.axis('off')
        
        # Last image
        ax3 = plt.subplot(3, 4, 3)
        plt.imshow(last_img, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f'Last Image (t={times[last_idx]:.2f}s)\nIndex: {last_idx}')
        plt.axis('off')
        
        # Difference image
        ax4 = plt.subplot(3, 4, 4)
        diff_abs_max = max(abs(diff_img.min()), abs(diff_img.max()))
        plt.imshow(diff_img, cmap='RdBu_r', vmin=-diff_abs_max, vmax=diff_abs_max)
        plt.title(f'Difference (Last - First)\nRange: [{diff_img.min():.1f}, {diff_img.max():.1f}]')
        plt.axis('off')
        plt.colorbar(shrink=0.6)
        
        # Row 2: Histograms
        # First image histogram
        ax5 = plt.subplot(3, 4, 5)
        plt.hist(first_img.ravel(), bins=50, alpha=0.7, color='blue', density=True)
        plt.title(f'First Image Histogram\nMean: {first_img.mean():.1f}, Std: {first_img.std():.1f}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Middle image histogram
        ax6 = plt.subplot(3, 4, 6)
        plt.hist(middle_img.ravel(), bins=50, alpha=0.7, color='green', density=True)
        plt.title(f'Middle Image Histogram\nMean: {middle_img.mean():.1f}, Std: {middle_img.std():.1f}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Last image histogram
        ax7 = plt.subplot(3, 4, 7)
        plt.hist(last_img.ravel(), bins=50, alpha=0.7, color='red', density=True)
        plt.title(f'Last Image Histogram\nMean: {last_img.mean():.1f}, Std: {last_img.std():.1f}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Difference histogram
        ax8 = plt.subplot(3, 4, 8)
        plt.hist(diff_img.ravel(), bins=50, alpha=0.7, color='purple', density=True)
        plt.title(f'Difference Histogram\nMean: {diff_img.mean():.1f}, Std: {diff_img.std():.1f}')
        plt.xlabel('Difference Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Row 3: Time series analysis
        # Mean intensity over time
        ax9 = plt.subplot(3, 4, 9)
        mean_intensities = [images[i].mean() for i in range(0, num_images, max(1, num_images//50))]
        time_points = [times[i] for i in range(0, num_images, max(1, num_images//50))]
        plt.plot(time_points, mean_intensities, 'b-', linewidth=2)
        plt.scatter([times[first_idx], times[middle_idx], times[last_idx]], 
                   [first_img.mean(), middle_img.mean(), last_img.mean()], 
                   c=['blue', 'green', 'red'], s=100, zorder=5)
        plt.title('Mean Intensity Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Pixel Value')
        plt.grid(True, alpha=0.3)
        
        # Standard deviation over time
        ax10 = plt.subplot(3, 4, 10)
        std_intensities = [images[i].std() for i in range(0, num_images, max(1, num_images//50))]
        plt.plot(time_points, std_intensities, 'g-', linewidth=2)
        plt.scatter([times[first_idx], times[middle_idx], times[last_idx]], 
                   [first_img.std(), middle_img.std(), last_img.std()], 
                   c=['blue', 'green', 'red'], s=100, zorder=5)
        plt.title('Std Dev Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Pixel Std Dev')
        plt.grid(True, alpha=0.3)
        
        # Combined histogram comparison
        ax11 = plt.subplot(3, 4, 11)
        plt.hist(first_img.ravel(), bins=50, alpha=0.5, color='blue', density=True, label='First')
        plt.hist(middle_img.ravel(), bins=50, alpha=0.5, color='green', density=True, label='Middle')
        plt.hist(last_img.ravel(), bins=50, alpha=0.5, color='red', density=True, label='Last')
        plt.title('Histogram Comparison')
        plt.xlabel('Pixel Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Summary statistics
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Get metadata if available
        folder_name = h5f.attrs.get('folder_name', 'Unknown')
        creation_time = h5f.attrs.get('creation_time', 'Unknown')
        
        summary_text = f"""
Dataset Summary:
• Folder: {folder_name}
• Images: {num_images}
• Duration: {times[-1]:.2f}s
• Frame rate: {num_images/times[-1]:.2f} fps
• Image size: {images.shape[1]}×{images.shape[2]}
• Data type: {images.dtype}

Intensity Changes:
• First mean: {first_img.mean():.1f}
• Last mean: {last_img.mean():.1f}
• Change: {last_img.mean()-first_img.mean():.1f}

• First std: {first_img.std():.1f}
• Last std: {last_img.std():.1f}
• Change: {last_img.std()-first_img.std():.1f}

Created: {creation_time}
        """
        plt.text(0.05, 0.95, summary_text.strip(), transform=ax12.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analysis saved to: {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze HDF5 timeseries files")
    parser.add_argument('--file', required=True, help='Path to HDF5 timeseries file')
    parser.add_argument('--save', help='Path to save the analysis plot (optional)')
    args = parser.parse_args()
    
    h5_file = Path(args.file)
    if not h5_file.exists():
        print(f"Error: File not found: {h5_file}")
        return
    
    analyze_timeseries(h5_file, args.save)


if __name__ == "__main__":
    main() 