#!/usr/bin/env python3
"""
Image Normalization Script

This script normalizes a set of images to match the histogram distribution 
of a reference image using various histogram matching techniques.

Author: Assistant
Date: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from skimage import exposure
from skimage.io import imread, imsave
import os
from tqdm import tqdm

class ImageNormalizer:
    """Class to handle image normalization using various histogram matching techniques."""
    
    def __init__(self, reference_image_path):
        """
        Initialize the normalizer with a reference image.
        
        Args:
            reference_image_path (str): Path to the reference image
        """
        self.reference_image_path = Path(reference_image_path)
        self.reference_image = None
        self.reference_histogram = None
        self.reference_peak = None
        self.reference_spread = None
        
        # Load and process reference image
        self._load_reference_image()
        
    def _load_reference_image(self):
        """Load the reference image and compute its histogram statistics."""
        print(f"Loading reference image: {self.reference_image_path}")
        
        if not self.reference_image_path.exists():
            raise FileNotFoundError(f"Reference image not found: {self.reference_image_path}")
            
        # Load the image
        self.reference_image = imread(str(self.reference_image_path))
        
        # Convert to grayscale if it's a color image
        if len(self.reference_image.shape) == 3:
            self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2GRAY)
            
        # Compute reference statistics
        self.reference_peak = self._find_histogram_peak(self.reference_image)
        self.reference_spread = np.std(self.reference_image)
        
        print(f"Reference image shape: {self.reference_image.shape}")
        print(f"Reference image dtype: {self.reference_image.dtype}")
        print(f"Reference image range: [{self.reference_image.min()}, {self.reference_image.max()}]")
        print(f"Reference histogram peak: {self.reference_peak}")
        print(f"Reference spread (std): {self.reference_spread:.2f}")
        
    def _find_histogram_peak(self, image):
        """
        Find the peak (mode) of the histogram.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            int: Peak value (mode) of the histogram
        """
        # Compute histogram
        hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 255))
        
        # Find the bin with maximum frequency
        peak_bin = np.argmax(hist)
        peak_value = bins[peak_bin]
        
        return int(peak_value)
    
    def _compute_spread_metrics(self, image):
        """
        Compute spread metrics for an image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            dict: Dictionary containing spread metrics
        """
        return {
            'std': np.std(image),
            'var': np.var(image),
            'iqr': np.percentile(image, 75) - np.percentile(image, 25),
            'range': image.max() - image.min()
        }
        
    def normalize_image(self, input_image_path, method='histogram_matching'):
        """
        Normalize an image to match the reference image.
        
        Args:
            input_image_path (str): Path to the image to normalize
            method (str): Normalization method ('histogram_matching', 'min_max', 'z_score', 
                         'peak_align', 'peak_spread_align')
            
        Returns:
            np.ndarray: Normalized image
        """
        # Load input image
        input_image = imread(str(input_image_path))
        
        # Convert to grayscale if it's a color image
        if len(input_image.shape) == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            
        if method == 'histogram_matching':
            normalized_image = self._histogram_matching(input_image)
        elif method == 'min_max':
            normalized_image = self._min_max_normalization(input_image)
        elif method == 'z_score':
            normalized_image = self._z_score_normalization(input_image)
        elif method == 'peak_align':
            normalized_image = self._peak_alignment(input_image)
        elif method == 'peak_spread_align':
            normalized_image = self._peak_spread_alignment(input_image)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return normalized_image
    
    def _histogram_matching(self, input_image):
        """
        Apply histogram matching to make input image match reference histogram.
        
        Args:
            input_image (np.ndarray): Input image to normalize
            
        Returns:
            np.ndarray: Histogram-matched image
        """
        # Use skimage's match_histograms function
        matched = exposure.match_histograms(input_image, self.reference_image)
        return matched.astype(input_image.dtype)
    
    def _min_max_normalization(self, input_image):
        """
        Apply min-max normalization to match reference image range.
        
        Args:
            input_image (np.ndarray): Input image to normalize
            
        Returns:
            np.ndarray: Min-max normalized image
        """
        ref_min, ref_max = self.reference_image.min(), self.reference_image.max()
        img_min, img_max = input_image.min(), input_image.max()
        
        # Normalize to [0, 1] then scale to reference range
        normalized = (input_image - img_min) / (img_max - img_min)
        normalized = normalized * (ref_max - ref_min) + ref_min
        
        return normalized.astype(input_image.dtype)
    
    def _z_score_normalization(self, input_image):
        """
        Apply z-score normalization to match reference mean and std.
        
        Args:
            input_image (np.ndarray): Input image to normalize
            
        Returns:
            np.ndarray: Z-score normalized image
        """
        ref_mean, ref_std = self.reference_image.mean(), self.reference_image.std()
        img_mean, img_std = input_image.mean(), input_image.std()
        
        # Z-score normalization
        normalized = (input_image - img_mean) / img_std
        normalized = normalized * ref_std + ref_mean
        
        # Ensure values are within reasonable range
        normalized = np.clip(normalized, 0, np.iinfo(input_image.dtype).max)
        
        return normalized.astype(input_image.dtype)
    
    def _peak_alignment(self, input_image):
        """
        Align histogram peaks by shifting and clipping.
        
        Args:
            input_image (np.ndarray): Input image to normalize
            
        Returns:
            np.ndarray: Peak-aligned image
        """
        # Find the peak of the input image
        input_peak = self._find_histogram_peak(input_image)
        
        # If peaks are already aligned, return original
        if input_peak == self.reference_peak:
            return input_image
        
        # Find the difference in peaks
        difference = self.reference_peak - input_peak
        
        # Simple approach: use int32 for the arithmetic, then clip and convert back
        shifted = input_image.astype(np.int32) + difference
        clipped = np.clip(shifted, 0, 255)
        
        return clipped.astype(np.uint8)
    
    def _peak_spread_alignment(self, input_image):
        """
        Align both histogram peaks and spread to match reference.
        
        Args:
            input_image (np.ndarray): Input image to normalize
            
        Returns:
            np.ndarray: Peak and spread aligned image
        """
        # First align the peaks
        input_peak = self._find_histogram_peak(input_image)
        shift = self.reference_peak - input_peak
        peak_aligned = input_image.astype(np.float32) + shift
        
        # Then adjust the spread around the aligned peak
        input_spread = np.std(peak_aligned)
        
        if input_spread > 0:  # Avoid division by zero
            # Center around the peak, scale spread, then shift back
            centered = peak_aligned - self.reference_peak
            spread_adjusted = centered * (self.reference_spread / input_spread)
            normalized = spread_adjusted + self.reference_peak
        else:
            normalized = peak_aligned
        
        # Ensure values are within valid range
        normalized = np.clip(normalized, 0, 255)
        
        return normalized.astype(input_image.dtype)
    
    def plot_histogram_comparison(self, input_image_path, normalized_image, save_path=None):
        """
        Plot histogram comparison between original, reference, and normalized images.
        
        Args:
            input_image_path (str): Path to original image
            normalized_image (np.ndarray): Normalized image
            save_path (str): Path to save the plot (optional)
        """
        # Load original image
        original_image = imread(str(input_image_path))
        if len(original_image.shape) == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate peaks for annotation
        orig_peak = self._find_histogram_peak(original_image)
        norm_peak = self._find_histogram_peak(normalized_image)
        
        # Create comparison plot with images on top row and combined histogram below
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Top row: Images
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.reference_image, cmap='gray')
        axes[0, 1].set_title('Reference Image')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(normalized_image, cmap='gray')
        axes[0, 2].set_title('Normalized Image')
        axes[0, 2].axis('off')
        
        # Bottom row: Combined histogram spanning all three columns
        # Remove the individual histogram subplots
        axes[1, 1].remove()
        axes[1, 2].remove()
        
        # Create a single subplot spanning the bottom row
        gs = fig.add_gridspec(2, 3)
        ax_hist = fig.add_subplot(gs[1, :])
        
        # Plot all histograms on the same graph
        ax_hist.hist(original_image.ravel(), bins=range(257), alpha=0.6, color='blue', 
                    label=f'Original (Peak: {orig_peak}, Std: {np.std(original_image):.1f})')
        ax_hist.hist(self.reference_image.ravel(), bins=range(257), alpha=0.6, color='green', 
                    label=f'Reference (Peak: {self.reference_peak}, Std: {self.reference_spread:.1f})')
        ax_hist.hist(normalized_image.ravel(), bins=range(257), alpha=0.6, color='red', 
                    label=f'Normalized (Peak: {norm_peak}, Std: {np.std(normalized_image):.1f})')
        
        # Add peak lines
        ax_hist.axvline(orig_peak, color='blue', linestyle='--', alpha=0.8)
        ax_hist.axvline(self.reference_peak, color='green', linestyle='--', alpha=0.8)
        ax_hist.axvline(norm_peak, color='red', linestyle='--', alpha=0.8)
        
        ax_hist.set_title('Histogram Comparison', fontsize=14, fontweight='bold')
        ax_hist.set_xlabel('Pixel Intensity')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_xlim(0, 255)
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Remove the unused subplot
        axes[1, 0].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Histogram comparison saved to: {save_path}")
        
        plt.show()

def normalize_images_in_directory(input_dir, output_dir, reference_image_name, method='histogram_matching'):
    """
    Normalize all images in a directory to match a reference image.
    
    Args:
        input_dir (str): Directory containing images to normalize
        output_dir (str): Directory to save normalized images
        reference_image_name (str): Name of the reference image in input_dir
        method (str): Normalization method
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Path to reference image
    reference_image_path = input_path / reference_image_name
    
    # Initialize normalizer
    normalizer = ImageNormalizer(reference_image_path)
    
    # Get all image files in input directory
    image_extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp']
    image_files = [f for f in input_path.iterdir() 
                  if f.suffix.lower() in image_extensions and f.name != reference_image_name]
    
    print(f"Found {len(image_files)} images to normalize (excluding reference)")
    print(f"Using normalization method: {method}")
    
    # Copy reference image to output directory
    reference_output_path = output_path / reference_image_name
    if not reference_output_path.exists():
        import shutil
        shutil.copy2(reference_image_path, reference_output_path)
        print(f"Copied reference image to: {reference_output_path}")
    
    # Process each image
    for img_file in tqdm(image_files, desc="Normalizing images"):
        try:
            # Normalize the image
            normalized_image = normalizer.normalize_image(img_file, method=method)
            
            # Save normalized image
            output_file = output_path / f"normalized_{img_file.name}"
            imsave(str(output_file), normalized_image)
            
            print(f"Normalized and saved: {output_file}")
            
            # Create histogram comparison for first image
            if img_file == image_files[0]:
                comparison_plot_path = output_path / f"histogram_comparison_{img_file.stem}.png"
                normalizer.plot_histogram_comparison(img_file, normalized_image, str(comparison_plot_path))
                
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

def main():
    """Main function to run the normalization script."""
    parser = argparse.ArgumentParser(description="Normalize images to match a reference image")
    parser.add_argument("--input_dir", default="images_for_normalisation", 
                       help="Directory containing images to normalize")
    parser.add_argument("--output_dir", default="normalized_images", 
                       help="Directory to save normalized images")
    parser.add_argument("--reference_image", default="3mM-0628-4-mix_20250628213544658_17_o.tiff",
                       help="Name of the reference image")
    parser.add_argument("--method", 
                       choices=['histogram_matching', 'min_max', 'z_score', 'peak_align', 'peak_spread_align'], 
                       default='histogram_matching',
                       help="Normalization method: "
                            "histogram_matching (full histogram matching), "
                            "min_max (range normalization), "
                            "z_score (mean/std normalization), "
                            "peak_align (align histogram peaks only), "
                            "peak_spread_align (align peaks and spread)")
    
    args = parser.parse_args()
    
    # Run normalization
    normalize_images_in_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        reference_image_name=args.reference_image,
        method=args.method
    )
    
    print("\nNormalization complete!")
    print(f"Normalized images saved in: {args.output_dir}")
    
    # Print method explanation
    method_explanations = {
        'histogram_matching': 'Full histogram distribution matching',
        'min_max': 'Min-max range normalization',
        'z_score': 'Mean and standard deviation matching',
        'peak_align': 'Histogram peak alignment only (preserves spread)',
        'peak_spread_align': 'Both peak and spread alignment'
    }
    print(f"Method used: {method_explanations[args.method]}")

if __name__ == "__main__":
    main() 