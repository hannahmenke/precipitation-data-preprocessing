#!/usr/bin/env python3
"""
Demo Script for XGBoost Precipitation Particle Classification

This script demonstrates how to run the XGBoost classifier on the precipitation data.
It serves as a simple example and can be customized for different datasets.
"""

import os
import sys
from pathlib import Path

def main():
    print("🧠 XGBoost Precipitation Particle Classification Demo")
    print("=" * 60)
    
    # Check if training data exists
    training_data_dir = Path("training_data")
    if not training_data_dir.exists():
        print("❌ Error: training_data directory not found!")
        print("Please ensure you have the training_data folder with Excel files.")
        return
    
    # Check for Excel files
    excel_files = list(training_data_dir.glob("*.xlsx"))
    if not excel_files:
        print("❌ Error: No Excel files found in training_data directory!")
        print("Please ensure you have .xlsx files with labeled particle data.")
        return
    
    print(f"✓ Found {len(excel_files)} Excel file(s):")
    for file in excel_files:
        print(f"  - {file.name}")
    
    print("\n🚀 Running XGBoost Classification...")
    print("This will:")
    print("  1. Load and preprocess the Excel data")
    print("  2. Train an XGBoost model with hyperparameter tuning")
    print("  3. Evaluate model performance")
    print("  4. Generate visualization plots")
    print("  5. Save results to PNG files")
    
    # Run the classifier
    cmd = "python excel_xgboost_classifier.py --save-plots"
    print(f"\nExecuting: {cmd}")
    print("-" * 60)
    
    exit_code = os.system(cmd)
    
    print("-" * 60)
    if exit_code == 0:
        print("✅ Classification completed successfully!")
        print("\nGenerated files:")
        print("  📊 xgboost_classification_results.png - Visualization plots")
        print("\nNext steps:")
        print("  - Review the classification results")
        print("  - Examine feature importance rankings") 
        print("  - Analyze confusion matrix for model performance")
        print("  - Try different parameters with excel_xgboost_classifier.py")
    else:
        print("❌ Classification failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("  - Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  - Check that Excel files have the correct format")
        print("  - Verify that 'type' column exists in your data")

if __name__ == "__main__":
    main() 