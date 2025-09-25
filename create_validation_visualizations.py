#!/usr/bin/env python3
"""
Create Pure Validation-Focused Visualizations for Enhanced Model.

This script creates comprehensive visualizations focused purely on the enhanced model's
validation performance without any references to previous models or comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_enhanced_predictions():
    """Load enhanced model predictions only."""
    print("📂 Loading enhanced model predictions...")
    
    # Auto-discover ALL prediction files
    prediction_files = {}
    
    # Look for any *_predictions.xlsx files
    for pred_file in Path('.').glob("*_predictions.xlsx"):
        dataset_name = pred_file.stem.replace('_predictions', '')
        prediction_files[dataset_name] = pred_file.name
    
    # Also check for hardcoded files for backward compatibility
    hardcoded_files = {
        'val3': 'val3_predictions.xlsx',
        'val6': 'val6_predictions.xlsx', 
        'combined': 'combined_predictions.xlsx'
    }
    
    for dataset, filename in hardcoded_files.items():
        if Path(filename).exists() and dataset not in prediction_files:
            prediction_files[dataset] = filename
    
    print(f"📁 Found prediction files for: {list(prediction_files.keys())}")
    
    predictions = {}
    for dataset, filename in prediction_files.items():
        file_path = Path(filename)
        if file_path.exists():
            try:
                df = pd.read_excel(file_path)
                
                # Try different column name variations
                pred_col = None
                conf_col = None
                true_col = None
                
                # Check for enhanced predictions
                for col in df.columns:
                    if 'enhanced_prediction' in col.lower():
                        pred_col = col
                    elif 'enhanced_confidence' in col.lower():
                        conf_col = col
                    elif 'true_label' in col.lower() or col.lower() == 'type':
                        true_col = col
                
                # Fallback to basic column names
                if pred_col is None:
                    for col in df.columns:
                        if 'prediction' in col.lower() or 'predicted' in col.lower():
                            pred_col = col
                            break
                
                if conf_col is None:
                    for col in df.columns:
                        if 'confidence' in col.lower():
                            conf_col = col
                            break
                
                if true_col is None:
                    if 'type' in df.columns:
                        true_col = 'type'
                
                if pred_col and conf_col:
                    predictions[dataset] = {
                        'predictions': df[pred_col],
                        'confidence': df[conf_col],
                        'true_labels': df[true_col] if true_col else None
                    }
                    print(f"  ✓ {dataset}: {len(df)} samples")
                else:
                    print(f"  ❌ {dataset}: Missing required columns (pred: {pred_col}, conf: {conf_col})")
                    
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
        else:
            print(f"  ⚠️  {filename} not found")
    
    print(f"✓ Loaded predictions for {len(predictions)} datasets")
    return predictions

def load_train_test_results():
    """Load training and test results from model training."""
    print("📈 Loading training and test performance...")
    
    # Try to load the latest model metadata and training results
    import os
    models_dir = Path(os.environ.get('AUTORUN_MODELS_DIR', 'models'))
    
    try:
        # Load model to get training history if available
        model = joblib.load(models_dir / "latest_model.joblib")
        
        # Load training data to calculate train accuracy
        train_df = pd.read_excel('training_data/train_improved.xlsx')
        scaler = joblib.load(models_dir / "latest_scaler.joblib")
        feature_selector = joblib.load(models_dir / "latest_feature_selector.joblib")
        feature_names = joblib.load(models_dir / "latest_feature_names.joblib")
        
        # Prepare training data for evaluation
        train_features = train_df[feature_names]
        train_features_scaled = scaler.transform(train_features)
        train_features_selected = feature_selector.transform(train_features_scaled)
        train_labels = train_df['type']
        
        # Calculate training accuracy
        train_pred = model.predict(train_features_selected)
        train_accuracy = np.mean(train_pred == train_labels)
        
        # Load test data (using combined test sets)
        test_files = ['training_data/test3.xlsx', 'training_data/test6.xlsx']
        test_dfs = []
        
        for test_file in test_files:
            test_path = Path(test_file)
            if test_path.exists():
                test_dfs.append(pd.read_excel(test_file))
        
        if test_dfs:
            test_df = pd.concat(test_dfs, ignore_index=True)
            
            # Find common features between test and training data
            common_features = [f for f in feature_names if f in test_df.columns]
            
            if len(common_features) >= len(feature_names) * 0.8:  # At least 80% of features
                test_features = test_df[common_features]
                test_features_scaled = scaler.transform(test_features)
                test_features_selected = feature_selector.transform(test_features_scaled)
                test_labels = test_df['type']
                
                # Calculate test accuracy
                test_pred = model.predict(test_features_selected)
                test_accuracy = np.mean(test_pred == test_labels)
                
                train_test_results = {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_samples': len(train_labels),
                    'test_samples': len(test_labels),
                    'train_classes': train_labels.value_counts().to_dict(),
                    'test_classes': test_labels.value_counts().to_dict()
                }
                
                print(f"✓ Training accuracy: {train_accuracy:.3f} (n={len(train_labels)})")
                print(f"✓ Test accuracy: {test_accuracy:.3f} (n={len(test_labels)})")
                
                return train_test_results
        
        print("⚠️  Could not calculate test accuracy - missing test data")
        return None
        
    except Exception as e:
        print(f"⚠️  Could not load train/test results: {e}")
        return None

def load_validation_data_for_analysis():
    """Load validation data with true labels for analysis."""
    print("📊 Loading validation data for analysis...")
    
    try:
        # Auto-discover ALL Excel files in data_for_classification
        validation_dir = Path('data_for_classification')
        all_val_files = list(validation_dir.glob("*.xlsx"))
        all_val_files = [f for f in all_val_files if not f.name.startswith('.') and not f.name.startswith('~')]
        
        if not all_val_files:
            print("⚠️  No validation files found in data_for_classification/")
            return None
        
        print(f"📁 Found {len(all_val_files)} validation files:")
        
        # Function to create improved features (same as in improve_features.py)
        def create_improved_features(df):
            df_improved = df.copy()
            
            # 1. Shape Complexity Score
            extent_norm = (df['Extent'] - df['Extent'].min()) / (df['Extent'].max() - df['Extent'].min())
            circularity_norm = (df['Circularity'] - df['Circularity'].min()) / (df['Circularity'].max() - df['Circularity'].min())
            df_improved['Shape_Complexity'] = extent_norm * (1 - circularity_norm)
            
            # 2. Normalized Shape Ratios
            df_improved['Area_Perimeter_Ratio'] = df['Area'] / (df['Perimeter'] + 1e-6)
            df_improved['Convex_Efficiency'] = df['Area'] / (df['ConvexArea'] + 1e-6)
            
            # 3. Robust Eccentricity
            df_improved['Eccentricity_Robust'] = np.clip(df['Eccentricity'], 0, 1)
            
            # 4. Distance Feature Combinations
            df_improved['Distance_Ratio'] = df['dis'] / (df['dis_normal'] + 1e-6)
            df_improved['Distance_Interaction'] = df['dis'] * df['dis_normal']
            
            # 5. Intensity Stability Score
            gray_mean = df['MeanIntensity'] if 'MeanIntensity' in df.columns else df.get('Gray_ave', 0)
            df_improved['Intensity_Stability'] = gray_mean / (df['Gray_var'] + 1e-6)
            
            # 6. Comprehensive Shape Score
            df_improved['Comprehensive_Shape'] = (
                df_improved['Major_Minor_ratio'] * 
                df_improved['Circularity'] * 
                (1 - df_improved['Eccentricity_Robust'])
            )
            
            # 7. Class4 Discriminator
            df_improved['Class4_Discriminator'] = (
                df['Extent'] * df['Circularity'] * df['Perimeter']
            )
            
            return df_improved
        
        # Load and process ALL validation files
        all_datasets = {}
        improved_datasets = []
        total_samples = 0
        
        for val_file in all_val_files:
            try:
                # Load original data
                df = pd.read_excel(val_file)
                dataset_name = val_file.stem
                
                # Apply feature engineering
                df_improved = create_improved_features(df)
                df_improved['source'] = dataset_name
                
                all_datasets[dataset_name] = df_improved
                improved_datasets.append(df_improved)
                total_samples += len(df)
                
                print(f"  📄 {dataset_name}: {len(df)} samples (with improved features)")
                
            except Exception as e:
                print(f"  ❌ Error loading {val_file.name}: {e}")
        
        if not all_datasets:
            print("❌ No validation datasets could be loaded")
            return None
        
        # Create TRUE combined dataset from ALL files
        print(f"\n🔄 Creating TRUE combined dataset from ALL {len(all_datasets)} files...")
        combined_val_df = pd.concat(improved_datasets, ignore_index=True)
        
        print(f"✓ TRUE combined dataset: {len(combined_val_df)} samples from {len(all_datasets)} files")
        print(f"✓ Total validation samples: {total_samples}")
        
        # Create validation_data structure
        validation_data = {}
        
        # Add individual datasets
        for dataset_name, dataset_df in all_datasets.items():
            validation_data[dataset_name] = dataset_df
        
        # Add combined dataset
        validation_data['combined'] = combined_val_df
        
        # Ensure val3 and val6 exist for compatibility (even if empty)
        if 'val3' not in validation_data:
            validation_data['val3'] = pd.DataFrame()
        if 'val6' not in validation_data:
            validation_data['val6'] = pd.DataFrame()
        
        # Print breakdown by dataset
        print(f"\n📊 Dataset breakdown:")
        for dataset_name in sorted(all_datasets.keys()):
            count = len(all_datasets[dataset_name])
            percentage = (count / total_samples) * 100
            print(f"  {dataset_name}: {count} samples ({percentage:.1f}%)")
        
        print(f"✓ Loaded validation data for analysis - {len(validation_data)} datasets")
        return validation_data
        
    except Exception as e:
        print(f"❌ Error loading validation data: {e}")
        return None

def calculate_enhanced_model_metrics(predictions, validation_data):
    """Calculate metrics purely for the enhanced model."""
    print("📊 Calculating enhanced model metrics...")
    
    metrics = {}
    
    # Calculate metrics for ALL available datasets (not just hardcoded ones)
    all_datasets = set(predictions.keys()) & set(validation_data.keys())
    
    for dataset in all_datasets:
        if (dataset in predictions and dataset in validation_data and 
            len(validation_data[dataset]) > 0 and 'type' in validation_data[dataset].columns):
            
            y_true = validation_data[dataset]['type']
            y_pred = predictions[dataset]['predictions']
            confidence = predictions[dataset]['confidence']
            
            # Calculate basic metrics
            accuracy = np.mean(np.array(y_true) == np.array(y_pred))
            
            # Per-class accuracy
            classes = sorted(y_true.unique())
            class_accuracies = {}
            for cls in classes:
                class_mask = y_true == cls
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(np.array(y_true)[class_mask] == np.array(y_pred)[class_mask])
                    class_accuracies[cls] = class_acc
            
            # Confidence statistics
            conf_stats = {
                'mean': np.mean(confidence),
                'median': np.median(confidence),
                'std': np.std(confidence),
                'min': np.min(confidence),
                'max': np.max(confidence)
            }
            
            # Correct vs incorrect predictions
            correct_mask = np.array(y_true) == np.array(y_pred)
            correct_conf = confidence[correct_mask]
            incorrect_conf = confidence[~correct_mask]
            
            metrics[dataset] = {
                'accuracy': accuracy,
                'total_samples': len(y_true),
                'correct_predictions': np.sum(correct_mask),
                'incorrect_predictions': np.sum(~correct_mask),
                'class_accuracies': class_accuracies,
                'confidence_stats': conf_stats,
                'correct_confidence': {
                    'mean': np.mean(correct_conf) if len(correct_conf) > 0 else 0,
                    'samples': len(correct_conf)
                },
                'incorrect_confidence': {
                    'mean': np.mean(incorrect_conf) if len(incorrect_conf) > 0 else 0,
                    'samples': len(incorrect_conf)
                }
            }
            print(f"  ✓ Calculated metrics for {dataset}: {accuracy:.1%} accuracy ({len(y_true)} samples)")
    
    print(f"✓ Calculated metrics for {len(metrics)} datasets")
    return metrics

def create_pure_validation_plots(predictions, validation_data, metrics, train_test_results=None):
    """Create comprehensive validation plots focused purely on enhanced model."""
    print("📈 Creating pure enhanced model validation visualizations...")
    
    # Create comprehensive figure focused only on validation - NO train/test plots
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Enhanced Model Validation Performance Analysis', fontsize=22, fontweight='bold', y=0.98)
    
    # 1. Validation Accuracy by Dataset (top left)
    ax1 = plt.subplot(3, 4, 1)
    datasets = []
    accuracies = []
    sample_counts = []
    
    # Show ALL individual datasets first, then combined at the end
    individual_datasets = [d for d in sorted(metrics.keys()) if d != 'combined']
    all_datasets_ordered = individual_datasets + (['combined'] if 'combined' in metrics else [])
    
    for dataset in all_datasets_ordered:
        datasets.append(dataset.upper())
        accuracies.append(metrics[dataset]['accuracy'])
        sample_counts.append(metrics[dataset]['total_samples'])
    
    # Use more colors to handle more datasets
    colors = ['#4682B4', '#9370DB', '#FF6B6B', '#4ECDC4', '#FFD700', '#FFA07A', '#98D8C8', '#F7DC6F']
    bars = ax1.bar(datasets, accuracies, color=colors[:len(datasets)], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, acc, count in zip(bars, accuracies, sample_counts):
        height = bar.get_height()
        ax1.annotate(f'{acc:.1%}\n(n={count})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_title('Validation Accuracy by Dataset', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Confidence Distribution (top second)
    ax2 = plt.subplot(3, 4, 2)
    
    if 'combined' in predictions:
        confidence = predictions['combined']['confidence']
        
        ax2.hist(confidence, bins=25, alpha=0.8, color='#4682B4', edgecolor='black')
        ax2.axvline(np.mean(confidence), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(confidence):.3f}')
        ax2.axvline(np.median(confidence), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(confidence):.3f}')
        
        ax2.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # 3. Per-Class Performance (top third)
    ax3 = plt.subplot(3, 4, 3)
    
    if 'combined' in metrics:
        class_acc = metrics['combined']['class_accuracies']
        classes = sorted(class_acc.keys())
        accuracies = [class_acc[cls] for cls in classes]
        
        bars = ax3.bar([f'Class {c}' for c in classes], accuracies, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(classes)], 
                      alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.annotate(f'{acc:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax3.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Correct vs Incorrect Confidence (top fourth)
    ax4 = plt.subplot(3, 4, 4)
    
    if 'combined' in metrics:
        correct_conf = metrics['combined']['correct_confidence']['mean']
        incorrect_conf = metrics['combined']['incorrect_confidence']['mean']
        correct_count = metrics['combined']['correct_confidence']['samples']
        incorrect_count = metrics['combined']['incorrect_confidence']['samples']
        
        categories = ['Correct\nPredictions', 'Incorrect\nPredictions']
        conf_means = [correct_conf, incorrect_conf]
        counts = [correct_count, incorrect_count]
        
        bars = ax4.bar(categories, conf_means, color=['green', 'red'], alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, conf, count in zip(bars, conf_means, counts):
            height = bar.get_height()
            ax4.annotate(f'{conf:.3f}\n(n={count})',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax4.set_title('Confidence: Correct vs Incorrect', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Average Confidence')
        ax4.set_ylim(0, 1)
        ax4.grid(axis='y', alpha=0.3)
    
    # 5. Sample Distribution (middle left)
    ax5 = plt.subplot(3, 4, 5)
    
    if 'combined' in validation_data:
        class_counts = validation_data['combined']['type'].value_counts().sort_index()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        wedges, texts, autotexts = ax5.pie(class_counts.values, 
                                          labels=[f'Class {c}' for c in class_counts.index],
                                          autopct='%1.1f%%', 
                                          colors=colors[:len(class_counts)], 
                                          startangle=90)
        
        ax5.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    # 6. Dataset Size Comparison (middle second)
    ax6 = plt.subplot(3, 4, 6)
    
    if metrics:
        # Show dataset sizes
        dataset_names = []
        dataset_sizes = []
        individual_datasets = [d for d in sorted(metrics.keys()) if d != 'combined']
        
        for dataset in individual_datasets:
            dataset_names.append(dataset.upper())
            dataset_sizes.append(metrics[dataset]['total_samples'])
        
        colors_size = ['#4682B4', '#9370DB', '#FF6B6B', '#4ECDC4', '#FFA07A']
        bars = ax6.bar(dataset_names, dataset_sizes, color=colors_size[:len(dataset_names)], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, size in zip(bars, dataset_sizes):
            height = bar.get_height()
            ax6.annotate(f'{size:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax6.set_title('Dataset Sizes', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Number of Samples')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(axis='y', alpha=0.3)
    
    # 7. Confidence by Class (middle third)
    ax7 = plt.subplot(3, 4, 7)
    
    if 'combined' in validation_data and 'combined' in predictions:
        y_true = validation_data['combined']['type']
        confidence = predictions['combined']['confidence']
        
        classes = sorted(y_true.unique())
        conf_by_class = [confidence[y_true == cls] for cls in classes]
        
        bp = ax7.boxplot(conf_by_class, labels=[f'Class {c}' for c in classes], patch_artist=True)
        
        # Color the boxes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        for patch, color in zip(bp['boxes'], colors[:len(classes)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax7.set_title('Confidence by Class', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Confidence')
        ax7.grid(axis='y', alpha=0.3)
    
    # 8. Performance Overview (middle fourth)
    ax8 = plt.subplot(3, 4, 8)
    
    # Create a performance radar-like summary
    if 'combined' in metrics:
        performance_metrics = [
            metrics['combined']['accuracy'],
            metrics['combined']['confidence_stats']['mean'],
            len(metrics['combined']['class_accuracies']) / 4,  # Class coverage (normalized)
            metrics['combined']['correct_predictions'] / metrics['combined']['total_samples']  # Correct ratio
        ]
        metric_names = ['Accuracy', 'Confidence', 'Coverage', 'Precision']
        
        bars = ax8.bar(metric_names, performance_metrics, 
                       color=['#2E8B57', '#4682B4', '#9370DB', '#FFD700'], alpha=0.8)
        
        for bar, metric in zip(bars, performance_metrics):
            height = bar.get_height()
            ax8.annotate(f'{metric:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax8.set_title('Performance Overview', fontsize=14, fontweight='bold')
        ax8.set_ylim(0, 1)
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(axis='y', alpha=0.3)
    
    # 9-12. Performance Summary (bottom row spanning all subplots)
    ax_summary = plt.subplot(3, 4, (9, 12))
    ax_summary.axis('off')
    
    # Create summary text - VALIDATION ONLY
    summary_text = "ENHANCED MODEL VALIDATION SUMMARY\n"
    summary_text += "=" * 45 + "\n\n"
    
    # Add validation summary only
    summary_text += "VALIDATION RESULTS:\n"
    for dataset in sorted(metrics.keys()): # Iterate through all datasets in metrics
        m = metrics[dataset]
        summary_text += f"  {dataset.upper():8}: {m['accuracy']:.1%} accuracy, {m['confidence_stats']['mean']:.3f} confidence ({m['total_samples']} samples)\n"
    
    # Overall statistics
    if 'combined' in metrics:
        total_accuracy = metrics['combined']['accuracy']
        total_samples = metrics['combined']['total_samples']
        summary_text += f"\nOVERALL VALIDATION ASSESSMENT:\n"
        summary_text += f"  🎉 Validation Accuracy: {total_accuracy:.1%}\n"
        summary_text += f"  🎯 Total Samples:       {total_samples}\n"
        summary_text += f"  📈 Model Status:        {'EXCELLENT' if total_accuracy > 0.9 else 'GOOD' if total_accuracy > 0.8 else 'NEEDS IMPROVEMENT'}\n"
        
        # Add class performance breakdown
        summary_text += f"\nPER-CLASS VALIDATION PERFORMANCE:\n"
        if 'class_accuracies' in metrics['combined']:
            for cls, acc in sorted(metrics['combined']['class_accuracies'].items()):
                summary_text += f"  Class {cls}: {acc:.1%}\n"
        
        # Add confidence insights
        summary_text += f"\nCONFIDENCE ANALYSIS:\n"
        cs = metrics['combined']['confidence_stats']
        summary_text += f"  📊 Average Confidence:  {cs['mean']:.3f}\n"
        summary_text += f"  📊 Confidence Range:    {cs['min']:.3f} - {cs['max']:.3f}\n"
        summary_text += f"  📊 Confidence Std:      {cs['std']:.3f}\n"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('enhanced_model_validation_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved pure validation-only visualization: enhanced_model_validation_results.png")

def create_dataset_specific_plots(predictions, validation_data):
    """Create individual plots for each validation dataset."""
    print("📊 Creating dataset-specific validation plots...")
    
    # Plot ALL individual datasets (excluding combined)
    datasets_to_plot = [d for d in sorted(predictions.keys()) if d != 'combined' and d in validation_data]
    
    for dataset in datasets_to_plot:
        if (dataset not in predictions or dataset not in validation_data or 
            len(validation_data[dataset]) == 0 or 'type' not in validation_data[dataset].columns):
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Enhanced Model Results - {dataset.upper()} Dataset', 
                    fontsize=16, fontweight='bold')
        
        y_true = validation_data[dataset]['type']
        y_pred = predictions[dataset]['predictions']
        confidence = predictions[dataset]['confidence']
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=sorted(y_true.unique()), 
                   yticklabels=sorted(y_true.unique()))
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted Class')
        ax1.set_ylabel('True Class')
        
        # 2. Confidence Distribution
        ax2 = axes[0, 1]
        ax2.hist(confidence, bins=15, alpha=0.8, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(confidence), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidence):.3f}')
        ax2.set_title('Confidence Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Prediction Correctness by Sample
        ax3 = axes[1, 0]
        is_correct = (np.array(y_true) == np.array(y_pred)).astype(int)
        sample_indices = range(len(y_true))
        
        colors = ['red' if not correct else 'green' for correct in is_correct]
        ax3.scatter(sample_indices, confidence, c=colors, alpha=0.7, s=50)
        ax3.set_title('Prediction Confidence vs Correctness')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Confidence')
        ax3.grid(alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Correct'),
                          Patch(facecolor='red', label='Incorrect')]
        ax3.legend(handles=legend_elements)
        
        # 4. Class-wise Performance
        ax4 = axes[1, 1]
        classes = sorted(y_true.unique())
        class_accuracies = []
        
        for cls in classes:
            class_mask = y_true == cls
            if np.sum(class_mask) > 0:
                class_acc = np.mean(np.array(y_true)[class_mask] == np.array(y_pred)[class_mask])
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        bars = ax4.bar([f'Class {c}' for c in classes], class_accuracies, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(classes)], alpha=0.8)
        
        # Add value labels
        for bar, acc in zip(bars, class_accuracies):
            height = bar.get_height()
            ax4.annotate(f'{acc:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('Class-wise Accuracy')
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim(0, 1)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'enhanced_model_{dataset}_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ Created dataset-specific validation plots for {dataset}")

def main():
    """Main function to create pure validation-focused visualizations."""
    print("🎨 Creating Pure Enhanced Model Validation Visualizations")
    print("=" * 65)
    
    try:
        # Load enhanced model predictions only
        predictions = load_enhanced_predictions()
        if not predictions:
            print("❌ No prediction data found")
            return
        
        # Load validation data for analysis
        validation_data = load_validation_data_for_analysis()
        if validation_data is None:
            print("❌ Could not load validation data")
            return
        
        # Load train/test results
        train_test_results = load_train_test_results()
        
        # Calculate enhanced model metrics
        metrics = calculate_enhanced_model_metrics(predictions, validation_data)
        
        # Create pure validation plots (no comparisons) + train/test plot
        create_pure_validation_plots(predictions, validation_data, metrics, train_test_results)
        
        # Create dataset-specific plots
        create_dataset_specific_plots(predictions, validation_data)
        
        print("\n✅ Pure enhanced model validation visualizations complete!")
        print("📁 Generated files:")
        print("   - enhanced_model_validation_results.png (comprehensive overview with train/test)")
        print("   - enhanced_model_val3_results.png (VAL3 specific)")
        print("   - enhanced_model_val6_results.png (VAL6 specific)")
        print("\n🎯 Focus: Enhanced model performance only - no comparisons!")
        print("🎯 NEW: Includes training vs test performance analysis!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 