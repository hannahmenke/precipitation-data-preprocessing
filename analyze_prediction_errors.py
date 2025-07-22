#!/usr/bin/env python3
"""
Analyze which features are leading to wrong predictions for each class.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix
from scipy import stats

def load_model_and_analyze_validation():
    """Load model components and validation data for analysis."""
    models_dir = Path("models")
    
    # Load model components
    model = joblib.load(models_dir / "latest_model.joblib")
    scaler = joblib.load(models_dir / "latest_scaler.joblib")
    feature_selector = joblib.load(models_dir / "latest_feature_selector.joblib")
    feature_names = joblib.load(models_dir / "latest_feature_names.joblib")
    
    # Load and preprocess validation data
    val3 = pd.read_excel('data_for_classification/val3.xlsx')
    val6 = pd.read_excel('data_for_classification/val6.xlsx')
    val_data = pd.concat([val3, val6], ignore_index=True)
    
    print(f"✓ Loaded validation data: {val_data.shape}")
    print(f"✓ Model features: {feature_names}")
    
    return model, scaler, feature_selector, feature_names, val_data

def preprocess_validation_data(val_data, scaler, feature_selector):
    """Preprocess validation data using the same pipeline as training."""
    # Get the original 17 features that the scaler expects
    original_features = [
        'Area', 'Major', 'Minor', 'Eccentricity', 'ConvexArea', 'Circularity', 
        'Extent', 'Perimeter', 'MeanIntensity', 'Gray_var', 'Gray_skew', 
        'Gray_skew_abs', 'Gray_kur', 'dis', 'dis_normal', 'Major_Minor_ratio', 
        'normalized_local_mean_gray'
    ]
    
    val_data_clean = val_data.dropna(subset=['type']).copy()
    X_raw = val_data_clean[original_features].copy()
    y_true = val_data_clean['type'].copy()
    
    # Handle missing values
    for col in X_raw.columns:
        if X_raw[col].isnull().sum() > 0:
            X_raw[col].fillna(X_raw[col].median(), inplace=True)
    
    # Apply scaling
    X_scaled = scaler.transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=original_features, index=X_raw.index)
    
    # Apply feature selection
    X_selected = feature_selector.transform(X_scaled_df)
    
    return X_selected, X_scaled_df, y_true, val_data_clean.index

def analyze_feature_distributions_by_prediction_outcome(X_scaled_df, y_true, y_pred, feature_names):
    """Analyze feature distributions for correct vs incorrect predictions by class."""
    
    # Create results dataframe
    results = []
    
    # Get predictions and correctness
    correct_mask = (y_true == y_pred)
    
    print(f"\n🔍 FEATURE ANALYSIS BY PREDICTION OUTCOME")
    print("="*60)
    
    for true_class in sorted(y_true.unique()):
        class_mask = (y_true == true_class)
        class_correct = correct_mask & class_mask
        class_wrong = (~correct_mask) & class_mask
        
        n_correct = class_correct.sum()
        n_wrong = class_wrong.sum()
        n_total = class_mask.sum()
        
        print(f"\nClass {true_class}:")
        print(f"  Correct predictions: {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")
        print(f"  Wrong predictions: {n_wrong}/{n_total} ({n_wrong/n_total*100:.1f}%)")
        
        if n_correct > 0 and n_wrong > 0:
            print(f"  Analyzing feature differences...")
            
            # Compare feature distributions for this class
            for feature in X_scaled_df.columns:
                correct_values = X_scaled_df.loc[class_correct, feature]
                wrong_values = X_scaled_df.loc[class_wrong, feature]
                
                # Statistical test for difference
                if len(correct_values) > 1 and len(wrong_values) > 1:
                    statistic, p_value = stats.mannwhitneyu(
                        correct_values, wrong_values, alternative='two-sided'
                    )
                    
                    # Effect size (difference in means)
                    effect_size = abs(correct_values.mean() - wrong_values.mean())
                    
                    results.append({
                        'class': true_class,
                        'feature': feature,
                        'correct_mean': correct_values.mean(),
                        'wrong_mean': wrong_values.mean(),
                        'effect_size': effect_size,
                        'p_value': p_value,
                        'n_correct': len(correct_values),
                        'n_wrong': len(wrong_values)
                    })
    
    return pd.DataFrame(results)

def analyze_confusion_patterns(y_true, y_pred, X_scaled_df):
    """Analyze which features distinguish between commonly confused classes."""
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(y_true.unique())
    
    print(f"\n🎯 CONFUSION PATTERN ANALYSIS")
    print("="*60)
    
    confusion_results = []
    
    # Find the most common confusions
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            if i != j and cm[i, j] > 0:  # Wrong predictions
                confusion_count = cm[i, j]
                
                # Get samples of this confusion
                confusion_mask = (y_true == true_class) & (y_pred == pred_class)
                correct_mask = (y_true == true_class) & (y_pred == true_class)
                
                if confusion_mask.sum() > 0 and correct_mask.sum() > 0:
                    print(f"\nClass {true_class} predicted as Class {pred_class}: {confusion_count} cases")
                    
                    # Compare features for this specific confusion
                    for feature in X_scaled_df.columns:
                        confused_values = X_scaled_df.loc[confusion_mask, feature]
                        correct_values = X_scaled_df.loc[correct_mask, feature]
                        
                        if len(confused_values) > 0 and len(correct_values) > 1:
                            # Effect size
                            effect_size = abs(confused_values.mean() - correct_values.mean())
                            
                            confusion_results.append({
                                'true_class': true_class,
                                'pred_class': pred_class,
                                'feature': feature,
                                'correct_mean': correct_values.mean(),
                                'confused_mean': confused_values.mean(),
                                'effect_size': effect_size,
                                'confusion_count': confusion_count
                            })
    
    return pd.DataFrame(confusion_results)

def create_feature_analysis_plots(feature_results, confusion_results, feature_names):
    """Create visualizations of feature analysis results."""
    
    # Plot 1: Top problematic features by class
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Analysis: What Causes Wrong Predictions?', fontsize=16, fontweight='bold')
    
    # 1. Feature importance for wrong predictions by class
    if not feature_results.empty:
        # Get top problematic features for each class
        top_features_by_class = feature_results.groupby('class').apply(
            lambda x: x.nlargest(5, 'effect_size')
        ).reset_index(drop=True)
        
        # Plot heatmap of effect sizes
        pivot_data = feature_results.pivot_table(
            index='feature', columns='class', values='effect_size', fill_value=0
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='Reds', ax=axes[0,0])
        axes[0,0].set_title('Feature Effect Sizes by Class\n(Higher = More Problematic)', fontweight='bold')
        axes[0,0].set_xlabel('True Class')
        axes[0,0].set_ylabel('Feature')
    
    # 2. P-values for statistical significance
    if not feature_results.empty:
        # Show significance of differences
        pivot_pvals = feature_results.pivot_table(
            index='feature', columns='class', values='p_value', fill_value=1
        )
        
        # Convert to -log10(p) for better visualization
        log_pvals = -np.log10(pivot_pvals.clip(lower=1e-10))
        
        sns.heatmap(log_pvals, annot=True, fmt='.1f', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title('Statistical Significance\n(-log10(p-value), >1.3 = significant)', fontweight='bold')
        axes[0,1].set_xlabel('True Class')
        axes[0,1].set_ylabel('Feature')
    
    # 3. Most problematic features overall
    if not feature_results.empty:
        overall_problems = feature_results.groupby('feature')['effect_size'].mean().sort_values(ascending=True)
        
        y_pos = np.arange(len(overall_problems))
        axes[1,0].barh(y_pos, overall_problems.values, color='lightcoral', alpha=0.8)
        axes[1,0].set_yticks(y_pos)
        axes[1,0].set_yticklabels(overall_problems.index)
        axes[1,0].set_xlabel('Average Effect Size')
        axes[1,0].set_title('Most Problematic Features Overall', fontweight='bold')
        axes[1,0].grid(axis='x', alpha=0.3)
    
    # 4. Confusion pattern analysis
    if not confusion_results.empty:
        # Show top confusions
        top_confusions = confusion_results.groupby(['true_class', 'pred_class']).apply(
            lambda x: x.nlargest(3, 'effect_size')
        ).reset_index(drop=True)
        
        # Create a summary plot
        confusion_summary = confusion_results.groupby(['true_class', 'pred_class'])['effect_size'].mean().reset_index()
        
        if len(confusion_summary) > 0:
            pivot_confusion = confusion_summary.pivot(
                index='true_class', columns='pred_class', values='effect_size'
            ).fillna(0)
            
            sns.heatmap(pivot_confusion, annot=True, fmt='.3f', cmap='Oranges', ax=axes[1,1])
            axes[1,1].set_title('Class Confusion Feature Effects', fontweight='bold')
            axes[1,1].set_xlabel('Predicted Class')
            axes[1,1].set_ylabel('True Class')
    
    plt.tight_layout()
    plt.savefig('feature_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved feature analysis plot to: feature_error_analysis.png")

def print_detailed_analysis(feature_results, confusion_results):
    """Print detailed analysis results."""
    
    print(f"\n📋 DETAILED FEATURE ANALYSIS RESULTS")
    print("="*60)
    
    if not feature_results.empty:
        print(f"\n🔴 TOP PROBLEMATIC FEATURES BY CLASS:")
        print("-"*40)
        
        for class_num in sorted(feature_results['class'].unique()):
            class_data = feature_results[feature_results['class'] == class_num]
            top_features = class_data.nlargest(3, 'effect_size')
            
            print(f"\nClass {class_num} (most problematic features):")
            for _, row in top_features.iterrows():
                significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                print(f"  {row['feature']:25} | Effect: {row['effect_size']:.3f} | p: {row['p_value']:.3f} {significance}")
                print(f"    {'':25} | Correct mean: {row['correct_mean']:6.3f} | Wrong mean: {row['wrong_mean']:6.3f}")
    
    if not confusion_results.empty:
        print(f"\n🎯 MOST COMMON CONFUSIONS:")
        print("-"*40)
        
        # Group by confusion pair and show top features
        confusion_groups = confusion_results.groupby(['true_class', 'pred_class'])
        
        for (true_cls, pred_cls), group in confusion_groups:
            confusion_count = group['confusion_count'].iloc[0]
            top_confusion_features = group.nlargest(2, 'effect_size')
            
            print(f"\nClass {true_cls} → Class {pred_cls} ({confusion_count} cases):")
            for _, row in top_confusion_features.iterrows():
                print(f"  {row['feature']:25} | Effect: {row['effect_size']:.3f}")
                print(f"    {'':25} | Correct: {row['correct_mean']:6.3f} | Confused: {row['confused_mean']:6.3f}")

def main():
    """Main function to analyze prediction errors."""
    print("🔍 Analyzing Prediction Errors by Feature")
    print("=" * 50)
    
    try:
        # Load model and data
        model, scaler, feature_selector, feature_names, val_data = load_model_and_analyze_validation()
        
        # Preprocess data
        X_selected, X_scaled_df, y_true, indices = preprocess_validation_data(val_data, scaler, feature_selector)
        
        # Make predictions
        y_pred = model.predict(X_selected)
        
        # Convert predictions back to original labels (assuming 1,2,3,4 mapping)
        label_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
        y_pred_labels = [label_mapping[pred] for pred in y_pred]
        
        print(f"✓ Made predictions on {len(y_pred)} samples")
        print(f"✓ Overall accuracy: {(y_true == y_pred_labels).mean():.1%}")
        
        # Analyze feature distributions
        feature_results = analyze_feature_distributions_by_prediction_outcome(
            X_scaled_df, y_true, y_pred_labels, feature_names
        )
        
        # Analyze confusion patterns
        confusion_results = analyze_confusion_patterns(y_true, y_pred_labels, X_scaled_df)
        
        # Create visualizations
        create_feature_analysis_plots(feature_results, confusion_results, feature_names)
        
        # Print detailed results
        print_detailed_analysis(feature_results, confusion_results)
        
        print("\n✅ Feature error analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 