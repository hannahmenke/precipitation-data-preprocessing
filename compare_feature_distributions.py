#!/usr/bin/env python3
"""
Compare feature distributions across training, test, and validation datasets.

This script creates comprehensive visualizations to understand domain shift
and distribution differences between datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_datasets():
    """
    Load and preprocess all datasets (training, test, validation).
    
    Returns:
        Dictionary with preprocessed datasets
    """
    print("📂 Loading datasets...")
    
    # Load datasets
    train_df = pd.read_excel('training_data/train_improved.xlsx')
    
    # Load test data from the same training file (it was split)
    # We'll use the original combined data to extract test portion
    try:
        # Try to load original combined data to extract test portion
        combined_df = pd.read_excel('data_for_classification/combined_features.xlsx')
        
        # Get unique source files from training
        train_files = set(train_df['source_file'].unique()) if 'source_file' in train_df.columns else set()
        
        # Test data is what's not in training (if source_file exists)
        if 'source_file' in combined_df.columns and len(train_files) > 0:
            test_df = combined_df[~combined_df['source_file'].isin(train_files)].copy()
        else:
            # If no source_file info, we'll use a portion of combined data
            n_train = len(train_df)
            test_df = combined_df.iloc[n_train:].copy()
            
    except FileNotFoundError:
        print("⚠️  Combined features file not found, using validation as test comparison")
        test_df = None
    
    # Load validation data
    val_df = pd.read_excel('validation_data_improved.xlsx')
    
    # Define feature columns (exclude metadata)
    exclude_cols = ['NO.', 'ID', 'type', 'source_file', 'Centroid', 'BoundingBox', 'WeightedCentroid']
    
    def get_numeric_features(df):
        return [col for col in df.columns 
                if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    # Get common features across all datasets
    train_features = set(get_numeric_features(train_df))
    val_features = set(get_numeric_features(val_df))
    
    if test_df is not None:
        test_features = set(get_numeric_features(test_df))
        common_features = list(train_features & test_features & val_features)
    else:
        common_features = list(train_features & val_features)
        test_features = set()
    
    print(f"✓ Training features: {len(train_features)}")
    if test_df is not None:
        print(f"✓ Test features: {len(test_features)}")
    print(f"✓ Validation features: {len(val_features)}")
    print(f"✓ Common features: {len(common_features)}")
    
    # Prepare datasets with common features
    datasets = {
        'train': {
            'data': train_df[common_features + ['type']].copy(),
            'name': 'Training',
            'color': 'blue',
            'alpha': 0.6
        },
        'val': {
            'data': val_df[common_features + ['type']].copy(),
            'name': 'Validation', 
            'color': 'red',
            'alpha': 0.6
        }
    }
    
    if test_df is not None and len(test_df) > 10:  # Only include if we have substantial test data
        datasets['test'] = {
            'data': test_df[common_features + ['type']].copy(),
            'name': 'Test',
            'color': 'green', 
            'alpha': 0.6
        }
    
    # Print dataset sizes
    for key, dataset in datasets.items():
        print(f"✓ {dataset['name']}: {len(dataset['data'])} samples")
    
    return datasets, common_features

def calculate_distribution_statistics(datasets, features):
    """
    Calculate statistical comparisons between distributions.
    
    Args:
        datasets: Dictionary of datasets
        features: List of feature names
        
    Returns:
        Dictionary with statistical test results
    """
    print("📊 Calculating distribution statistics...")
    
    results = {}
    dataset_names = list(datasets.keys())
    
    for feature in features:
        results[feature] = {}
        
        # Get feature data for each dataset
        feature_data = {}
        for name, dataset in datasets.items():
            data = dataset['data'][feature].dropna()
            feature_data[name] = data
            
        # Calculate pairwise statistical tests
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                pair_key = f"{name1}_vs_{name2}"
                
                data1 = feature_data[name1]
                data2 = feature_data[name2]
                
                # Kolmogorov-Smirnov test (tests if distributions are different)
                ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)
                
                # Mann-Whitney U test (tests if medians are different)
                mw_stat, mw_pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                # Effect size (Cohen's d equivalent for non-parametric)
                pooled_std = np.sqrt(((len(data1)-1)*np.var(data1) + (len(data2)-1)*np.var(data2)) / 
                                   (len(data1)+len(data2)-2))
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                
                results[feature][pair_key] = {
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'mw_statistic': mw_stat,
                    'mw_pvalue': mw_pvalue,
                    'cohens_d': cohens_d,
                    'mean_diff': np.mean(data1) - np.mean(data2),
                    'median_diff': np.median(data1) - np.median(data2)
                }
    
    return results

def create_feature_distribution_plots(datasets, features, stats_results, max_features_per_plot=6):
    """
    Create comprehensive feature distribution comparison plots.
    
    Args:
        datasets: Dictionary of datasets
        features: List of feature names
        stats_results: Statistical test results
        max_features_per_plot: Maximum number of features per plot
    """
    print("📈 Creating distribution comparison plots...")
    
    # Split features into chunks for better visualization
    feature_chunks = [features[i:i+max_features_per_plot] 
                     for i in range(0, len(features), max_features_per_plot)]
    
    for chunk_idx, feature_chunk in enumerate(feature_chunks):
        n_features = len(feature_chunk)
        fig, axes = plt.subplots(2, n_features, figsize=(4*n_features, 10))
        
        if n_features == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Feature Distribution Comparison (Set {chunk_idx + 1})', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        for feat_idx, feature in enumerate(feature_chunk):
            # Top row: Histograms with density curves
            ax_hist = axes[0, feat_idx]
            
            for dataset_name, dataset in datasets.items():
                data = dataset['data'][feature].dropna()
                
                # Histogram
                ax_hist.hist(data, bins=30, alpha=dataset['alpha'], 
                           label=f"{dataset['name']} (n={len(data)})",
                           color=dataset['color'], density=True)
                
                # KDE curve
                if len(data) > 5:  # Only plot KDE if we have enough data
                    kde_x = np.linspace(data.min(), data.max(), 100)
                    kde = stats.gaussian_kde(data)
                    ax_hist.plot(kde_x, kde(kde_x), color=dataset['color'], 
                               linewidth=2, alpha=0.8)
            
            ax_hist.set_title(f'{feature}\nDistribution Comparison', fontweight='bold')
            ax_hist.set_xlabel('Value')
            ax_hist.set_ylabel('Density')
            ax_hist.legend()
            ax_hist.grid(alpha=0.3)
            
            # Bottom row: Box plots
            ax_box = axes[1, feat_idx]
            
            box_data = []
            box_labels = []
            box_colors = []
            
            for dataset_name, dataset in datasets.items():
                data = dataset['data'][feature].dropna()
                box_data.append(data)
                box_labels.append(f"{dataset['name']}\n(n={len(data)})")
                box_colors.append(dataset['color'])
            
            bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax_box.set_title(f'{feature}\nBox Plot Comparison', fontweight='bold')
            ax_box.set_ylabel('Value')
            ax_box.grid(axis='y', alpha=0.3)
            ax_box.tick_params(axis='x', rotation=45)
            
            # Add statistical significance annotations
            dataset_pairs = list(stats_results[feature].keys())
            if dataset_pairs:
                y_max = ax_box.get_ylim()[1]
                for i, pair in enumerate(dataset_pairs):
                    ks_p = stats_results[feature][pair]['ks_pvalue']
                    if ks_p < 0.001:
                        sig_text = "***"
                    elif ks_p < 0.01:
                        sig_text = "**"
                    elif ks_p < 0.05:
                        sig_text = "*"
                    else:
                        sig_text = "ns"
                    
                    ax_box.text(0.02, 0.98 - i*0.1, f"{pair}: {sig_text}", 
                              transform=ax_box.transAxes, fontsize=8,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'feature_distributions_comparison_{chunk_idx + 1}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"✓ Created {len(feature_chunks)} distribution comparison plots")

def create_statistical_summary_plots(stats_results, datasets):
    """
    Create summary plots of statistical test results.
    
    Args:
        stats_results: Statistical test results
        datasets: Dictionary of datasets
    """
    print("📊 Creating statistical summary plots...")
    
    # Prepare data for summary plots
    features = list(stats_results.keys())
    dataset_pairs = list(stats_results[features[0]].keys()) if features else []
    
    if not dataset_pairs:
        print("⚠️  No dataset pairs to compare")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Comparison Summary Across All Features', 
                fontsize=16, fontweight='bold')
    
    # 1. P-value heatmap (KS test)
    ks_pvalues = np.zeros((len(features), len(dataset_pairs)))
    for i, feature in enumerate(features):
        for j, pair in enumerate(dataset_pairs):
            ks_pvalues[i, j] = stats_results[feature][pair]['ks_pvalue']
    
    im1 = axes[0, 0].imshow(ks_pvalues, aspect='auto', cmap='RdYlBu_r')
    axes[0, 0].set_title('KS Test P-values\n(Red = Significant Difference)', fontweight='bold')
    axes[0, 0].set_xlabel('Dataset Pairs')
    axes[0, 0].set_ylabel('Features')
    axes[0, 0].set_xticks(range(len(dataset_pairs)))
    axes[0, 0].set_xticklabels(dataset_pairs, rotation=45)
    axes[0, 0].set_yticks(range(len(features)))
    axes[0, 0].set_yticklabels(features, fontsize=8)
    
    # Add significance threshold lines
    axes[0, 0].axhline(y=-0.5, color='white', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(x=-0.5, color='white', linestyle='--', alpha=0.7)
    
    plt.colorbar(im1, ax=axes[0, 0], label='P-value')
    
    # 2. Effect sizes (Cohen's d)
    cohens_d = np.zeros((len(features), len(dataset_pairs)))
    for i, feature in enumerate(features):
        for j, pair in enumerate(dataset_pairs):
            cohens_d[i, j] = abs(stats_results[feature][pair]['cohens_d'])
    
    im2 = axes[0, 1].imshow(cohens_d, aspect='auto', cmap='Reds')
    axes[0, 1].set_title('Effect Sizes (|Cohen\'s d|)\n(Red = Large Effect)', fontweight='bold')
    axes[0, 1].set_xlabel('Dataset Pairs')
    axes[0, 1].set_ylabel('Features')
    axes[0, 1].set_xticks(range(len(dataset_pairs)))
    axes[0, 1].set_xticklabels(dataset_pairs, rotation=45)
    axes[0, 1].set_yticks(range(len(features)))
    axes[0, 1].set_yticklabels(features, fontsize=8)
    
    plt.colorbar(im2, ax=axes[0, 1], label='|Cohen\'s d|')
    
    # 3. Distribution of p-values
    all_ks_pvalues = []
    all_mw_pvalues = []
    
    for feature in features:
        for pair in dataset_pairs:
            all_ks_pvalues.append(stats_results[feature][pair]['ks_pvalue'])
            all_mw_pvalues.append(stats_results[feature][pair]['mw_pvalue'])
    
    axes[1, 0].hist(all_ks_pvalues, bins=20, alpha=0.7, label='KS Test', color='blue')
    axes[1, 0].hist(all_mw_pvalues, bins=20, alpha=0.7, label='Mann-Whitney U', color='red')
    axes[1, 0].axvline(0.05, color='black', linestyle='--', label='α = 0.05')
    axes[1, 0].set_title('Distribution of P-values', fontweight='bold')
    axes[1, 0].set_xlabel('P-value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Summary statistics table
    axes[1, 1].axis('off')
    
    # Calculate summary statistics
    significant_features = []
    for feature in features:
        for pair in dataset_pairs:
            if stats_results[feature][pair]['ks_pvalue'] < 0.05:
                significant_features.append(f"{feature} ({pair})")
    
    n_significant = sum(1 for p in all_ks_pvalues if p < 0.05)
    n_total = len(all_ks_pvalues)
    
    summary_text = f"""
    STATISTICAL SUMMARY
    ═══════════════════
    
    Total Comparisons: {n_total}
    Significant Differences: {n_significant} ({n_significant/n_total*100:.1f}%)
    
    Significance Threshold: p < 0.05
    
    Dataset Sizes:
    """
    
    for dataset_name, dataset in datasets.items():
        summary_text += f"    {dataset['name']}: {len(dataset['data'])} samples\n"
    
    summary_text += f"""
    
    Most Different Features (p < 0.001):
    """
    
    very_significant = [(f, p, stats_results[f][p]['ks_pvalue']) 
                       for f in features for p in dataset_pairs 
                       if stats_results[f][p]['ks_pvalue'] < 0.001]
    very_significant.sort(key=lambda x: x[2])
    
    for feature, pair, pval in very_significant[:10]:  # Top 10
        summary_text += f"    {feature} ({pair}): p={pval:.2e}\n"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Created statistical summary plots")

def print_detailed_analysis(stats_results, datasets):
    """
    Print detailed analysis of distribution differences.
    
    Args:
        stats_results: Statistical test results
        datasets: Dictionary of datasets
    """
    print("\n" + "="*80)
    print("📋 DETAILED STATISTICAL ANALYSIS")
    print("="*80)
    
    features = list(stats_results.keys())
    dataset_pairs = list(stats_results[features[0]].keys()) if features else []
    
    # Overall summary
    all_ks_pvalues = [stats_results[f][p]['ks_pvalue'] 
                     for f in features for p in dataset_pairs]
    
    n_significant = sum(1 for p in all_ks_pvalues if p < 0.05)
    n_total = len(all_ks_pvalues)
    
    print(f"\n🔍 OVERALL SUMMARY:")
    print(f"   Total feature-pair comparisons: {n_total}")
    print(f"   Significantly different distributions: {n_significant}/{n_total} ({n_significant/n_total*100:.1f}%)")
    
    # Most different features
    print(f"\n⚠️  MOST DIFFERENT FEATURES (p < 0.001):")
    very_significant = []
    for feature in features:
        for pair in dataset_pairs:
            pval = stats_results[feature][pair]['ks_pvalue']
            effect_size = abs(stats_results[feature][pair]['cohens_d'])
            if pval < 0.001:
                very_significant.append((feature, pair, pval, effect_size))
    
    very_significant.sort(key=lambda x: x[2])  # Sort by p-value
    
    for feature, pair, pval, effect_size in very_significant[:15]:  # Top 15
        print(f"   {feature:25} ({pair:15}): p={pval:.2e}, |d|={effect_size:.3f}")
    
    # Dataset pair analysis
    print(f"\n📊 ANALYSIS BY DATASET PAIR:")
    for pair in dataset_pairs:
        pair_significant = [f for f in features if stats_results[f][pair]['ks_pvalue'] < 0.05]
        total_features = len(features)
        
        print(f"\n   {pair.upper().replace('_', ' vs ')}:")
        print(f"     Features with different distributions: {len(pair_significant)}/{total_features} ({len(pair_significant)/total_features*100:.1f}%)")
        
        if pair_significant:
            # Show top 5 most different
            top_different = [(f, stats_results[f][pair]['ks_pvalue'], 
                            abs(stats_results[f][pair]['cohens_d'])) 
                           for f in pair_significant]
            top_different.sort(key=lambda x: x[1])  # Sort by p-value
            
            print(f"     Most different:")
            for feature, pval, effect_size in top_different[:5]:
                print(f"       {feature:20}: p={pval:.3f}, |d|={effect_size:.3f}")

def main():
    """
    Main function to compare feature distributions across datasets.
    """
    print("🔍 Feature Distribution Comparison Analysis")
    print("=" * 60)
    
    try:
        # Load and preprocess datasets
        datasets, common_features = load_and_preprocess_datasets()
        
        if len(common_features) == 0:
            print("❌ No common features found across datasets!")
            return
        
        print(f"\n✓ Found {len(common_features)} common features to analyze")
        
        # Calculate distribution statistics
        stats_results = calculate_distribution_statistics(datasets, common_features)
        
        # Create visualizations
        create_feature_distribution_plots(datasets, common_features, stats_results)
        create_statistical_summary_plots(stats_results, datasets)
        
        # Print detailed analysis
        print_detailed_analysis(stats_results, datasets)
        
        # Save detailed results
        results_summary = []
        for feature in common_features:
            for pair in stats_results[feature].keys():
                result = stats_results[feature][pair]
                results_summary.append({
                    'feature': feature,
                    'dataset_pair': pair,
                    'ks_pvalue': result['ks_pvalue'],
                    'mw_pvalue': result['mw_pvalue'],
                    'cohens_d': result['cohens_d'],
                    'mean_difference': result['mean_diff'],
                    'median_difference': result['median_diff'],
                    'significant_05': result['ks_pvalue'] < 0.05,
                    'significant_001': result['ks_pvalue'] < 0.001
                })
        
        results_df = pd.DataFrame(results_summary)
        results_df.to_excel('distribution_comparison_results.xlsx', index=False)
        
        print(f"\n✅ Analysis complete!")
        print(f"📁 Saved detailed results to: distribution_comparison_results.xlsx")
        print(f"📊 Generated plots:")
        print(f"   - feature_distributions_comparison_*.png")
        print(f"   - statistical_summary.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 