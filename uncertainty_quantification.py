#!/usr/bin/env python3
"""
Uncertainty Quantification and Out-of-Distribution Detection for XGBoost Predictions.

This script implements several techniques to detect when predictions are made on 
data that's outside the training distribution, helping to identify overconfident
but potentially incorrect predictions.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns

class UncertaintyQuantifier:
    """
    Class to quantify prediction uncertainty and detect out-of-distribution samples.
    """
    
    def __init__(self, contamination=0.1):
        """
        Initialize uncertainty quantification methods.
        
        Args:
            contamination: Expected proportion of outliers in the data
        """
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.elliptic_envelope = EllipticEnvelope(contamination=contamination, random_state=42)
        self.lof = LocalOutlierFactor(contamination=contamination, novelty=True)
        
        # For Mahalanobis distance
        self.train_mean = None
        self.train_cov_inv = None
        
        # Feature range tracking
        self.feature_ranges = {}
        self.feature_quantiles = {}
        
    def fit(self, X_train, feature_names):
        """
        Fit all uncertainty quantification methods on training data.
        
        Args:
            X_train: Training feature matrix
            feature_names: List of feature names
        """
        print("🔧 Fitting uncertainty quantification methods...")
        
        # Convert to DataFrame if needed
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train, columns=feature_names)
        
        # Fit outlier detection methods
        self.isolation_forest.fit(X_train)
        self.elliptic_envelope.fit(X_train)
        self.lof.fit(X_train)
        
        # Calculate statistics for Mahalanobis distance
        self.train_mean = np.mean(X_train, axis=0)
        train_cov = np.cov(X_train.T)
        
        # Add small regularization to avoid singularity
        regularization = 1e-6 * np.eye(train_cov.shape[0])
        train_cov_reg = train_cov + regularization
        self.train_cov_inv = np.linalg.inv(train_cov_reg)
        
        # Calculate feature ranges and quantiles
        for i, feature in enumerate(feature_names):
            feature_data = X_train.iloc[:, i] if isinstance(X_train, pd.DataFrame) else X_train[:, i]
            
            self.feature_ranges[feature] = {
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'mean': np.mean(feature_data),
                'std': np.std(feature_data)
            }
            
            self.feature_quantiles[feature] = {
                'q01': np.percentile(feature_data, 1),
                'q05': np.percentile(feature_data, 5),
                'q95': np.percentile(feature_data, 95),
                'q99': np.percentile(feature_data, 99)
            }
        
        print(f"✓ Fitted uncertainty methods on {len(X_train)} training samples")
        
    def detect_outliers(self, X_test, feature_names):
        """
        Detect outliers using multiple methods.
        
        Args:
            X_test: Test feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with outlier scores and binary predictions
        """
        # Convert to DataFrame if needed
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        results = {}
        
        # 1. Isolation Forest
        iso_scores = self.isolation_forest.decision_function(X_test)
        iso_outliers = self.isolation_forest.predict(X_test) == -1
        results['isolation_forest'] = {
            'scores': iso_scores,
            'outliers': iso_outliers,
            'outlier_rate': np.mean(iso_outliers)
        }
        
        # 2. Elliptic Envelope
        elliptic_scores = self.elliptic_envelope.decision_function(X_test)
        elliptic_outliers = self.elliptic_envelope.predict(X_test) == -1
        results['elliptic_envelope'] = {
            'scores': elliptic_scores,
            'outliers': elliptic_outliers,
            'outlier_rate': np.mean(elliptic_outliers)
        }
        
        # 3. Local Outlier Factor
        lof_scores = self.lof.decision_function(X_test)
        lof_outliers = self.lof.predict(X_test) == -1
        results['lof'] = {
            'scores': lof_scores,
            'outliers': lof_outliers,
            'outlier_rate': np.mean(lof_outliers)
        }
        
        # 4. Mahalanobis Distance
        mahal_distances = []
        for i in range(len(X_test)):
            sample = X_test.iloc[i] if isinstance(X_test, pd.DataFrame) else X_test[i]
            distance = mahalanobis(sample, self.train_mean, self.train_cov_inv)
            mahal_distances.append(distance)
        
        mahal_distances = np.array(mahal_distances)
        # Use chi-square distribution for outlier threshold
        threshold = stats.chi2.ppf(1 - self.contamination, df=len(feature_names))
        mahal_outliers = mahal_distances > threshold
        
        results['mahalanobis'] = {
            'scores': mahal_distances,
            'outliers': mahal_outliers,
            'outlier_rate': np.mean(mahal_outliers),
            'threshold': threshold
        }
        
        # 5. Feature Range Analysis
        feature_outliers = self.analyze_feature_ranges(X_test, feature_names)
        results['feature_ranges'] = feature_outliers
        
        return results
    
    def analyze_feature_ranges(self, X_test, feature_names):
        """
        Analyze which features are outside training ranges.
        
        Args:
            X_test: Test feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature-wise outlier analysis
        """
        # Convert to DataFrame if needed
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=feature_names)
            
        feature_analysis = {}
        
        for feature in feature_names:
            if feature in X_test.columns:
                feature_data = X_test[feature]
                ranges = self.feature_ranges[feature]
                quantiles = self.feature_quantiles[feature]
                
                # Check various types of outliers
                outside_range = (feature_data < ranges['min']) | (feature_data > ranges['max'])
                outside_q01_q99 = (feature_data < quantiles['q01']) | (feature_data > quantiles['q99'])
                outside_q05_q95 = (feature_data < quantiles['q05']) | (feature_data > quantiles['q95'])
                
                # Z-score based outliers
                z_scores = np.abs((feature_data - ranges['mean']) / ranges['std'])
                z_outliers_2 = z_scores > 2
                z_outliers_3 = z_scores > 3
                
                feature_analysis[feature] = {
                    'outside_training_range': outside_range,
                    'outside_q01_q99': outside_q01_q99,
                    'outside_q05_q95': outside_q05_q95,
                    'z_outliers_2std': z_outliers_2,
                    'z_outliers_3std': z_outliers_3,
                    'z_scores': z_scores,
                    'outlier_rate_range': np.mean(outside_range),
                    'outlier_rate_q99': np.mean(outside_q01_q99),
                    'outlier_rate_z3': np.mean(z_outliers_3)
                }
        
        return feature_analysis
    
    def calculate_uncertainty_scores(self, outlier_results):
        """
        Calculate combined uncertainty scores from multiple outlier detection methods.
        
        Args:
            outlier_results: Results from detect_outliers method
            
        Returns:
            Combined uncertainty scores and classifications
        """
        n_samples = len(outlier_results['isolation_forest']['scores'])
        
        # Normalize scores to [0, 1] where 1 = high uncertainty/outlier
        iso_scores_norm = 1 / (1 + np.exp(outlier_results['isolation_forest']['scores']))
        elliptic_scores_norm = 1 / (1 + np.exp(outlier_results['elliptic_envelope']['scores']))
        lof_scores_norm = 1 / (1 + np.exp(-outlier_results['lof']['scores']))
        
        # Normalize Mahalanobis distances
        mahal_scores = outlier_results['mahalanobis']['scores']
        mahal_scores_norm = mahal_scores / (np.max(mahal_scores) + 1e-6)
        
        # Combine scores (weighted average)
        combined_uncertainty = (
            0.3 * iso_scores_norm +
            0.3 * elliptic_scores_norm +
            0.2 * lof_scores_norm +
            0.2 * mahal_scores_norm
        )
        
        # Binary outlier classification (majority voting)
        outlier_votes = (
            outlier_results['isolation_forest']['outliers'].astype(int) +
            outlier_results['elliptic_envelope']['outliers'].astype(int) +
            outlier_results['lof']['outliers'].astype(int) +
            outlier_results['mahalanobis']['outliers'].astype(int)
        )
        
        is_outlier = outlier_votes >= 2  # Majority vote
        
        return {
            'uncertainty_scores': combined_uncertainty,
            'is_outlier': is_outlier,
            'outlier_votes': outlier_votes,
            'individual_scores': {
                'isolation_forest': iso_scores_norm,
                'elliptic_envelope': elliptic_scores_norm,
                'lof': lof_scores_norm,
                'mahalanobis': mahal_scores_norm
            }
        }

def adjust_prediction_confidence(original_confidence, uncertainty_scores, method='linear'):
    """
    Adjust prediction confidence based on uncertainty scores.
    
    Args:
        original_confidence: Original model confidence scores
        uncertainty_scores: Uncertainty scores from UncertaintyQuantifier
        method: Method for adjustment ('linear', 'exponential', 'threshold')
        
    Returns:
        Adjusted confidence scores
    """
    if method == 'linear':
        # Linear reduction based on uncertainty
        adjusted_confidence = original_confidence * (1 - uncertainty_scores)
        
    elif method == 'exponential':
        # Exponential reduction for high uncertainty
        adjusted_confidence = original_confidence * np.exp(-2 * uncertainty_scores)
        
    elif method == 'threshold':
        # Threshold-based reduction
        high_uncertainty_mask = uncertainty_scores > 0.5
        adjusted_confidence = original_confidence.copy()
        adjusted_confidence[high_uncertainty_mask] *= 0.5  # Reduce by 50% for high uncertainty
        
    else:
        raise ValueError(f"Unknown adjustment method: {method}")
    
    # Ensure confidence stays in [0, 1] range
    adjusted_confidence = np.clip(adjusted_confidence, 0, 1)
    
    return adjusted_confidence

def create_uncertainty_analysis_plots(uncertainty_results, original_confidence, adjusted_confidence, 
                                    y_true=None, y_pred=None, save_path="uncertainty_analysis.png"):
    """
    Create comprehensive plots for uncertainty analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Uncertainty Quantification and Confidence Adjustment Analysis', fontsize=16, fontweight='bold')
    
    uncertainty_scores = uncertainty_results['uncertainty_scores']
    is_outlier = uncertainty_results['is_outlier']
    
    # 1. Uncertainty Score Distribution
    axes[0,0].hist(uncertainty_scores, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0,0].axvline(np.mean(uncertainty_scores), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(uncertainty_scores):.3f}')
    axes[0,0].set_xlabel('Uncertainty Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Uncertainty Score Distribution')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    # 2. Original vs Adjusted Confidence
    axes[0,1].scatter(original_confidence, adjusted_confidence, alpha=0.6, 
                     c=uncertainty_scores, cmap='Reds', s=50)
    axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No Change')
    axes[0,1].set_xlabel('Original Confidence')
    axes[0,1].set_ylabel('Adjusted Confidence')
    axes[0,1].set_title('Confidence Adjustment')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    # Add colorbar
    scatter = axes[0,1].collections[0]
    cbar = plt.colorbar(scatter, ax=axes[0,1])
    cbar.set_label('Uncertainty Score')
    
    # 3. Outlier Detection Methods Comparison
    methods = ['isolation_forest', 'elliptic_envelope', 'lof', 'mahalanobis']
    outlier_rates = [uncertainty_results['individual_scores'][method].mean() for method in methods]
    
    bars = axes[0,2].bar(methods, outlier_rates, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'],
                        alpha=0.8, edgecolor='black')
    axes[0,2].set_ylabel('Average Uncertainty Score')
    axes[0,2].set_title('Outlier Detection Methods')
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, rate in zip(bars, outlier_rates):
        height = bar.get_height()
        axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                      f'{rate:.3f}', ha='center', va='bottom')
    
    # 4. Confidence vs Uncertainty Relationship
    axes[1,0].scatter(uncertainty_scores, original_confidence, alpha=0.6, label='Original', color='blue')
    axes[1,0].scatter(uncertainty_scores, adjusted_confidence, alpha=0.6, label='Adjusted', color='red')
    axes[1,0].set_xlabel('Uncertainty Score')
    axes[1,0].set_ylabel('Confidence')
    axes[1,0].set_title('Confidence vs Uncertainty')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    
    # 5. Outlier Classification
    colors = ['blue' if not outlier else 'red' for outlier in is_outlier]
    axes[1,1].scatter(range(len(uncertainty_scores)), uncertainty_scores, c=colors, alpha=0.6)
    axes[1,1].axhline(0.5, color='orange', linestyle='--', label='Threshold')
    axes[1,1].set_xlabel('Sample Index')
    axes[1,1].set_ylabel('Uncertainty Score')
    axes[1,1].set_title('Outlier Classification (Red=Outlier)')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    
    # 6. Performance Impact (if true labels available)
    if y_true is not None and y_pred is not None:
        correct = (y_true == y_pred)
        
        # Separate correct and incorrect predictions
        correct_uncertainty = uncertainty_scores[correct]
        incorrect_uncertainty = uncertainty_scores[~correct]
        
        axes[1,2].hist(correct_uncertainty, bins=15, alpha=0.7, label='Correct', color='green')
        axes[1,2].hist(incorrect_uncertainty, bins=15, alpha=0.7, label='Incorrect', color='red')
        axes[1,2].set_xlabel('Uncertainty Score')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Uncertainty: Correct vs Incorrect Predictions')
        axes[1,2].legend()
        axes[1,2].grid(alpha=0.3)
    else:
        axes[1,2].text(0.5, 0.5, 'True labels not available\nfor performance analysis', 
                      ha='center', va='center', transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].set_title('Performance Analysis (N/A)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved uncertainty analysis plot to: {save_path}")

def main():
    """
    Main function to demonstrate uncertainty quantification on validation data.
    """
    print("🔍 Uncertainty Quantification and Out-of-Distribution Detection")
    print("=" * 70)
    
    try:
        # Load model and training data
        models_dir = Path("models")
        model = joblib.load(models_dir / "latest_model.joblib")
        scaler = joblib.load(models_dir / "latest_scaler.joblib")
        feature_selector = joblib.load(models_dir / "latest_feature_selector.joblib")
        feature_names = joblib.load(models_dir / "latest_feature_names.joblib")
        
        print(f"✓ Loaded model components")
        
        # Load training data for fitting uncertainty quantifier
        train_improved = pd.read_excel('training_data/train_improved.xlsx')
        
        # Preprocess training data (same as model training)
        exclude_cols = ['NO.', 'type', 'source_file', 'Centroid', 'BoundingBox', 'WeightedCentroid']
        original_features = [col for col in train_improved.columns 
                           if col not in exclude_cols and train_improved[col].dtype in ['int64', 'float64']]
        
        X_train_raw = train_improved[original_features].fillna(train_improved[original_features].median())
        X_train_scaled = scaler.transform(X_train_raw)
        X_train_selected = feature_selector.transform(X_train_scaled)
        
        print(f"✓ Preprocessed training data: {X_train_selected.shape}")
        
        # Fit uncertainty quantifier
        uq = UncertaintyQuantifier(contamination=0.15)  # Expect 15% outliers in validation
        uq.fit(X_train_selected, feature_names)
        
        # Load and preprocess validation data
        val_improved = pd.read_excel('validation_data_improved.xlsx')
        X_val_raw = val_improved[original_features].fillna(val_improved[original_features].median())
        X_val_scaled = scaler.transform(X_val_raw)
        X_val_selected = feature_selector.transform(X_val_scaled)
        y_val_true = val_improved['type']
        
        print(f"✓ Preprocessed validation data: {X_val_selected.shape}")
        
        # Make predictions
        y_val_pred = model.predict(X_val_selected)
        y_val_proba = model.predict_proba(X_val_selected)
        original_confidence = np.max(y_val_proba, axis=1)
        
        # Convert predictions back to original labels
        label_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
        y_val_pred_labels = [label_mapping[pred] for pred in y_val_pred]
        
        print(f"✓ Made predictions on validation data")
        
        # Detect outliers and calculate uncertainty
        outlier_results = uq.detect_outliers(X_val_selected, feature_names)
        uncertainty_results = uq.calculate_uncertainty_scores(outlier_results)
        
        # Adjust confidence based on uncertainty
        adjusted_confidence = adjust_prediction_confidence(
            original_confidence, uncertainty_results['uncertainty_scores'], method='linear'
        )
        
        # Print results
        print(f"\n📊 UNCERTAINTY ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Outlier detection rates:")
        for method in ['isolation_forest', 'elliptic_envelope', 'lof', 'mahalanobis']:
            rate = outlier_results[method]['outlier_rate']
            print(f"  {method:20}: {rate:.1%}")
        
        print(f"\nConfidence adjustment:")
        print(f"  Original mean confidence: {np.mean(original_confidence):.3f}")
        print(f"  Adjusted mean confidence: {np.mean(adjusted_confidence):.3f}")
        print(f"  Average uncertainty score: {np.mean(uncertainty_results['uncertainty_scores']):.3f}")
        print(f"  Samples flagged as outliers: {np.sum(uncertainty_results['is_outlier'])}/{len(uncertainty_results['is_outlier'])} ({np.mean(uncertainty_results['is_outlier']):.1%})")
        
        # Calculate accuracy for high vs low uncertainty samples
        is_correct = (y_val_true == y_val_pred_labels)
        high_uncertainty = uncertainty_results['uncertainty_scores'] > 0.5
        
        low_uncertainty_acc = np.mean(is_correct[~high_uncertainty]) if np.sum(~high_uncertainty) > 0 else 0
        high_uncertainty_acc = np.mean(is_correct[high_uncertainty]) if np.sum(high_uncertainty) > 0 else 0
        
        print(f"\nAccuracy by uncertainty level:")
        print(f"  Low uncertainty samples: {low_uncertainty_acc:.1%} (n={np.sum(~high_uncertainty)})")
        print(f"  High uncertainty samples: {high_uncertainty_acc:.1%} (n={np.sum(high_uncertainty)})")
        
        # Create plots
        create_uncertainty_analysis_plots(
            uncertainty_results, original_confidence, adjusted_confidence,
            y_val_true, y_val_pred_labels, "uncertainty_analysis.png"
        )
        
        # Save results
        results_df = pd.DataFrame({
            'true_class': y_val_true,
            'predicted_class': y_val_pred_labels,
            'original_confidence': original_confidence,
            'adjusted_confidence': adjusted_confidence,
            'uncertainty_score': uncertainty_results['uncertainty_scores'],
            'is_outlier': uncertainty_results['is_outlier'],
            'is_correct': is_correct
        })
        
        results_df.to_excel('uncertainty_analysis_results.xlsx', index=False)
        print(f"✓ Saved detailed results to: uncertainty_analysis_results.xlsx")
        
        print(f"\n✅ Uncertainty quantification complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 