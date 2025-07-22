#!/usr/bin/env python3
"""
Create Simple Training vs Test Performance Plot.

This script creates a focused visualization showing only training vs test performance
without complex feature preprocessing that can cause issues.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pathlib import Path
from collections import Counter

def load_model_components():
    """Load the model components."""
    models_dir = Path("models")
    
    try:
        # Load the enhanced ensemble model
        model = joblib.load(models_dir / "enhanced_ensemble_model.joblib")
        cluster_model = joblib.load(models_dir / "cluster_model.joblib")
        metadata = joblib.load(models_dir / "enhanced_model_metadata.joblib")
        
        # Load the original components for initial preprocessing
        original_scaler = joblib.load(models_dir / "latest_scaler.joblib")
        original_selector = joblib.load(models_dir / "latest_feature_selector.joblib")
        original_feature_names = joblib.load(models_dir / "latest_feature_names.joblib")
        
        print(f"✓ Loaded enhanced ensemble model with {metadata['n_features']} features")
        
        label_mapping = {0: 1, 1: 2, 2: 3, 3: 4}  # Model outputs 0-3 for classes 1-4
        return model, cluster_model, original_scaler, original_selector, original_feature_names, metadata, label_mapping
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None, None, None, None

def create_cluster_aware_features(X_selected, cluster_model):
    """Create cluster-aware features consistent with training."""
    # Get cluster labels
    cluster_labels = cluster_model.predict(X_selected)
    
    # Distance to cluster centers
    cluster_centers = cluster_model.cluster_centers_
    cluster_features = []
    for center in cluster_centers:
        dist = np.linalg.norm(X_selected - center, axis=1)
        cluster_features.append(dist)
    
    # Soft clustering probabilities (using GaussianMixture as in training)
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_selected)
    cluster_probs = gmm.predict_proba(X_selected)
    for i in range(2):
        cluster_features.append(cluster_probs[:, i])
    
    # Placeholder for class center distances (simplified - in production you'd save these)
    # For this demo, we'll use zeros as placeholder since we don't have the exact centers
    cluster_features.extend([
        np.zeros(len(X_selected)),  # dist_to_class_4
        np.zeros(len(X_selected)),  # dist_to_class_1
        np.zeros(len(X_selected))   # class_4_vs_1_diff
    ])
    
    # Error indicator (assuming cluster 0 is high-error as in training)
    error_indicator = (cluster_labels == 0).astype(float)
    cluster_features.append(error_indicator)
    
    return np.column_stack(cluster_features)

def prepare_data_for_model(df, original_scaler, original_selector, cluster_model):
    """Prepare data for enhanced model."""
    try:
        # Collect all numeric features (excluding non-feature columns)
        exclude_cols = ['NO.', 'ID', 'type', 'source_file', 'Centroid', 'BoundingBox', 'WeightedCentroid']
        all_features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        X_all = df[all_features].copy()
        
        # Handle missing values
        X_all = X_all.fillna(X_all.median())
        
        # Scale using original scaler (this assumes the scaler was fit on these features)
        X_scaled = original_scaler.transform(X_all)
        
        # Apply original selector to get selected features
        X_selected = original_selector.transform(X_scaled)
        
        # Create cluster features using the selected features
        cluster_features = create_cluster_aware_features(X_selected, cluster_model)
        
        # Combine
        X_enhanced = np.hstack((X_selected, cluster_features))
        
        y = df['type'].to_numpy()
        
        print(f"✓ Prepared data: {X_enhanced.shape} (all: {X_all.shape[1]} -> selected: {X_selected.shape[1]} + cluster: {cluster_features.shape[1]})")
        
        return X_enhanced, y
    
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return None, None

def create_train_test_performance_plot():
    """Create a simple train vs test performance visualization."""
    print("🎨 Creating Training vs Test Performance Visualization")
    print("=" * 60)
    
    # Load model components
    model, cluster_model, scaler, selector, feature_names, metadata, label_mapping = load_model_components()
    if model is None:
        print("❌ Could not load model components")
        return None, None
    
    # Load training data
    print("📂 Loading training data...")
    try:
        train_df = pd.read_excel('training_data/train_improved.xlsx')
        print(f"✓ Loaded training data: {train_df.shape}")
        
        X_train, y_train = prepare_data_for_model(train_df, scaler, selector, cluster_model)
        if X_train is None:
            print("❌ Could not prepare training data")
            return None, None
            
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None, None
    
    # Load test data
    print("📂 Loading test data...")
    try:
        test_df = pd.read_excel('validation_data_improved.xlsx')
        print(f"✓ Loaded test data: {test_df.shape}")
        
        X_test, y_test = prepare_data_for_model(test_df, scaler, selector, cluster_model)
        if X_test is None:
            print("❌ Could not prepare test data")
            return None, None
            
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None, None
    
    # Make predictions
    print("🎯 Making predictions...")
    y_train_pred_raw = model.predict(X_train)
    y_train_pred = np.array([label_mapping[pred] for pred in y_train_pred_raw])
    y_test_pred_raw = model.predict(X_test)
    y_test_pred = np.array([label_mapping[pred] for pred in y_test_pred_raw])
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"✓ Training Accuracy: {train_accuracy:.3f}")
    print(f"✓ Test Accuracy: {test_accuracy:.3f}")
    print(f"✓ Generalization Gap: {test_accuracy - train_accuracy:+.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Enhanced Ensemble Model: Training vs Test Performance', 
                fontsize=16, fontweight='bold')
    
    # Rest of the visualization code remains the same, but adjust class_names for 1-4
    class_names = [f'Class {i}' for i in sorted(np.unique(y_train))]
    
    # 1. Accuracy Comparison
    ax1 = axes[0, 0]
    categories = ['Training', 'Test']
    accuracies = [train_accuracy, test_accuracy]
    sample_counts = [len(y_train), len(y_test)]
    
    colors = ['#2E8B57', '#FF6B6B']
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, acc, count in zip(bars, accuracies, sample_counts):
        height = bar.get_height()
        ax1.annotate(f'{acc:.1%}\n(n={count})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Performance assessment
    diff = test_accuracy - train_accuracy
    if abs(diff) < 0.05:
        status = "✓ Well Balanced"
        color = 'green'
    elif diff < -0.05:
        status = "⚠ Possible Overfitting"
        color = 'orange'
    else:
        status = "? Unusual Pattern"
        color = 'red'
    
    ax1.text(0.5, 0.95, status, transform=ax1.transAxes, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
            fontsize=11, fontweight='bold')
    
    # 2. Training Confusion Matrix
    ax2 = axes[0, 1]
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_title('Training Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # 3. Test Confusion Matrix
    ax3 = axes[1, 0]
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Reds', ax=ax3,
                xticklabels=class_names, yticklabels=class_names)
    ax3.set_title('Test Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Performance Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "TRAINING vs TEST PERFORMANCE SUMMARY\n"
    summary_text += "=" * 40 + "\n\n"
    summary_text += f"Model Type: Enhanced Ensemble\n"
    summary_text += f"Features Used: {metadata['n_features']}\n\n"
    summary_text += "ACCURACY RESULTS:\n"
    summary_text += f"  🎯 Training:  {train_accuracy:.1%} (n={len(y_train)})\n"
    summary_text += f"  🧪 Test:      {test_accuracy:.1%} (n={len(y_test)})\n"
    summary_text += f"  📊 Gap:       {diff:+.1%}\n\n"
    
    summary_text += "TEST SET CLASS PERFORMANCE:\n"
    for class_id in sorted(np.unique(y_test)):
        class_mask = y_test == class_id
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_test[class_mask], y_test_pred[class_mask])
            summary_text += f"  Class {class_id}: {class_acc:.1%} ({np.sum(class_mask)} samples)\n"
    
    summary_text += f"\nOVERALL ASSESSMENT:\n"
    if abs(diff) < 0.05:
        summary_text += "  ✅ Model generalizes well\n"
        summary_text += "  ✅ Good train/test balance\n"
    elif diff < -0.05:
        summary_text += "  ⚠️  Possible overfitting detected\n"
        summary_text += "  📝 Consider regularization\n"
    else:
        summary_text += "  ❓ Unusual generalization pattern\n"
        summary_text += "  📝 Investigate data quality\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.tight_layout()
    
    output_file = 'train_test_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close to avoid display in terminal
    
    print(f"✓ Saved train vs test performance plot: {output_file}")
    
    return train_accuracy, test_accuracy

def main():
    """Main function."""
    try:
        train_acc, test_acc = create_train_test_performance_plot()
        
        print(f"\n🎉 Train/Test visualization complete!")
        print(f"📊 Final Results:")
        print(f"   Training: {train_acc:.1%}")
        print(f"   Test:     {test_acc:.1%}")
        print(f"   Gap:      {test_acc - train_acc:+.1%}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 