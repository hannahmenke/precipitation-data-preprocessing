#!/usr/bin/env python3
"""
Create Training vs Test Performance Plots.

This script creates the exact same visualization as recreate_plots.py
for training vs test performance analysis.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pathlib import Path
from collections import Counter

def load_model_and_data():
    """Load the latest saved model and test data."""
    models_dir = Path("models")
    
    # Load model components
    model = joblib.load(models_dir / "latest_model.joblib")
    scaler = joblib.load(models_dir / "latest_scaler.joblib")
    label_encoder = joblib.load(models_dir / "latest_label_encoder.joblib")
    feature_names = joblib.load(models_dir / "latest_feature_names.joblib")
    feature_selector = joblib.load(models_dir / "latest_feature_selector.joblib")
    metadata = joblib.load(models_dir / "latest_metadata.joblib")
    
    print(f"✓ Loaded model trained at: {metadata['timestamp']}")
    print(f"✓ Model accuracy: {metadata.get('test_accuracy', 'N/A')}")
    print(f"✓ Features used: {len(feature_names)}")
    
    return model, scaler, label_encoder, feature_names, feature_selector, metadata

def load_and_preprocess_test_data(scaler, label_encoder, feature_names, feature_selector):
    """Load and preprocess the test data using the enhanced features."""
    try:
        # Load the enhanced/improved test data that has all the engineered features
        test_df = pd.read_excel('validation_data_improved.xlsx')
        print(f"✓ Loaded enhanced test data: {test_df.shape}")
        
        # Use only the features that the model was trained on - exactly as they are
        available_features = [f for f in feature_names if f in test_df.columns]
        print(f"✓ Available features: {len(available_features)}/{len(feature_names)}")
        
        if len(available_features) < len(feature_names):
            missing_features = set(feature_names) - set(available_features)
            print(f"⚠️  Missing features: {missing_features}")
            
            # Create dummy columns for missing features with median/zero values
            for missing_feat in missing_features:
                test_df[missing_feat] = 0  # Use zero for missing engineered features
        
        # Now use exactly the feature names the model expects
        X_test_raw = test_df[feature_names].copy()
        y_test_raw = test_df['type'].copy()
        
        # Handle missing values
        for col in X_test_raw.columns:
            if X_test_raw[col].isnull().sum() > 0:
                X_test_raw[col].fillna(X_test_raw[col].median(), inplace=True)
        
        # Apply scaling (using the exact features the model expects)
        X_test_scaled = scaler.transform(X_test_raw)
        
        # Apply feature selection - this should now work since we have the right features
        X_test_selected = feature_selector.transform(X_test_scaled)
        X_test_final = pd.DataFrame(X_test_selected, columns=feature_names)
        
        # Encode labels
        y_test = label_encoder.transform(y_test_raw)
        
        print(f"✓ Preprocessed enhanced test data: {X_test_final.shape}")
        
        return X_test_final, y_test, y_test_raw
        
    except Exception as e:
        print(f"❌ Error loading enhanced test data: {e}")
        return None, None, None

def load_and_preprocess_train_data(scaler, label_encoder, feature_names, feature_selector):
    """Load and preprocess the training data using the enhanced features."""
    try:
        # Load enhanced training data
        train_df = pd.read_excel('training_data/train_improved.xlsx')
        print(f"✓ Loaded enhanced training data: {train_df.shape}")
        
        # Use only the features that the model was trained on - exactly as they are
        available_features = [f for f in feature_names if f in train_df.columns]
        print(f"✓ Available features: {len(available_features)}/{len(feature_names)}")
        
        if len(available_features) < len(feature_names):
            missing_features = set(feature_names) - set(available_features)
            print(f"⚠️  Missing features: {missing_features}")
            
            # Create dummy columns for missing features with median/zero values
            for missing_feat in missing_features:
                train_df[missing_feat] = 0  # Use zero for missing engineered features
        
        # Now use exactly the feature names the model expects
        X_train_raw = train_df[feature_names].copy()
        y_train_raw = train_df['type'].copy()
        
        # Handle missing values
        for col in X_train_raw.columns:
            if X_train_raw[col].isnull().sum() > 0:
                X_train_raw[col].fillna(X_train_raw[col].median(), inplace=True)
        
        # Apply scaling (using the exact features the model expects)
        X_train_scaled = scaler.transform(X_train_raw)
        
        # Apply feature selection - this should now work since we have the right features  
        X_train_selected = feature_selector.transform(X_train_scaled)
        X_train_final = pd.DataFrame(X_train_selected, columns=feature_names)
        
        # Encode labels
        y_train = label_encoder.transform(y_train_raw)
        
        print(f"✓ Preprocessed enhanced training data: {X_train_final.shape}")
        
        return X_train_final, y_train, y_train_raw
        
    except Exception as e:
        print(f"⚠️  Could not load enhanced training data: {e}")
        return None, None, None

def create_train_test_visualization(model, X_train, y_train, y_train_raw, X_test, y_test, y_test_raw, feature_names, label_encoder):
    """Create and save the exact same classification visualization as recreate_plots.py."""
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"✓ Training Accuracy: {train_accuracy:.4f}")
    print(f"✓ Test Accuracy: {test_accuracy:.4f}")
    
    # Create the visualization - EXACT SAME as recreate_plots.py
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'XGBoost Classification Results (Test Accuracy: {test_accuracy:.1%})', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    class_names = [f'Class {i}' for i in label_encoder.classes_]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix', fontweight='bold')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. Feature Importance
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    y_pos = np.arange(len(feature_names))
    axes[0,1].barh(y_pos, feature_importance_df['importance'], color='skyblue', alpha=0.8)
    axes[0,1].set_yticks(y_pos)
    axes[0,1].set_yticklabels(feature_importance_df['feature'])
    axes[0,1].set_xlabel('Importance')
    axes[0,1].set_title('Feature Importance', fontweight='bold')
    axes[0,1].grid(axis='x', alpha=0.3)
    
    # 3. Training vs Test Accuracy - EXACT SAME as recreate_plots.py
    accuracies = [train_accuracy, test_accuracy]
    labels = ['Training', 'Test']
    colors = ['lightgreen', 'lightcoral']
    
    bars = axes[1,0].bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black')
    axes[1,0].set_ylim([0, 1])
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].set_title('Training vs Test Accuracy', fontweight='bold')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Class Distribution and Predictions
    true_counts = Counter(y_test_raw)
    pred_counts = Counter(label_encoder.inverse_transform(y_test_pred))
    
    classes = sorted(true_counts.keys())
    true_values = [true_counts[cls] for cls in classes]
    pred_values = [pred_counts.get(cls, 0) for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[1,1].bar(x - width/2, true_values, width, label='True', alpha=0.8, color='lightblue')
    axes[1,1].bar(x + width/2, pred_values, width, label='Predicted', alpha=0.8, color='orange')
    
    axes[1,1].set_xlabel('Class')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Class Distribution: True vs Predicted', fontweight='bold')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels([f'Class {cls}' for cls in classes])
    axes[1,1].legend()
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'train_test_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved train/test visualization to: {output_file}")
    
    # Print classification report
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    return train_accuracy, test_accuracy

def main():
    """Main function to create the train/test performance plots."""
    print("🎨 Creating Training vs Test Performance Plots")
    print("=" * 60)
    
    try:
        # Load model and components
        model, scaler, label_encoder, feature_names, feature_selector, metadata = load_model_and_data()
        
        # Load and preprocess training data
        X_train, y_train, y_train_raw = load_and_preprocess_train_data(scaler, label_encoder, feature_names, feature_selector)
        
        # Load and preprocess test data
        X_test, y_test, y_test_raw = load_and_preprocess_test_data(scaler, label_encoder, feature_names, feature_selector)
        
        if X_train is None or X_test is None:
            print("❌ Could not load required data")
            return
        
        # Create visualizations - EXACT SAME as recreate_plots.py
        train_acc, test_acc = create_train_test_visualization(
            model, X_train, y_train, y_train_raw, X_test, y_test, y_test_raw, feature_names, label_encoder)
        
        print(f"\n✅ Train/Test plots created successfully!")
        print(f"📊 Training Accuracy: {train_acc:.1%}")
        print(f"📊 Test Accuracy: {test_acc:.1%}")
        print(f"📊 Generalization Gap: {test_acc - train_acc:+.1%}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 