#!/usr/bin/env python3
"""
Create validation plots for val3 and val6 predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter

def load_validation_data_and_predictions():
    """Load validation data and predictions."""
    # Load original validation data
    val3 = pd.read_excel('data_for_classification/val3.xlsx')
    val6 = pd.read_excel('data_for_classification/val6.xlsx')
    
    # Load predictions
    val3_pred = pd.read_excel('val3_predictions_improved.xlsx')
    val6_pred = pd.read_excel('val6_predictions_improved.xlsx')
    
    print(f"✓ Loaded val3: {len(val3)} samples")
    print(f"✓ Loaded val6: {len(val6)} samples")
    print(f"✓ Total validation samples: {len(val3) + len(val6)}")
    
    return val3, val6, val3_pred, val6_pred

def create_validation_plots(val3, val6, val3_pred, val6_pred):
    """Create comprehensive validation plots."""
    
    # Combine validation data
    all_true = list(val3['type']) + list(val6['type'])
    all_pred = list(val3_pred['predicted_class']) + list(val6_pred['predicted_class'])
    all_confidence = list(val3_pred['confidence']) + list(val6_pred['confidence'])
    
    # Calculate accuracies
    val3_accuracy = accuracy_score(val3['type'], val3_pred['predicted_class'])
    val6_accuracy = accuracy_score(val6['type'], val6_pred['predicted_class'])
    combined_accuracy = accuracy_score(all_true, all_pred)
    
    print(f"✓ Val3 Accuracy: {val3_accuracy:.4f} ({val3_accuracy*100:.1f}%)")
    print(f"✓ Val6 Accuracy: {val6_accuracy:.4f} ({val6_accuracy*100:.1f}%)")
    print(f"✓ Combined Validation Accuracy: {combined_accuracy:.4f} ({combined_accuracy*100:.1f}%)")
    
    # Create the main validation plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Validation Results (Combined Accuracy: {combined_accuracy:.1%})', fontsize=16, fontweight='bold')
    
    # 1. Combined Confusion Matrix
    cm_combined = confusion_matrix(all_true, all_pred)
    class_names = [f'Class {i}' for i in sorted(set(all_true))]
    
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
    axes[0,0].set_title('Combined Validation Confusion Matrix', fontweight='bold')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. Accuracy Comparison
    accuracies = [val3_accuracy, val6_accuracy, combined_accuracy]
    labels = ['Val3\n(3mM)', 'Val6\n(6mM)', 'Combined']
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    bars = axes[0,1].bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black')
    axes[0,1].set_ylim([0, 1])
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].set_title('Validation Accuracy by Dataset', fontweight='bold')
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Class Distribution Comparison
    true_counts = Counter(all_true)
    pred_counts = Counter(all_pred)
    
    classes = sorted(true_counts.keys())
    true_values = [true_counts[cls] for cls in classes]
    pred_values = [pred_counts.get(cls, 0) for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[1,0].bar(x - width/2, true_values, width, label='True', alpha=0.8, color='lightblue')
    axes[1,0].bar(x + width/2, pred_values, width, label='Predicted', alpha=0.8, color='orange')
    
    axes[1,0].set_xlabel('Class')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Class Distribution: True vs Predicted', fontweight='bold')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([f'Class {cls}' for cls in classes])
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 4. Confidence Distribution
    axes[1,1].hist(all_confidence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,1].axvline(np.mean(all_confidence), color='red', linestyle='--', linewidth=2, 
                     label=f'Mean: {np.mean(all_confidence):.3f}')
    axes[1,1].set_xlabel('Prediction Confidence')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Prediction Confidence Distribution', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the main plot
    main_output = 'validation_results_combined.png'
    plt.savefig(main_output, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved main validation plot to: {main_output}")
    
    # Create individual dataset plots
    create_individual_validation_plots(val3, val6, val3_pred, val6_pred, val3_accuracy, val6_accuracy)
    
    # Print detailed classification reports
    print(f"\n" + "="*60)
    print("DETAILED VALIDATION RESULTS")
    print("="*60)
    
    print(f"\nVAL3 (3mM) Classification Report:")
    print(classification_report(val3['type'], val3_pred['predicted_class']))
    
    print(f"\nVAL6 (6mM) Classification Report:")
    print(classification_report(val6['type'], val6_pred['predicted_class']))
    
    print(f"\nCOMBINED Classification Report:")
    print(classification_report(all_true, all_pred))

def create_individual_validation_plots(val3, val6, val3_pred, val6_pred, val3_accuracy, val6_accuracy):
    """Create individual plots for val3 and val6."""
    
    # Create side-by-side individual plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Individual Validation Dataset Results', fontsize=16, fontweight='bold')
    
    # Val3 Confusion Matrix
    cm_val3 = confusion_matrix(val3['type'], val3_pred['predicted_class'])
    class_names_val3 = [f'Class {i}' for i in sorted(set(val3['type']))]
    
    sns.heatmap(cm_val3, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names_val3, yticklabels=class_names_val3, ax=axes[0,0])
    axes[0,0].set_title(f'Val3 (3mM) - Accuracy: {val3_accuracy:.1%}', fontweight='bold')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # Val6 Confusion Matrix
    cm_val6 = confusion_matrix(val6['type'], val6_pred['predicted_class'])
    class_names_val6 = [f'Class {i}' for i in sorted(set(val6['type']))]
    
    sns.heatmap(cm_val6, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names_val6, yticklabels=class_names_val6, ax=axes[0,1])
    axes[0,1].set_title(f'Val6 (6mM) - Accuracy: {val6_accuracy:.1%}', fontweight='bold')
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('Actual')
    
    # Val3 Class Distribution
    val3_true_counts = Counter(val3['type'])
    val3_pred_counts = Counter(val3_pred['predicted_class'])
    
    classes_val3 = sorted(val3_true_counts.keys())
    val3_true_values = [val3_true_counts[cls] for cls in classes_val3]
    val3_pred_values = [val3_pred_counts.get(cls, 0) for cls in classes_val3]
    
    x_val3 = np.arange(len(classes_val3))
    width = 0.35
    
    axes[1,0].bar(x_val3 - width/2, val3_true_values, width, label='True', alpha=0.8, color='lightcoral')
    axes[1,0].bar(x_val3 + width/2, val3_pred_values, width, label='Predicted', alpha=0.8, color='darkorange')
    axes[1,0].set_xlabel('Class')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Val3 (3mM) Class Distribution', fontweight='bold')
    axes[1,0].set_xticks(x_val3)
    axes[1,0].set_xticklabels([f'Class {cls}' for cls in classes_val3])
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Val6 Class Distribution
    val6_true_counts = Counter(val6['type'])
    val6_pred_counts = Counter(val6_pred['predicted_class'])
    
    classes_val6 = sorted(val6_true_counts.keys())
    val6_true_values = [val6_true_counts[cls] for cls in classes_val6]
    val6_pred_values = [val6_pred_counts.get(cls, 0) for cls in classes_val6]
    
    x_val6 = np.arange(len(classes_val6))
    
    axes[1,1].bar(x_val6 - width/2, val6_true_values, width, label='True', alpha=0.8, color='lightblue')
    axes[1,1].bar(x_val6 + width/2, val6_pred_values, width, label='Predicted', alpha=0.8, color='navy')
    axes[1,1].set_xlabel('Class')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Val6 (6mM) Class Distribution', fontweight='bold')
    axes[1,1].set_xticks(x_val6)
    axes[1,1].set_xticklabels([f'Class {cls}' for cls in classes_val6])
    axes[1,1].legend()
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save individual plot
    individual_output = 'validation_results_individual.png'
    plt.savefig(individual_output, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved individual validation plot to: {individual_output}")

def main():
    """Main function to create validation plots."""
    print("🎨 Creating Validation Result Plots")
    print("=" * 50)
    
    try:
        # Load data
        val3, val6, val3_pred, val6_pred = load_validation_data_and_predictions()
        
        # Create plots
        create_validation_plots(val3, val6, val3_pred, val6_pred)
        
        print("\n✅ Validation plots created successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 