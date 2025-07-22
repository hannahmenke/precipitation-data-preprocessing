#!/usr/bin/env python3
"""
Test Enhanced Model on Validation Datasets.

This script comprehensively tests the cluster-enhanced ensemble model
on the validation datasets and provides detailed performance comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class EnhancedModelTester:
    """
    Class to test the enhanced model on validation data.
    """
    
    def __init__(self):
        self.models_dir = Path("models")
        self.original_model = None
        self.enhanced_model = None
        self.cluster_model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        
    def load_models(self):
        """Load all model components."""
        print("📂 Loading model components...")
        
        # Load original model components
        self.original_model = joblib.load(self.models_dir / "latest_model.joblib")
        self.scaler = joblib.load(self.models_dir / "latest_scaler.joblib")
        self.feature_selector = joblib.load(self.models_dir / "latest_feature_selector.joblib")
        self.feature_names = joblib.load(self.models_dir / "latest_feature_names.joblib")
        
        # Load enhanced model components
        self.enhanced_model = joblib.load(self.models_dir / "enhanced_ensemble_model.joblib")
        self.cluster_model = joblib.load(self.models_dir / "cluster_model.joblib")
        
        print("✓ Loaded original model components")
        print("✓ Loaded enhanced ensemble model")
        print("✓ Loaded cluster model")
        
    def load_validation_data(self):
        """Load and preprocess validation datasets."""
        print("\n📊 Loading validation datasets...")
        
        # Auto-discover ALL Excel files in data_for_classification
        validation_dir = Path('data_for_classification')
        all_val_files = list(validation_dir.glob("*.xlsx"))
        all_val_files = [f for f in all_val_files if not f.name.startswith('.') and not f.name.startswith('~')]
        
        if not all_val_files:
            print("⚠️  No validation files found in data_for_classification/")
            return None, None, None
        
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
            return None, None, None
        
        # Create TRUE combined dataset from ALL files
        print(f"\n🔄 Creating TRUE combined dataset from ALL {len(all_datasets)} files...")
        combined_val_df = pd.concat(improved_datasets, ignore_index=True)
        
        print(f"✓ TRUE combined dataset: {len(combined_val_df)} samples from {len(all_datasets)} files")
        print(f"✓ Total validation samples: {total_samples}")
        
        # Extract individual datasets for compatibility
        val3_improved = all_datasets.get('val3', None)
        val6_improved = all_datasets.get('val6', None)
        
        if val3_improved is not None:
            print(f"✓ val3 dataset: {len(val3_improved)} samples")
        if val6_improved is not None:
            print(f"✓ val6 dataset: {len(val6_improved)} samples")
        
        # Print breakdown by dataset
        print(f"\n📊 Dataset breakdown:")
        for dataset_name in all_datasets:
            count = len(all_datasets[dataset_name])
            percentage = (count / total_samples) * 100
            print(f"  {dataset_name}: {count} samples ({percentage:.1f}%)")
        
        return val3_improved, val6_improved, combined_val_df
    
    def preprocess_validation_data(self, val_df):
        """Preprocess validation data for both models."""
        print("🔄 Preprocessing validation data...")
        
        # Handle column mapping issues
        val_df = val_df.copy()
        
        # Map column names
        if 'NO.' in val_df.columns and 'ID' not in val_df.columns:
            val_df['ID'] = val_df['NO.']
        
        # Create missing features if needed
        if 'Gray_ave' not in val_df.columns:
            if 'MeanIntensity' in val_df.columns:
                val_df['Gray_ave'] = val_df['MeanIntensity']
            else:
                val_df['Gray_ave'] = 0
        
        if 'overall_mean_gray' not in val_df.columns:
            if 'MeanIntensity' in val_df.columns:
                val_df['overall_mean_gray'] = val_df['MeanIntensity']
            else:
                val_df['overall_mean_gray'] = 0
        
        # Get features for processing
        exclude_cols = ['NO.', 'ID', 'type', 'source_file', 'source', 'Centroid', 'BoundingBox', 'WeightedCentroid']
        available_features = [col for col in val_df.columns 
                            if col not in exclude_cols and val_df[col].dtype in ['int64', 'float64']]
        
        # Use only features that exist in both datasets
        common_features = [feat for feat in available_features if feat in val_df.columns]
        
        # Get training feature names for comparison
        training_features_path = Path('training_data/train_improved.xlsx')
        if training_features_path.exists():
            train_df = pd.read_excel(training_features_path)
            train_features = [col for col in train_df.columns 
                            if col not in exclude_cols and train_df[col].dtype in ['int64', 'float64']]
            # Use intersection of available and training features
            common_features = [feat for feat in common_features if feat in train_features]
        
        print(f"✓ Using {len(common_features)} common features")
        
        # Prepare feature matrix
        X_raw = val_df[common_features].fillna(val_df[common_features].median())
        X_scaled = self.scaler.transform(X_raw)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Get true labels
        y_true = val_df['type'] if 'type' in val_df.columns else None
        
        return X_selected, X_raw, y_true, val_df
    
    def create_enhanced_features(self, X_selected):
        """Create enhanced features for the ensemble model."""
        print("🔧 Creating enhanced features...")
        
        # Get cluster predictions
        cluster_labels = self.cluster_model.predict(X_selected)
        
        # Create cluster features
        cluster_features = []
        
        # 1. Distance to cluster centers
        cluster_centers = self.cluster_model.cluster_centers_
        for center in cluster_centers:
            dist = np.sqrt(np.sum((X_selected - center) ** 2, axis=1))
            cluster_features.append(dist)
        
        # 2. Cluster membership probabilities (soft clustering)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X_selected)
        cluster_probs = gmm.predict_proba(X_selected)
        for i in range(2):
            cluster_features.append(cluster_probs[:, i])
        
        # 3. Simplified class distance features (placeholders)
        # In production, you'd store the actual class centers from training
        cluster_features.extend([
            np.zeros(len(X_selected)),  # dist_to_class_4
            np.zeros(len(X_selected)),  # dist_to_class_1
            np.zeros(len(X_selected))   # class_4_vs_1_diff
        ])
        
        # 4. Error indicator based on cluster membership
        error_indicator = (cluster_labels == 0).astype(float)  # Cluster 0 was high-error
        cluster_features.append(error_indicator)
        
        # Combine features
        cluster_features_array = np.column_stack(cluster_features)
        X_enhanced = np.column_stack([X_selected, cluster_features_array])
        
        print(f"✓ Enhanced features shape: {X_enhanced.shape}")
        
        return X_enhanced, cluster_labels, error_indicator
    
    def make_predictions(self, X_selected, X_enhanced):
        """Make predictions with both models."""
        print("🎯 Making predictions...")
        
        # Original model predictions
        y_pred_original = self.original_model.predict(X_selected)
        y_prob_original = self.original_model.predict_proba(X_selected)
        
        # Enhanced model predictions
        y_pred_enhanced = self.enhanced_model.predict(X_enhanced)
        y_prob_enhanced = self.enhanced_model.predict_proba(X_enhanced)
        
        # Convert predictions back to original labels (1, 2, 3, 4)
        label_mapping = {0: 1, 1: 2, 2: 3, 3: 4}
        y_pred_original_labels = [label_mapping[pred] for pred in y_pred_original]
        y_pred_enhanced_labels = [label_mapping[pred] for pred in y_pred_enhanced]
        
        # Calculate confidence scores
        confidence_original = np.max(y_prob_original, axis=1)
        confidence_enhanced = np.max(y_prob_enhanced, axis=1)
        
        print("✓ Generated predictions and confidence scores")
        
        return {
            'original': {
                'predictions': y_pred_original_labels,
                'probabilities': y_prob_original,
                'confidence': confidence_original
            },
            'enhanced': {
                'predictions': y_pred_enhanced_labels,
                'probabilities': y_prob_enhanced,
                'confidence': confidence_enhanced
            }
        }
    
    def evaluate_performance(self, y_true, predictions, dataset_name):
        """Evaluate model performance with detailed metrics."""
        print(f"\n📈 Evaluating performance on {dataset_name}...")
        
        if y_true is None:
            print("⚠️  No true labels available for evaluation")
            return None
        
        results = {}
        
        for model_name, pred_data in predictions.items():
            y_pred = pred_data['predictions']
            confidence = pred_data['confidence']
            
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Confidence analysis
            avg_confidence = np.mean(confidence)
            confidence_std = np.std(confidence)
            
            # Calibration analysis (accuracy vs confidence)
            correct = (np.array(y_true) == np.array(y_pred))
            high_conf_mask = confidence > 0.9
            high_conf_accuracy = np.mean(correct[high_conf_mask]) if np.sum(high_conf_mask) > 0 else 0
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'class_report': class_report,
                'confusion_matrix': cm,
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'high_conf_accuracy': high_conf_accuracy,
                'n_high_conf': np.sum(high_conf_mask)
            }
            
            print(f"  {model_name.title()} Model:")
            print(f"    Accuracy: {accuracy:.1%}")
            print(f"    Precision: {precision:.3f}")
            print(f"    Recall: {recall:.3f}")
            print(f"    F1-Score: {f1:.3f}")
            print(f"    Avg Confidence: {avg_confidence:.3f}")
            print(f"    High Conf Accuracy: {high_conf_accuracy:.1%} (n={np.sum(high_conf_mask)})")
        
        return results
    
    def create_comparison_visualizations(self, results_dict, predictions_dict, save_prefix="validation_comparison"):
        """Create comprehensive comparison visualizations."""
        print("📊 Creating comparison visualizations...")
        
        # Create a large comparison figure
        fig = plt.figure(figsize=(20, 15))
        
        # Define datasets to compare
        datasets = list(results_dict.keys())
        
        # 1. Accuracy Comparison
        ax1 = plt.subplot(3, 4, 1)
        accuracies = {}
        for dataset in datasets:
            if results_dict[dataset] is not None:
                accuracies[dataset] = {
                    'Original': results_dict[dataset]['original']['accuracy'],
                    'Enhanced': results_dict[dataset]['enhanced']['accuracy']
                }
        
        if accuracies:
            dataset_names = list(accuracies.keys())
            original_accs = [accuracies[d]['Original'] for d in dataset_names]
            enhanced_accs = [accuracies[d]['Enhanced'] for d in dataset_names]
            
            x = np.arange(len(dataset_names))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, original_accs, width, label='Original', alpha=0.7, color='blue')
            bars2 = ax1.bar(x + width/2, enhanced_accs, width, label='Enhanced', alpha=0.7, color='red')
            
            ax1.set_xlabel('Dataset')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(dataset_names)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.annotate(f'{height:.1%}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=10)
        
        # 2. Confidence Distribution Comparison
        ax2 = plt.subplot(3, 4, 2)
        
        # Collect confidence data from predictions_dict
        confidence_data = {}
        for dataset in datasets:
            if results_dict[dataset] is not None and dataset in predictions_dict and predictions_dict[dataset] is not None:
                confidence_data[dataset] = {
                    'original': predictions_dict[dataset]['original']['confidence'],
                    'enhanced': predictions_dict[dataset]['enhanced']['confidence']
                }
        
        # Plot confidence distributions
        if confidence_data:
            # Use combined dataset for clearest comparison
            if 'combined' in confidence_data:
                dataset_to_plot = 'combined'
            else:
                dataset_to_plot = list(confidence_data.keys())[0]
            
            orig_conf = confidence_data[dataset_to_plot]['original']
            enh_conf = confidence_data[dataset_to_plot]['enhanced']
            
            ax2.hist(orig_conf, bins=20, alpha=0.7, label='Original', color='blue', density=True)
            ax2.hist(enh_conf, bins=20, alpha=0.7, label='Enhanced', color='red', density=True)
            ax2.axvline(np.mean(orig_conf), color='blue', linestyle='--', alpha=0.8, 
                       label=f'Orig Mean: {np.mean(orig_conf):.3f}')
            ax2.axvline(np.mean(enh_conf), color='red', linestyle='--', alpha=0.8,
                       label=f'Enh Mean: {np.mean(enh_conf):.3f}')
            ax2.legend(fontsize=8)
        
        ax2.set_title(f'Confidence Distributions\n({dataset_to_plot if confidence_data else "No Data"})')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Density')
        ax2.grid(alpha=0.3)
        
        # 3-4. Confusion Matrices for first dataset with labels
        if datasets and results_dict[datasets[0]] is not None:
            # Original model confusion matrix
            ax3 = plt.subplot(3, 4, 3)
            cm_orig = results_dict[datasets[0]]['original']['confusion_matrix']
            sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', ax=ax3)
            ax3.set_title(f'Original Model\n{datasets[0]}')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('True')
            
            # Enhanced model confusion matrix
            ax4 = plt.subplot(3, 4, 4)
            cm_enh = results_dict[datasets[0]]['enhanced']['confusion_matrix']
            sns.heatmap(cm_enh, annot=True, fmt='d', cmap='Reds', ax=ax4)
            ax4.set_title(f'Enhanced Model\n{datasets[0]}')
            ax4.set_xlabel('Predicted')
            ax4.set_ylabel('True')
        
        # 5. Per-Class Performance Comparison
        ax5 = plt.subplot(3, 4, (5, 8))
        
        # Aggregate per-class metrics across datasets
        class_metrics = {'Original': {}, 'Enhanced': {}}
        
        for dataset in datasets:
            if results_dict[dataset] is not None:
                for model in ['original', 'enhanced']:
                    model_name = model.title()
                    class_report = results_dict[dataset][model]['class_report']
                    
                    for class_id in ['1', '2', '3', '4']:
                        if class_id in class_report:
                            if class_id not in class_metrics[model_name]:
                                class_metrics[model_name][class_id] = []
                            class_metrics[model_name][class_id].append(class_report[class_id]['f1-score'])
        
        # Plot per-class F1-scores
        if class_metrics['Original'] or class_metrics['Enhanced']:
            classes = ['1', '2', '3', '4']
            x = np.arange(len(classes))
            width = 0.35
            
            orig_f1 = [np.mean(class_metrics['Original'].get(c, [0])) for c in classes]
            enh_f1 = [np.mean(class_metrics['Enhanced'].get(c, [0])) for c in classes]
            
            bars1 = ax5.bar(x - width/2, orig_f1, width, label='Original', alpha=0.7, color='blue')
            bars2 = ax5.bar(x + width/2, enh_f1, width, label='Enhanced', alpha=0.7, color='red')
            
            ax5.set_xlabel('Class')
            ax5.set_ylabel('F1-Score')
            ax5.set_title('Per-Class F1-Score Comparison')
            ax5.set_xticks(x)
            ax5.set_xticklabels(classes)
            ax5.legend()
            ax5.grid(axis='y', alpha=0.3)
        
        # 6. Improvement Summary
        ax6 = plt.subplot(3, 4, (9, 12))
        ax6.axis('off')
        
        # Create improvement summary text
        summary_text = "PERFORMANCE IMPROVEMENT SUMMARY\n"
        summary_text += "=" * 40 + "\n\n"
        
        total_improvements = []
        for dataset in datasets:
            if results_dict[dataset] is not None:
                orig_acc = results_dict[dataset]['original']['accuracy']
                enh_acc = results_dict[dataset]['enhanced']['accuracy']
                improvement = enh_acc - orig_acc
                total_improvements.append(improvement)
                
                summary_text += f"{dataset.upper()}:\n"
                summary_text += f"  Original: {orig_acc:.1%}\n"
                summary_text += f"  Enhanced: {enh_acc:.1%}\n"
                summary_text += f"  Change: {improvement:+.1%}\n\n"
        
        if total_improvements:
            avg_improvement = np.mean(total_improvements)
            summary_text += f"AVERAGE IMPROVEMENT: {avg_improvement:+.1%}\n"
            
            if avg_improvement > 0:
                summary_text += "\n✅ ENHANCED MODEL PERFORMS BETTER"
            elif avg_improvement < -0.01:
                summary_text += "\n⚠️  ORIGINAL MODEL PERFORMS BETTER"
            else:
                summary_text += "\n➖ PERFORMANCE SIMILAR"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Saved comparison visualization: {save_prefix}.png")
    
    def save_detailed_results(self, results_dict, predictions_dict):
        """Save detailed results to Excel files."""
        print("💾 Saving detailed results...")
        
        # Create summary DataFrame
        summary_data = []
        
        for dataset_name, results in results_dict.items():
            if results is not None:
                for model_name, metrics in results.items():
                    summary_data.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score'],
                        'avg_confidence': metrics['avg_confidence'],
                        'confidence_std': metrics['confidence_std'],
                        'high_conf_accuracy': metrics['high_conf_accuracy'],
                        'n_high_conf': metrics['n_high_conf']
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel('enhanced_model_test_results.xlsx', index=False)
        
        print("✓ Saved summary results to: enhanced_model_test_results.xlsx")
        
        # Save individual predictions if needed
        for dataset_name, preds in predictions_dict.items():
            if preds is not None:
                pred_df = pd.DataFrame({
                    'original_predictions': preds['original']['predictions'],
                    'enhanced_predictions': preds['enhanced']['predictions'],
                    'original_confidence': preds['original']['confidence'],
                    'enhanced_confidence': preds['enhanced']['confidence']
                })
                pred_df.to_excel(f'{dataset_name}_predictions.xlsx', index=False)
                print(f"✓ Saved {dataset_name} predictions to: {dataset_name}_predictions.xlsx")

def main():
    """Main function to test enhanced model on validation datasets."""
    print("🧪 Testing Enhanced Model on Validation Datasets")
    print("=" * 60)
    
    try:
        # Initialize tester
        tester = EnhancedModelTester()
        
        # Load models
        tester.load_models()
        
        # Load validation data
        val3_df, val6_df, combined_val_df = tester.load_validation_data()
        
        # Test on different datasets
        datasets = {
            'val3': val3_df,
            'val6': val6_df,
            'combined': combined_val_df
        }
        
        results_dict = {}
        predictions_dict = {}
        
        for dataset_name, dataset_df in datasets.items():
            print(f"\n🎯 Testing on {dataset_name}...")
            
            try:
                # Preprocess data
                X_selected, X_raw, y_true, processed_df = tester.preprocess_validation_data(dataset_df)
                
                # Create enhanced features
                X_enhanced, cluster_labels, error_indicator = tester.create_enhanced_features(X_selected)
                
                # Make predictions
                predictions = tester.make_predictions(X_selected, X_enhanced)
                
                # Evaluate performance
                results = tester.evaluate_performance(y_true, predictions, dataset_name)
                
                # Store results
                results_dict[dataset_name] = results
                predictions_dict[dataset_name] = predictions
                
                # Print cluster analysis
                print(f"  Cluster analysis:")
                print(f"    Cluster 0 (high-error): {np.sum(cluster_labels == 0)} samples")
                print(f"    Cluster 1 (low-error): {np.sum(cluster_labels == 1)} samples")
                print(f"    Error-prone samples: {np.sum(error_indicator)} samples")
                
            except Exception as e:
                print(f"❌ Error processing {dataset_name}: {e}")
                results_dict[dataset_name] = None
                predictions_dict[dataset_name] = None
        
        # Create visualizations (DISABLED - use create_validation_visualizations.py for pure plots)
        # tester.create_comparison_visualizations(results_dict, predictions_dict)
        print("📊 Skipping comparison visualizations (use create_validation_visualizations.py for pure validation plots)")
        
        # Save detailed results
        tester.save_detailed_results(results_dict, predictions_dict)
        
        print(f"\n✅ Enhanced model testing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 