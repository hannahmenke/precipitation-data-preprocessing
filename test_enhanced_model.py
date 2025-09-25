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
    
    def __init__(self, output_dir="models"):
        self.models_dir = Path(output_dir)
        self.original_model = None
        self.enhanced_model = None
        self.cluster_model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.enhanced_models = {}  # Store concentration-specific models
        self.cluster_models = {}   # Store concentration-specific cluster models
        
    def load_models(self):
        """Load all model components."""
        print("📂 Loading model components...")

        # Load original model components
        self.original_model = joblib.load(self.models_dir / "latest_model.joblib")
        self.scaler = joblib.load(self.models_dir / "latest_scaler.joblib")
        self.feature_selector = joblib.load(self.models_dir / "latest_feature_selector.joblib")
        self.feature_names = joblib.load(self.models_dir / "latest_feature_names.joblib")

        # Try to load concentration-specific models first
        concentration_models = {}
        cluster_models = {}

        for concentration in ['_3mM', '_6mM']:
            enhanced_path = self.models_dir / f"enhanced_ensemble_model{concentration}.joblib"
            cluster_path = self.models_dir / f"cluster_model{concentration}.joblib"

            if enhanced_path.exists() and cluster_path.exists():
                concentration_models[concentration] = joblib.load(enhanced_path)
                cluster_models[concentration] = joblib.load(cluster_path)
                print(f"✓ Loaded {concentration} enhanced ensemble model")

        # If we found concentration-specific models, use them
        if concentration_models:
            self.enhanced_models = concentration_models
            self.cluster_models = cluster_models
            print(f"✓ Using {len(concentration_models)} concentration-specific models")
        else:
            # Fall back to single model
            try:
                self.enhanced_model = joblib.load(self.models_dir / "enhanced_ensemble_model.joblib")
                self.cluster_model = joblib.load(self.models_dir / "cluster_model.joblib")
                print("✓ Loaded single enhanced ensemble model")
                print("✓ Loaded single cluster model")
            except FileNotFoundError:
                print("❌ No enhanced models found (neither concentration-specific nor single)")
                raise

        print("✓ Loaded original model components")

    def get_concentration_suffix(self, experiment_name):
        """Determine concentration suffix from experiment name."""
        if '3mM' in experiment_name or '3mm' in experiment_name.lower():
            return '_3mM'
        elif '6mM' in experiment_name or '6mm' in experiment_name.lower():
            return '_6mM'
        else:
            # Default fallback - try to infer from patterns
            if experiment_name.startswith('val3') or 'val3' in experiment_name:
                return '_3mM'
            elif experiment_name.startswith('val6') or 'val6' in experiment_name:
                return '_6mM'
            return None  # Unknown concentration

    def get_model_for_experiment(self, experiment_name):
        """Get the appropriate enhanced and cluster models for an experiment."""
        if self.enhanced_models:  # Using concentration-specific models
            concentration = self.get_concentration_suffix(experiment_name)
            if concentration and concentration in self.enhanced_models:
                return self.enhanced_models[concentration], self.cluster_models[concentration]
            else:
                # If concentration is unknown, try both and use first available
                for conc in ['_3mM', '_6mM']:
                    if conc in self.enhanced_models:
                        print(f"⚠️  Unknown concentration for {experiment_name}, using {conc} model")
                        return self.enhanced_models[conc], self.cluster_models[conc]
                raise ValueError(f"No suitable model found for experiment {experiment_name}")
        else:
            # Using single model
            return self.enhanced_model, self.cluster_model

    def load_validation_data(self):
        """Load and preprocess validation datasets from training experiment splits."""
        print("\n📊 Loading validation datasets...")

        # Discover test/val files from training experiments
        # Determine network directory from model metadata
        network_dir_name = "first_network"  # default
        if "third_network" in str(self.models_dir):
            network_dir_name = "third_network"

        network_dir = Path('training_data') / network_dir_name
        if not network_dir.exists():
            print(f"⚠️  Network directory not found: {network_dir}")
            return None, None, None

        all_val_files = []
        experiment_names = []

        # Collect test and val files from each experiment
        for exp_dir in network_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                split_data_dir = exp_dir / "split_data_tables"
                if split_data_dir.exists():
                    test_file = split_data_dir / "test.xlsx"
                    val_file = split_data_dir / "val.xlsx"

                    if test_file.exists():
                        all_val_files.append(test_file)
                        experiment_names.append(f"{exp_dir.name}_test")
                    if val_file.exists():
                        all_val_files.append(val_file)
                        experiment_names.append(f"{exp_dir.name}_val")

        if not all_val_files:
            print(f"⚠️  No test/val files found in {network_dir}")
            return None, None, None

        print(f"📁 Found {len(all_val_files)} validation files from training experiments:")
        
        # Check if we should use improved features based on training metadata
        metadata_path = self.models_dir / 'enhanced_model_metadata.joblib'
        use_improved_features = False
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            train_features = metadata.get('feature_columns', [])
            # Check if any improved features were used during training
            improved_feature_names = ['Area_Perimeter_Ratio', 'Class4_Discriminator', 'Comprehensive_Shape',
                                      'Convex_Efficiency', 'Distance_Interaction', 'Shape_Complexity',
                                      'Eccentricity_Robust', 'Distance_Ratio', 'Intensity_Stability']
            use_improved_features = any(feat in train_features for feat in improved_feature_names)
            print(f"ℹ️  Training used {'improved' if use_improved_features else 'original'} features")

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

        for val_file, exp_name in zip(all_val_files, experiment_names):
            try:
                # Load original data
                df = pd.read_excel(val_file)
                dataset_name = exp_name  # Use experiment name instead of file stem

                # Apply feature engineering conditionally
                if use_improved_features:
                    df_processed = create_improved_features(df)
                    feature_status = "with improved features"
                else:
                    df_processed = df.copy()
                    feature_status = "with original features"

                df_processed['source'] = dataset_name

                all_datasets[dataset_name] = df_processed
                improved_datasets.append(df_processed)
                total_samples += len(df)

                print(f"  📄 {dataset_name}: {len(df)} samples ({feature_status})")
                
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
        
        # Print breakdown by dataset
        print(f"\n📊 Dataset breakdown:")
        for dataset_name in all_datasets:
            count = len(all_datasets[dataset_name])
            percentage = (count / total_samples) * 100
            print(f"  {dataset_name}: {count} samples ({percentage:.1f}%)")

        # Return all datasets (not just val3/val6)
        return all_datasets, combined_val_df
    
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

        # Check what features the ORIGINAL model needs (for comparison predictions)
        original_metadata_path = Path('third_network_combined/models/latest_metadata.joblib')
        enhanced_metadata_path = Path('third_network_combined/models/enhanced_model_metadata.joblib')

        original_features = []
        enhanced_features = []

        if original_metadata_path.exists():
            orig_metadata = joblib.load(original_metadata_path)
            original_features = orig_metadata.get('feature_names', [])
            print(f"ℹ️  Original model needs {len(original_features)} features (with improvements)")

        if enhanced_metadata_path.exists():
            enh_metadata = joblib.load(enhanced_metadata_path)
            enhanced_features = enh_metadata.get('feature_columns', [])
            print(f"ℹ️  Enhanced model needs {len(enhanced_features)} features (original only)")

        # Create improved features if original model needs them
        if original_features:
            # Check if any improved features are needed
            improved_feature_names = ['Area_Perimeter_Ratio', 'Class4_Discriminator', 'Comprehensive_Shape',
                                      'Convex_Efficiency', 'Distance_Interaction', 'Shape_Complexity',
                                      'Eccentricity_Robust', 'Distance_Ratio', 'Intensity_Stability']
            needs_improved = any(feat in original_features for feat in improved_feature_names)

            if needs_improved:
                # Create improved features for original model
                val_df = self.create_improved_features_for_original(val_df)
                print("ℹ️  Created improved features for original model")

        # Prepare features for enhanced model (17 original features)
        enhanced_common_features = [feat for feat in available_features if feat in enhanced_features]
        X_raw_enhanced = val_df[enhanced_common_features].fillna(val_df[enhanced_common_features].median())
        X_scaled_enhanced = self.scaler.transform(X_raw_enhanced)
        X_selected_enhanced = self.feature_selector.transform(X_scaled_enhanced)

        # Prepare features for original model (20 features with improvements)
        if original_features:
            original_common_features = [feat for feat in val_df.columns
                                      if feat in original_features and feat not in exclude_cols]
            X_raw_original = val_df[original_common_features].fillna(val_df[original_common_features].median())
        else:
            X_raw_original = X_raw_enhanced

        print(f"✓ Enhanced model: {len(enhanced_common_features)} features")
        print(f"✓ Original model: {len(original_common_features) if original_features else len(enhanced_common_features)} features")

        # Get true labels
        y_true = val_df['type'] if 'type' in val_df.columns else None

        return X_selected_enhanced, X_raw_original, y_true, val_df

    def create_improved_features_for_original(self, df):
        """Create improved features for the original model (same as in improve_features.py)."""
        df_improved = df.copy()

        # 1. Shape Complexity Score
        if 'Extent' in df.columns and 'Circularity' in df.columns:
            extent_norm = (df['Extent'] - df['Extent'].min()) / (df['Extent'].max() - df['Extent'].min() + 1e-6)
            circularity_norm = (df['Circularity'] - df['Circularity'].min()) / (df['Circularity'].max() - df['Circularity'].min() + 1e-6)
            df_improved['Shape_Complexity'] = extent_norm * (1 - circularity_norm)

        # 2. Normalized Shape Ratios
        if 'Area' in df.columns and 'Perimeter' in df.columns:
            df_improved['Area_Perimeter_Ratio'] = df['Area'] / (df['Perimeter'] + 1e-6)
        if 'Area' in df.columns and 'ConvexArea' in df.columns:
            df_improved['Convex_Efficiency'] = df['Area'] / (df['ConvexArea'] + 1e-6)

        # 3. Robust Eccentricity
        if 'Eccentricity' in df.columns:
            df_improved['Eccentricity_Robust'] = np.clip(df['Eccentricity'], 0, 1)

        # 4. Distance Feature Combinations
        if 'dis' in df.columns and 'dis_normal' in df.columns:
            df_improved['Distance_Ratio'] = df['dis'] / (df['dis_normal'] + 1e-6)
            df_improved['Distance_Interaction'] = df['dis'] * df['dis_normal']

        # 5. Intensity Stability Score
        gray_mean = df['MeanIntensity'] if 'MeanIntensity' in df.columns else df.get('Gray_ave', 0)
        if 'Gray_var' in df.columns:
            df_improved['Intensity_Stability'] = gray_mean / (df['Gray_var'] + 1e-6)

        # 6. Comprehensive Shape Score
        if all(col in df_improved.columns for col in ['Major_Minor_ratio', 'Circularity', 'Eccentricity_Robust']):
            df_improved['Comprehensive_Shape'] = (
                df_improved['Major_Minor_ratio'] *
                df_improved['Circularity'] *
                (1 - df_improved['Eccentricity_Robust'])
            )

        # 7. Class 4 Discriminator
        if all(col in df_improved.columns for col in ['Gray_var', 'Circularity']):
            df_improved['Class4_Discriminator'] = df_improved['Gray_var'] * (1 - df_improved['Circularity'])

        return df_improved

    def create_enhanced_features(self, X_selected, cluster_model):
        """Create enhanced features for the ensemble model."""
        print("🔧 Creating enhanced features...")

        # Handle cluster model - it might be a dict containing multiple models
        if isinstance(cluster_model, dict):
            kmeans_model = cluster_model['kmeans']
            gmm_model = cluster_model['gmm']
            cluster_centers = cluster_model['cluster_centers']
        else:
            # Assume it's a single cluster model
            kmeans_model = cluster_model
            gmm_model = None
            cluster_centers = cluster_model.cluster_centers_

        # Get cluster predictions using KMeans model
        cluster_labels = kmeans_model.predict(X_selected)

        # Create cluster features
        cluster_features = []

        # 1. Distance to cluster centers
        for center in cluster_centers:
            dist = np.sqrt(np.sum((X_selected - center) ** 2, axis=1))
            cluster_features.append(dist)

        # 2. Cluster membership probabilities (soft clustering)
        if gmm_model:
            # Use saved GMM model
            cluster_probs = gmm_model.predict_proba(X_selected)
        else:
            # Fallback: create and fit new GMM
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X_selected)
            cluster_probs = gmm.predict_proba(X_selected)
        for i in range(2):
            cluster_features.append(cluster_probs[:, i])

        # 3. Cluster membership (one-hot encoded) - MISSING FROM ORIGINAL CODE
        for i in range(4):  # 4 clusters from KMeans
            cluster_membership = (cluster_labels == i).astype(float)
            cluster_features.append(cluster_membership)

        # 4. Simplified class distance features (placeholders)
        # In production, you'd store the actual class centers from training
        cluster_features.extend([
            np.zeros(len(X_selected)),  # dist_to_class_4
            np.zeros(len(X_selected)),  # dist_to_class_1
            np.zeros(len(X_selected))   # class_4_vs_1_diff
        ])

        # 5. Error indicator based on cluster membership
        error_indicator = (cluster_labels == 0).astype(float)  # Cluster 0 was high-error
        cluster_features.append(error_indicator)

        # Combine features
        cluster_features_array = np.column_stack(cluster_features)
        X_enhanced = np.column_stack([X_selected, cluster_features_array])

        print(f"✓ Enhanced features shape: {X_enhanced.shape}")

        return X_enhanced, cluster_labels, error_indicator
    
    def make_predictions(self, X_original, X_enhanced, enhanced_model):
        """Make predictions with both models."""
        print("🎯 Making predictions...")

        # Original model predictions (using original features prepared in preprocessing)
        y_pred_original = self.original_model.predict(X_original)
        y_prob_original = self.original_model.predict_proba(X_original)

        # Enhanced model predictions using provided model
        # Handle enhanced model - it might be a dict containing the actual model
        if isinstance(enhanced_model, dict):
            actual_model = enhanced_model['voting_classifier']
        else:
            actual_model = enhanced_model

        y_pred_enhanced = actual_model.predict(X_enhanced)
        y_prob_enhanced = actual_model.predict_proba(X_enhanced)

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
        results_dir = self.models_dir.parent  # Get parent directory (e.g., third_network_combined)
        reports_dir = results_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)  # Ensure reports directory exists

        summary_file = reports_dir / 'enhanced_model_test_results.xlsx'
        summary_df.to_excel(summary_file, index=False)

        print(f"✓ Saved summary results to: {summary_file}")

        # Save individual predictions if needed
        for dataset_name, preds in predictions_dict.items():
            if preds is not None:
                pred_df = pd.DataFrame({
                    'original_predictions': preds['original']['predictions'],
                    'enhanced_predictions': preds['enhanced']['predictions'],
                    'original_confidence': preds['original']['confidence'],
                    'enhanced_confidence': preds['enhanced']['confidence']
                })
                pred_file = reports_dir / f'{dataset_name}_predictions.xlsx'
                pred_df.to_excel(pred_file, index=False)
                print(f"✓ Saved {dataset_name} predictions to: {pred_file}")

def main(output_dir="models"):
    """Main function to test enhanced model on validation datasets."""
    print("🧪 Testing Enhanced Model on Validation Datasets")
    print("=" * 60)

    try:
        # Initialize tester
        tester = EnhancedModelTester(output_dir)
        
        # Load models
        tester.load_models()
        
        # Load validation data
        all_datasets, combined_val_df = tester.load_validation_data()

        if all_datasets is None:
            print("❌ No validation data found")
            return

        # Test on actual experiment datasets + combined
        datasets = dict(all_datasets)  # Copy all experiment datasets
        datasets['combined'] = combined_val_df  # Add combined dataset
        
        results_dict = {}
        predictions_dict = {}
        
        for dataset_name, dataset_df in datasets.items():
            print(f"\n🎯 Testing on {dataset_name}...")

            try:
                # Get appropriate models for this dataset
                enhanced_model, cluster_model = tester.get_model_for_experiment(dataset_name)

                if tester.enhanced_models:  # Using concentration-specific models
                    concentration = tester.get_concentration_suffix(dataset_name)
                    if concentration:
                        print(f"🔧 Using {concentration} concentration-specific model")
                    else:
                        print(f"🔧 Using fallback concentration-specific model for {dataset_name}")
                else:
                    print(f"🔧 Using single combined model for {dataset_name}")

                # Preprocess data
                X_selected, X_original, y_true, processed_df = tester.preprocess_validation_data(dataset_df)

                # Create enhanced features with the appropriate cluster model
                X_enhanced, cluster_labels, error_indicator = tester.create_enhanced_features(X_selected, cluster_model)

                # Make predictions with the appropriate enhanced model
                predictions = tester.make_predictions(X_original, X_enhanced, enhanced_model)

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
    import sys

    # Allow specifying output directory as command line argument or environment variable
    import os
    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('AUTORUN_MODELS_DIR', 'models')
    main(output_dir) 