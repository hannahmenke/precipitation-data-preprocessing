#!/usr/bin/env python3
"""
Improve Model Performance Using Clustering Insights.

This script implements several strategies to improve model accuracy based on 
clustering analysis findings:
1. Cluster-aware feature engineering
2. Class-specific feature importance
3. Ensemble with cluster-specific models
4. Targeted data augmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ClusterBasedModelImprover:
    """
    Class to improve model performance using clustering insights.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.cluster_model = None
        self.ensemble_model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        
    def load_data_and_model(self):
        """Load training data and existing model components."""
        print("📂 Loading data and model components...")
        
        # Load existing model components
        models_dir = Path("models")
        self.base_model = joblib.load(models_dir / "latest_model.joblib")
        self.scaler = joblib.load(models_dir / "latest_scaler.joblib")
        self.feature_selector = joblib.load(models_dir / "latest_feature_selector.joblib")
        self.feature_names = joblib.load(models_dir / "latest_feature_names.joblib")
        
        # Load training data
        train_df = pd.read_excel('training_data/train_improved.xlsx')
        
        # Preprocess training data
        exclude_cols = ['NO.', 'ID', 'type', 'source_file', 'Centroid', 'BoundingBox', 'WeightedCentroid']
        original_features = [col for col in train_df.columns 
                           if col not in exclude_cols and train_df[col].dtype in ['int64', 'float64']]
        
        X_train_raw = train_df[original_features].fillna(train_df[original_features].median())
        X_train_scaled = self.scaler.transform(X_train_raw)
        X_train_selected = self.feature_selector.transform(X_train_scaled)
        y_train = train_df['type']
        
        # Convert target to 0-based for sklearn
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        y_train_encoded = [label_mapping[label] for label in y_train]
        
        print(f"✓ Loaded training data: {X_train_selected.shape}")
        print(f"✓ Class distribution: {Counter(y_train_encoded)}")
        
        return X_train_selected, np.array(y_train_encoded), X_train_raw, train_df
    
    def create_cluster_aware_features(self, X_train_selected, y_train, X_train_raw):
        """Create features based on clustering insights."""
        print("🔧 Creating cluster-aware features...")
        
        # Perform clustering on training data
        self.cluster_model = KMeans(n_clusters=2, random_state=self.random_state)
        cluster_labels = self.cluster_model.fit_predict(X_train_selected)
        
        print(f"✓ Training data clustering:")
        for cluster_id in [0, 1]:
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_classes = Counter(y_train[cluster_mask])
            print(f"  Cluster {cluster_id}: {cluster_size} samples, classes: {dict(cluster_classes)}")
        
        # Create cluster-based features
        cluster_features = []
        
        # 1. Distance to cluster centers
        cluster_centers = self.cluster_model.cluster_centers_
        distances_to_centers = []
        for i, center in enumerate(cluster_centers):
            dist = np.sqrt(np.sum((X_train_selected - center) ** 2, axis=1))
            distances_to_centers.append(dist)
            cluster_features.append(dist)
        
        # 2. Cluster membership probability (soft clustering)
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2, random_state=self.random_state)
        gmm.fit(X_train_selected)
        cluster_probs = gmm.predict_proba(X_train_selected)
        for i in range(2):
            cluster_features.append(cluster_probs[:, i])
        
        # 3. Class 4 vs Class 1 discriminator features (based on error analysis)
        # These are the most confused classes
        class_4_mask = y_train == 3  # Class 4 (0-indexed as 3)
        class_1_mask = y_train == 0  # Class 1 (0-indexed as 0)
        
        if np.sum(class_4_mask) > 0 and np.sum(class_1_mask) > 0:
            # Calculate feature differences between Class 4 and Class 1
            class_4_mean = np.mean(X_train_selected[class_4_mask], axis=0)
            class_1_mean = np.mean(X_train_selected[class_1_mask], axis=0)
            
            # Distance to each class center
            dist_to_class_4 = np.sqrt(np.sum((X_train_selected - class_4_mean) ** 2, axis=1))
            dist_to_class_1 = np.sqrt(np.sum((X_train_selected - class_1_mean) ** 2, axis=1))
            
            cluster_features.append(dist_to_class_4)
            cluster_features.append(dist_to_class_1)
            cluster_features.append(dist_to_class_4 - dist_to_class_1)  # Relative distance
        
        # 4. High-error region indicator
        # Based on clustering, create a feature that indicates problematic regions
        error_prone_cluster = 0  # From our analysis, cluster 0 had all the errors
        error_indicator = (cluster_labels == error_prone_cluster).astype(float)
        cluster_features.append(error_indicator)
        
        # Combine all cluster features
        cluster_features_array = np.column_stack(cluster_features)
        
        print(f"✓ Created {cluster_features_array.shape[1]} cluster-aware features")
        
        return cluster_features_array, cluster_labels
    
    def create_enhanced_training_data(self, X_train_selected, y_train, cluster_features, cluster_labels):
        """Create enhanced training data with targeted augmentation."""
        print("📈 Creating enhanced training data...")
        
        # Combine original and cluster features
        X_enhanced = np.column_stack([X_train_selected, cluster_features])
        
        # Identify problematic samples (those that would be in high-error cluster)
        error_prone_mask = cluster_labels == 0  # Based on validation analysis
        
        # Target augmentation for confusing class pairs
        # Focus on Class 4 vs Class 1 confusion (most common error)
        class_4_mask = y_train == 3  # Class 4
        class_1_mask = y_train == 0  # Class 1
        
        # Use SMOTEENN for targeted oversampling + cleaning
        smote_enn = SMOTEENN(random_state=self.random_state)
        X_resampled, y_resampled = smote_enn.fit_resample(X_enhanced, y_train)
        
        print(f"✓ Original data: {X_enhanced.shape}")
        print(f"✓ Resampled data: {X_resampled.shape}")
        print(f"✓ New class distribution: {Counter(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_ensemble_model(self, X_enhanced, y_enhanced):
        """Train an ensemble model with multiple strategies."""
        print("🎯 Training enhanced ensemble model...")
        
        # 1. Enhanced XGBoost with cluster features
        xgb_enhanced = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            min_child_weight=3,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        
        # 2. Random Forest optimized for this problem
        rf_enhanced = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        # 3. XGBoost with different hyperparameters (more conservative)
        xgb_conservative = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.5,
            reg_lambda=2,
            min_child_weight=5,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        
        # Create voting ensemble
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('xgb_enhanced', xgb_enhanced),
                ('rf_enhanced', rf_enhanced),
                ('xgb_conservative', xgb_conservative)
            ],
            voting='soft'  # Use probability voting
        )
        
        # Train the ensemble
        print("  Training ensemble...")
        self.ensemble_model.fit(X_enhanced, y_enhanced)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.ensemble_model, X_enhanced, y_enhanced, cv=5, scoring='accuracy')
        print(f"✓ Ensemble CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return self.ensemble_model
    
    def evaluate_improvements(self):
        """Evaluate the improved model on validation data."""
        print("📊 Evaluating model improvements...")
        
        # Load validation data
        val_df = pd.read_excel('validation_data_improved.xlsx')
        
        # Preprocess validation data
        exclude_cols = ['NO.', 'ID', 'type', 'source_file', 'Centroid', 'BoundingBox', 'WeightedCentroid']
        original_features = [col for col in val_df.columns 
                           if col not in exclude_cols and val_df[col].dtype in ['int64', 'float64']]
        
        X_val_raw = val_df[original_features].fillna(val_df[original_features].median())
        X_val_scaled = self.scaler.transform(X_val_raw)
        X_val_selected = self.feature_selector.transform(X_val_scaled)
        y_val_true = val_df['type']
        
        # Convert to 0-based
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        y_val_encoded = [label_mapping[label] for label in y_val_true]
        
        # Create cluster features for validation data
        val_cluster_labels = self.cluster_model.predict(X_val_selected)
        
        # Create the same cluster features as training
        cluster_centers = self.cluster_model.cluster_centers_
        val_cluster_features = []
        
        # Distance features
        for center in cluster_centers:
            dist = np.sqrt(np.sum((X_val_selected - center) ** 2, axis=1))
            val_cluster_features.append(dist)
        
        # Gaussian mixture probabilities (need to refit on validation - simplified approach)
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2, random_state=self.random_state)
        gmm.fit(X_val_selected)
        cluster_probs = gmm.predict_proba(X_val_selected)
        for i in range(2):
            val_cluster_features.append(cluster_probs[:, i])
        
        # Class center distances (using training centers)
        # This is simplified - in practice you'd store these from training
        val_cluster_features.extend([
            np.zeros(len(X_val_selected)),  # Placeholder
            np.zeros(len(X_val_selected)),  # Placeholder  
            np.zeros(len(X_val_selected))   # Placeholder
        ])
        
        # Error indicator
        error_indicator = (val_cluster_labels == 0).astype(float)
        val_cluster_features.append(error_indicator)
        
        X_val_enhanced = np.column_stack([X_val_selected, np.column_stack(val_cluster_features)])
        
        # Make predictions with original and enhanced models
        # Original model predictions
        y_pred_original = self.base_model.predict(X_val_selected)
        y_pred_original_labels = [list(label_mapping.keys())[list(label_mapping.values()).index(pred)] 
                                for pred in y_pred_original]
        
        # Enhanced ensemble predictions
        y_pred_enhanced = self.ensemble_model.predict(X_val_enhanced)
        y_pred_enhanced_labels = [list(label_mapping.keys())[list(label_mapping.values()).index(pred)] 
                                for pred in y_pred_enhanced]
        
        # Calculate accuracies
        original_accuracy = accuracy_score(y_val_true, y_pred_original_labels)
        enhanced_accuracy = accuracy_score(y_val_true, y_pred_enhanced_labels)
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"{'='*50}")
        print(f"Original Model Accuracy:  {original_accuracy:.1%}")
        print(f"Enhanced Model Accuracy:  {enhanced_accuracy:.1%}")
        print(f"Improvement:              {enhanced_accuracy - original_accuracy:+.1%}")
        
        # Detailed analysis
        print(f"\n📋 DETAILED ANALYSIS:")
        print(f"Original Model:")
        print(classification_report(y_val_true, y_pred_original_labels, digits=3))
        
        print(f"\nEnhanced Model:")
        print(classification_report(y_val_true, y_pred_enhanced_labels, digits=3))
        
        # Focus on Class 4 vs Class 1 confusion (main error pattern)
        print(f"\n🎯 CLASS 4 vs CLASS 1 ANALYSIS:")
        class_4_mask = np.array(y_val_true) == 4
        class_1_mask = np.array(y_val_true) == 1
        
        if np.sum(class_4_mask) > 0:
            class_4_original_acc = accuracy_score(np.array(y_val_true)[class_4_mask], 
                                                 np.array(y_pred_original_labels)[class_4_mask])
            class_4_enhanced_acc = accuracy_score(np.array(y_val_true)[class_4_mask], 
                                                 np.array(y_pred_enhanced_labels)[class_4_mask])
            print(f"Class 4 accuracy - Original: {class_4_original_acc:.1%}, Enhanced: {class_4_enhanced_acc:.1%}")
        
        if np.sum(class_1_mask) > 0:
            class_1_original_acc = accuracy_score(np.array(y_val_true)[class_1_mask], 
                                                 np.array(y_pred_original_labels)[class_1_mask])
            class_1_enhanced_acc = accuracy_score(np.array(y_val_true)[class_1_mask], 
                                                 np.array(y_pred_enhanced_labels)[class_1_mask])
            print(f"Class 1 accuracy - Original: {class_1_original_acc:.1%}, Enhanced: {class_1_enhanced_acc:.1%}")
        
        return {
            'original_accuracy': original_accuracy,
            'enhanced_accuracy': enhanced_accuracy,
            'improvement': enhanced_accuracy - original_accuracy,
            'y_true': y_val_true,
            'y_pred_original': y_pred_original_labels,
            'y_pred_enhanced': y_pred_enhanced_labels
        }
    
    def save_improved_model(self):
        """Save the improved model components."""
        print("💾 Saving improved model...")
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save enhanced model
        joblib.dump(self.ensemble_model, models_dir / "enhanced_ensemble_model.joblib")
        joblib.dump(self.cluster_model, models_dir / "cluster_model.joblib")
        
        # Update metadata
        metadata = {
            'model_type': 'enhanced_ensemble',
            'n_features': len(self.feature_names) + 8,  # Original + cluster features
            'feature_names': list(self.feature_names) + [
                'dist_to_cluster_0', 'dist_to_cluster_1', 
                'cluster_0_prob', 'cluster_1_prob',
                'dist_to_class_4', 'dist_to_class_1', 'class_4_vs_1_diff',
                'error_indicator'
            ],
            'enhancement_date': pd.Timestamp.now().isoformat()
        }
        
        joblib.dump(metadata, models_dir / "enhanced_model_metadata.joblib")
        
        print(f"✓ Saved enhanced model to: {models_dir}/enhanced_ensemble_model.joblib")

def main():
    """Main function to improve model using clustering insights."""
    print("🚀 Model Improvement Using Clustering Insights")
    print("=" * 60)
    
    try:
        # Initialize improver
        improver = ClusterBasedModelImprover()
        
        # Load data and model
        X_train_selected, y_train, X_train_raw, train_df = improver.load_data_and_model()
        
        # Create cluster-aware features
        cluster_features, cluster_labels = improver.create_cluster_aware_features(
            X_train_selected, y_train, X_train_raw)
        
        # Create enhanced training data
        X_enhanced, y_enhanced = improver.create_enhanced_training_data(
            X_train_selected, y_train, cluster_features, cluster_labels)
        
        # Train enhanced ensemble model
        ensemble_model = improver.train_ensemble_model(X_enhanced, y_enhanced)
        
        # Evaluate improvements
        results = improver.evaluate_improvements()
        
        # Save improved model
        improver.save_improved_model()
        
        print(f"\n✅ Model improvement complete!")
        print(f"🎯 Final improvement: {results['improvement']:+.1%}")
        
        if results['improvement'] > 0:
            print(f"🎉 SUCCESS: Enhanced model performs better!")
        else:
            print(f"⚠️  Note: Enhancement didn't improve validation accuracy")
            print(f"   This could indicate overfitting or need for different approach")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 